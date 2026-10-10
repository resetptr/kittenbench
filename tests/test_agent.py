"""Tests for the LLM harness (run_agent.py), with no API calls."""

import types

import run_agent
from kittenbench.env import KittenEnv

# Qwen 3.5 9B's reasoning on turn 2 of a real run: five tool calls written as text, no structured calls.
QWEN_TURN_2 = """Let me check inventory and order supplies.

<tool_call>
<function=check_inventory>
</function>
</tool_call>
<tool_call>
<function=shop>
<parameter=item>
kibble_bag
</parameter>
<parameter=quantity>
1
</parameter>
</function>
</tool_call>
<tool_call>
<function=shop>
<parameter=item>
food_bowl
</parameter>
<parameter=quantity>
1
</parameter>
</function>
</tool_call>
<tool_call>
<function=shop>
<parameter=item>
water_bowl
</parameter>
<parameter=quantity>
1
</parameter>
</function>
</tool_call>
<tool_call>
<function=shop>
<parameter=item>
litter_box
</parameter>
<parameter=quantity>
1
</parameter>
</function>
</tool_call>"""

EXPECTED = [("check_inventory", {}), ("shop", {"item": "kibble_bag", "quantity": 1}),
            ("shop", {"item": "food_bowl", "quantity": 1}), ("shop", {"item": "water_bowl", "quantity": 1}),
            ("shop", {"item": "litter_box", "quantity": 1})]


def message(content=None, reasoning=None):
    return types.SimpleNamespace(content=content, reasoning=reasoning, tool_calls=None)


def test_every_text_tool_call_in_the_reply_is_recovered_in_order():
    assert run_agent.recover_text_tool_calls(message(content=QWEN_TURN_2)) == EXPECTED
    assert run_agent.recover_text_tool_calls(message(content=QWEN_TURN_2, reasoning=QWEN_TURN_2)) == EXPECTED


def test_calls_only_in_reasoning_are_not_run():
    """The reasoning holds drafts; calls written there never run."""
    assert run_agent.recover_text_tool_calls(message(reasoning=QWEN_TURN_2)) == []


def test_recovered_calls_all_run():
    """All five recovered calls run in one turn: four orders go through."""
    env = KittenEnv(seed=0, days=2)
    env.reset()
    calls = [(f"c{i}", name, args) for i, (name, args) in enumerate(run_agent.recover_text_tool_calls(message(content=QWEN_TURN_2)))]
    actions, _ = run_agent.play_turn(env, calls, recovered=True)
    ran = [a["call"]["name"] for a in actions if a["ran"] and not a.get("auto")]
    assert ran == ["check_inventory", "shop", "shop", "shop", "shop"]
    assert env.metrics.spent == 25 + 8 + 8 + 30


def test_structured_calls_all_run():
    """Several real tool calls in one turn all run, in order."""
    env = KittenEnv(seed=0, days=2)
    env.reset()
    calls = [("a", "go_to", {"room": "kitchen"}), ("b", "use", {"verb": "fill", "object": "water_bowl"}),
             ("c", "look", {})]
    actions, results = run_agent.play_turn(env, calls, recovered=False)
    assert [a["call"]["name"] for a in actions if not a.get("auto")] == ["go_to", "use", "look"]
    assert all(a["ran"] for a in actions) and len(results) == 3
    assert env.world.objects["water_bowl"].props["ml"] > 0


# ── context windows and prompt caching ─────────────────────────

def fake_turns(n):
    """n stored turns, each: a reply with reasoning and one tool call, then its result."""
    turns = []
    for i in range(n):
        reply = run_agent.stored_reply(message(content=f"said {i}", reasoning=f"thought {i}"),
                                       [{"id": f"c{i}", "type": "function", "function": {"name": "look", "arguments": "{}"}}])
        turns.append([reply, {"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"}])
    return turns


def plain(messages):
    """Messages with cache markers removed, to compare what the model reads."""
    return [{**m, "content": run_agent_text(m["content"])} for m in messages]


def run_agent_text(content):
    return "".join(p["text"] for p in content) if isinstance(content, list) else content


def test_windows_move_in_steps():
    assert run_agent.window_starts(5) == (0, 0)
    assert run_agent.window_starts(25) == (0, 10)      # reasoning for the last 15 turns
    assert run_agent.window_starts(30) == (0, 20)      # ... then the last 10
    assert run_agent.window_starts(151) == (25, 140)   # 126 turns shown
    assert run_agent.window_starts(175) == (50, 160)   # 125 turns shown


def test_reasoning_only_shown_for_recent_turns():
    messages = run_agent.build_messages("intro", fake_turns(25))
    replies = [m["content"] for m in messages if m["role"] == "assistant"]
    assert all("thought" not in r for r in replies[:10]) and all("said" in r for r in replies[:10])
    assert all("(my reasoning) thought" in r for r in replies[10:])
    assert not any(k.startswith("_") for m in messages for k in m)   # private fields never sent


def test_context_start_stays_the_same_between_steps():
    """Between window steps each turn's context starts with the previous turn's context (so it can be cached)."""
    for n in (12, 23, 140, 160):
        before = plain(run_agent.build_messages("intro", fake_turns(n), "qwen/x"))
        after = plain(run_agent.build_messages("intro", fake_turns(n + 1), "qwen/x"))
        assert after[:len(before)] == before, n


def test_cache_markers_only_where_needed():
    qwen = run_agent.build_messages("intro", fake_turns(25), "qwen/qwen3.8-2.4t-a95b")
    marked = [i for i, m in enumerate(qwen) if isinstance(m["content"], list)]
    assert 0 in marked and len(marked) <= 4
    assert qwen[marked[-1]]["role"] == "assistant" and marked[-1] >= len(qwen) - 2   # the latest reply
    for model in ("anthropic/claude-sonnet-5.5", "openai/gpt-6-sol"):
        assert not any(isinstance(m["content"], list) for m in run_agent.build_messages("intro", fake_turns(25), model))
    assert run_agent.cache_request_options("anthropic/claude-sonnet-5.5") == {"cache_control": {"type": "ephemeral"}}
    assert run_agent.cache_request_options("openai/gpt-6-sol") == {}


def test_reply_with_nothing_visible_is_not_empty():
    """A no-action reply whose reasoning is hidden still has text (providers reject empty assistant messages)."""
    stored = run_agent.stored_reply(message(content=None, reasoning="thinking"))
    assert run_agent.render_reply(stored, show_reasoning=False)["content"] == "(no reply)"
