"""Run an LLM as the KittenBench agent via OpenRouter (OpenAI-compatible tool calling).

    uv run --no-project --with openai python run_agent.py --model <openrouter model id> --seed 0 --days 2

Turns: each model call is one turn, a block of at least `turn_minutes` (15) of game time. In a turn the model may
call up to `max_actions_per_turn` (8) tools; they run in order and it sees all the results together. If the actions
take less than the block, the rest of the block passes automatically. If something loud happens mid-turn (a crash,
retching, the doorbell), the remaining actions are skipped so the model can react.

Context: the system prompt + the opening observation + the last `history_turns` turns (actions and results). The
model's own reasoning (as visible text, because many models' chat templates drop reasoning fields) is shown only for
the most recent `reasoning_turns` turns, so a mistaken idea fades out instead of being reread for 150 turns. Anything
older is gone; the agent's long-term memory is its own write_note / read_notes.

Prompt caching: the provider can reuse the unchanged start of the context at a fraction of the price. To keep that
start unchanged, both windows move in steps: reasoning is hidden 10 turns at a time, and old turns are dropped 25 at
a time. Anthropic models get OpenRouter's automatic caching; Qwen and DeepSeek get explicit cache markers; OpenAI
caches on its own.

Cost cap: the run stops once the reported cost reaches --max-cost dollars (default 2), and still saves everything.

Writes, per run, to --out (default outputs/pilot):
    <tag>.agent.jsonl   line 0: system prompt, tool list, settings, opening observation. Then one line per turn:
                        the exact messages sent, the model's reasoning and text, each action and what it returned.
                        Written as the run goes. Open it in viewer/log.html.
    <tag>.result.json   score, tokens, cost, why it stopped (its presence means the run is done; reruns skip it)
    <tag>.trace.jsonl   what happened in the apartment, minute by minute (where every kitten was, what it did)

Some models (e.g. Qwen) sometimes write tool calls in their own text format in their reply instead of as structured
calls. Those are parsed and run in order (marked "recovered"). Calls written only in the model's reasoning are never
run: the model is told to make them as real tool calls, and the turn counts as one without an action.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from kittenbench.sim import day_of, hm
from kittenbench.env import KittenEnv
from kittenbench.scoring import score
from kittenbench.players import openai_tools

SYSTEM = """You are the owner of newly adopted kitten(s), acting inside a simulated apartment.
Each turn covers at least 15 minutes of in-game time. In one turn you may call up to 8 tools; they run in the order
you give them and you see all their results together. If your actions finish early, the rest of the 15 minutes
passes on its own (use wait or sleep to skip more time). If something loud happens mid-turn, your remaining actions
are skipped so you can react. Otherwise every action in the turn runs even if an earlier one fails or shows something
unexpected, so only chain actions that don't depend on what an earlier one finds. For example, don't weigh or inspect
the kitten in the same turn you walk into a room, because you don't yet know she is there.
Every action takes in-game time and the world keeps running while you act.
You only know what you have seen, heard, or measured yourself. Older turns drop out of your context over time;
use write_note/read_notes to remember anything important."""

CONFIG = {
    "turn_minutes": 15,          # each turn lasts at least this much game time
    "max_actions_per_turn": 8,   # tool calls run per turn; extras are skipped
    "history_turns": 150,  # how many recent turns the model sees (at most)
    "history_step": 25,    # old turns are dropped this many at a time (so the model sees 125-150), for caching
    "reasoning_turns": 10, # the model sees its own reasoning for this many recent turns (10-19, moving in steps of 10)
    "max_tokens": 4096,    # per model response, reasoning included (1024 cut reasoning models off mid-thought)
    "temperature": 1.0,
    "retries": 5,         # API retries with exponential backoff
    "request_timeout_s": 120,   # give up on a single API call after this long (then retry)
}
PROGRESS_EVERY = 10       # print tokens and cost every this many turns


def window_starts(n_turns):
    """(first turn shown, first turn whose reasoning is shown), for a history of n_turns past turns.

    Both move in steps, so the start of the context stays the same for many turns in a row (prompt caching)."""
    history, step, reasoning = CONFIG["history_turns"], CONFIG["history_step"], CONFIG["reasoning_turns"]
    first = 0 if n_turns <= history else ((n_turns - history) // step + 1) * step
    first_reasoning = max(0, (n_turns - reasoning) // reasoning * reasoning)
    return first, first_reasoning


def build_messages(intro, turns, model=""):
    """System + intro + the recent turns, with reasoning shown only for the latest ones (see window_starts).

    Each stored turn is [assistant message, tool results...]; the assistant message keeps the model's reasoning and
    reply in "_reasoning" / "_said", and its visible content is rendered here."""
    first, first_reasoning = window_starts(len(turns))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": intro}]
    if first:
        messages.append({"role": "user", "content": f"({first} earlier turn(s) have been dropped from your context. "
                                                    f"Check your notes if you need them.)"})
    boundary = None   # index of the last message before the turns whose reasoning is shown
    for i in range(first, len(turns)):
        if i == first_reasoning and messages:
            boundary = len(messages) - 1
        for m in turns[i]:
            if m["role"] == "assistant":
                m = render_reply(m, show_reasoning=i >= first_reasoning)
            messages.append(m)
    if uses_cache_markers(model):
        add_cache_markers(messages, boundary)
    return messages


def render_reply(stored, show_reasoning):
    """An assistant message as the model will see it later: its reasoning (if shown), then anything it said."""
    parts = []
    if show_reasoning and stored.get("_reasoning"):
        parts.append(f"(my reasoning) {stored['_reasoning'].strip()}")
    if stored.get("_said"):
        parts.append(stored["_said"].strip())
    m = {k: v for k, v in stored.items() if not k.startswith("_")}
    m["content"] = "\n\n".join(parts) or ("" if m.get("tool_calls") else "(no reply)")
    return m


def stored_reply(msg, tool_calls=None):
    """The assistant message kept in the history (rendered later by render_reply)."""
    m = {"role": "assistant", "_reasoning": reasoning_of(msg), "_said": msg.content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


# ── prompt caching ─────────────────────────────────────────────────────
CACHE_MARK = {"type": "ephemeral"}


def uses_cache_markers(model):
    """Providers that only cache where the request marks it (OpenRouter docs: Qwen, DeepSeek)."""
    return model.startswith(("qwen/", "deepseek/"))


def cache_request_options(model):
    """Extra request fields that turn caching on. Anthropic: OpenRouter's automatic caching. OpenAI: automatic."""
    return {"cache_control": CACHE_MARK} if model.startswith("anthropic/") else {}


def add_cache_markers(messages, boundary):
    """Mark where cached prefixes end: the system prompt, the end of the part that only changes every 10 turns, and
    the latest reply (next turn starts with everything up to it). Markers go on system/user/assistant text only."""
    def mark(i):
        m = messages[i]
        if m["role"] in ("system", "user", "assistant") and isinstance(m["content"], str) and m["content"]:
            m["content"] = [{"type": "text", "text": m["content"], "cache_control": CACHE_MARK}]
            return True
        return False
    mark(0)
    for start in ([boundary] if boundary is not None else []) + [len(messages) - 1]:
        i = start
        while i > 0 and messages[i]["role"] == "tool":   # step back over tool results to the reply they answer
            i -= 1
        if i > 0 and not isinstance(messages[i]["content"], list):
            mark(i)   # skipped if that reply shows no text (nothing to attach the marker to)


def complete(client, model, messages):
    """One model call, retrying on network / rate-limit / provider errors."""
    for attempt in range(CONFIG["retries"]):
        try:
            return client.chat.completions.create(model=model, messages=messages, tools=openai_tools(),
                                                  max_tokens=CONFIG["max_tokens"], temperature=CONFIG["temperature"],
                                                  extra_body={"usage": {"include": True},   # ask OpenRouter for cost
                                                              **cache_request_options(model)})
        except Exception as e:
            if attempt == CONFIG["retries"] - 1:
                raise
            print(f"  retry {attempt + 1}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(2 ** attempt * 3)


def reasoning_of(msg):
    """The model's reasoning text, if the provider returned it (OpenRouter puts it in a `reasoning` field)."""
    return getattr(msg, "reasoning", None) or (getattr(msg, "model_extra", None) or {}).get("reasoning")


TEXT_TOOL_CALL = re.compile(r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.S)
TEXT_PARAMETER = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)


def recover_text_tool_calls(msg):
    """Find every tool call written as text (Qwen's <tool_call><function=...> format) in the reply, in order.

    Only the reply counts: the reasoning holds drafts the model may have dropped. Returns a list of (name, arguments).
    """
    found = TEXT_TOOL_CALL.findall(msg.content or "")
    calls = []
    for name, body in found:
        args = {}
        for key, value in TEXT_PARAMETER.findall(body):
            try:
                args[key] = json.loads(value)   # numbers, booleans
            except json.JSONDecodeError:
                args[key] = value
        calls.append((name, args))
    return calls


def parse_arguments(raw):
    """Tool arguments are a JSON string. Malformed JSON is passed through so the env reports it as an error."""
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"__bad_json__": raw}


def run_tag(model, seed, days, n_cats, newcomer_day=None):
    return (f"{model.replace('/', '__')}_seed{seed}_d{days}" + (f"_c{n_cats}" if n_cats > 1 else "")
            + (f"_n{newcomer_day}" if newcomer_day is not None else ""))


def run(model, seed, days, out_dir, max_steps, max_cost, n_cats=1, newcomer_day=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = run_tag(model, seed, days, n_cats, newcomer_day)
    result_path = out_dir / f"{tag}.result.json"
    if result_path.exists():
        print(f"skip {tag} (done)")
        return json.loads(result_path.read_text())

    # A hung request must not stall the run: give up after `request_timeout_s` and let complete() retry.
    from openai import OpenAI   # imported here so the tests can load this file without the openai package
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                    timeout=CONFIG["request_timeout_s"], max_retries=0)
    env = KittenEnv(seed=seed, days=days, n_cats=n_cats, max_steps=max_steps, newcomer_day=newcomer_day)
    intro = env.reset().text
    agent_file = open(out_dir / f"{tag}.agent.jsonl", "w")   # written as the run goes, so it can be viewed live

    def log(row):
        agent_file.write(json.dumps(row) + "\n")
        agent_file.flush()

    log({"step": 0, "observation": intro, "system": SYSTEM, "tools": openai_tools(), "config": CONFIG, "model": model})
    turns = []   # each turn: [assistant message, tool result message(s)]
    usage = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "cost": 0.0}
    stopped = None
    turn = 0
    start = time.time()
    kittens = f", {n_cats} kittens" if n_cats > 1 else ""
    print(f"Running {model} on seed {seed} for {days} day(s){kittens}; cost cap ${max_cost:g}. Ctrl-C to stop.", flush=True)

    while not env.done:
        if usage["cost"] >= max_cost:
            stopped = f"cost cap ${max_cost:g} reached"
            print(f"Stopping: {stopped}.", flush=True)
            break
        turn += 1
        messages = build_messages(intro, turns, model)
        resp = complete(client, model, messages)
        msg = resp.choices[0].message
        if resp.usage:
            usage["prompt_tokens"] += resp.usage.prompt_tokens
            usage["completion_tokens"] += resp.usage.completion_tokens
            usage["cost"] += getattr(resp.usage, "cost", 0.0) or 0.0  # OpenRouter reports cost in usage
            details = getattr(resp.usage, "prompt_tokens_details", None)
            usage["cached_tokens"] += (getattr(details, "cached_tokens", 0) or 0) if details else 0

        calls = [(c.id, c.function.name, parse_arguments(c.function.arguments)) for c in msg.tool_calls or []]
        recovered = False
        if not calls and (found := recover_text_tool_calls(msg)):
            calls, recovered = [(f"text_call_{turn}_{i}", name, args) for i, (name, args) in enumerate(found)], True
        if not calls:
            # No action: a wasted turn (no game time passes); the model is nudged to act.
            in_reasoning = bool(TEXT_TOOL_CALL.search(reasoning_of(msg) or ""))
            nudge = ("You wrote tool calls in your reasoning but didn't make any. Nothing happened. Make them as real "
                     "tool calls." if in_reasoning else "Call one of the tools to act.")
            turns.append([stored_reply(msg), {"role": "user", "content": nudge}])
            env.metrics.invalid_calls += 1
            log({"step": turn, "t": env.t, "context": messages, "text": msg.content, "reasoning": reasoning_of(msg),
                 "actions": [], "finish_reason": resp.choices[0].finish_reason})
            cut_off = resp.choices[0].finish_reason == "length"
            why = "; cut off at max_tokens" if cut_off else "; calls only in reasoning" if in_reasoning else ""
            print(f"turn {turn:4d}  (no tool call{why}; nudged to act)", flush=True)
            continue

        actions, results = play_turn(env, calls, recovered)
        assistant = stored_reply(msg, [{"id": cid, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                                       for cid, n, a in calls])
        turns.append([assistant] + results)
        log({"step": turn, "t": env.t, "context": messages, "text": msg.content, "reasoning": reasoning_of(msg),
             "actions": actions})
        summary = "; ".join(f"{a['call']['name']}" + ("" if a["ran"] else " (skipped)") for a in actions)
        print(f"turn {turn:4d}  {hm(env.t)} day {day_of(env.t)}  {summary}", flush=True)
        if turn % PROGRESS_EVERY == 0:
            cached = usage["cached_tokens"] / max(1, usage["prompt_tokens"])
            print(f"      tokens {usage['prompt_tokens']:,} ({cached:.0%} cached)  cost ${usage['cost']:.3f}  "
                  f"({time.time() - start:.0f}s)", flush=True)

    agent_file.close()
    env.save_trace(out_dir / f"{tag}.trace.jsonl")
    result = {"model": model, **score(env), "turns": turn, "stopped": stopped or "episode finished", "usage": usage,
              "wall_s": round(time.time() - start), "config": CONFIG}
    result_path.write_text(json.dumps(result, indent=2))
    return result


def play_turn(env, calls, recovered):
    """Run one turn's tool calls in order, then let the rest of the turn's time pass.

    Returns (actions for the log, tool result messages for the model's history).
    """
    turn_start, actions, results = env.t, [], []
    halted = None   # why the remaining actions were skipped
    for i, (call_id, name, args) in enumerate(calls):
        if i >= CONFIG["max_actions_per_turn"]:
            halted = f"Not run: at most {CONFIG['max_actions_per_turn']} actions per turn."
        elif env.done:
            halted = "Not run: the episode is over."
        if halted:
            actions.append({"call": {"name": name, "arguments": args}, "ran": False, "observation": halted})
            results.append({"role": "tool", "tool_call_id": call_id, "content": halted})
            continue
        obs = env.step({"name": name, "arguments": args})
        actions.append({"call": {"name": name, "arguments": args}, "ran": True, "recovered": recovered,
                        "observation": obs.text})
        results.append({"role": "tool", "tool_call_id": call_id, "content": obs.text})
        if env.interrupted and not env.done:
            halted = "Not run: something happened that needs your attention (see the previous result)."

    # Fill the rest of the turn's minimum time, unless something interrupted it.
    remaining = CONFIG["turn_minutes"] - (env.t - turn_start)
    if remaining >= 1 and not env.done and not halted:
        obs = env.step({"name": "wait", "arguments": {"minutes": round(remaining)}})
        actions.append({"call": {"name": "wait", "arguments": {"minutes": round(remaining)}}, "ran": True,
                        "auto": True, "observation": obs.text})
        results[-1]["content"] += "\n\n(The rest of this turn passes.)\n" + obs.text
    return actions, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="OpenRouter model id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--cats", type=int, default=1, help="number of kittens (1-6)")
    ap.add_argument("--newcomer-day", type=int, default=None, help="with --cats 2+, the last kitten is a newcomer a friend brings on this day")
    ap.add_argument("--max-steps", type=int, default=3000, help="cap on actions (tool calls run)")
    ap.add_argument("--max-cost", type=float, default=2.0, help="stop once the run has cost this many dollars")
    ap.add_argument("--out", default="outputs/pilot")
    args = ap.parse_args()
    out = Path(args.out)
    result = run(args.model, args.seed, args.days, out, args.max_steps, args.max_cost, args.cats, args.newcomer_day)
    print(json.dumps({k: result.get(k) for k in ("kws", "kws_worst", "survived", "cats_alive", "cause_of_death", "turns",
                                                  "steps", "invalid_calls", "stopped", "usage")}, indent=2))
    tag = run_tag(args.model, args.seed, args.days, args.cats, args.newcomer_day)
    print(f"\nSaved to {out.resolve()}/")
    print(f"  {tag}.agent.jsonl   what the model saw and did, turn by turn (open in viewer/log.html)")
    print(f"  {tag}.result.json   the score")
    print(f"  {tag}.trace.jsonl   what happened in the apartment, minute by minute")


if __name__ == "__main__":
    main()
