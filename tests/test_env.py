"""Tests for the simulator. Run with: uvx --python 3.13 pytest -q"""

import random

import pytest

from kittenbench.sim import MINUTES_PER_DAY, WAKE_TIME, WORK_END
from kittenbench.cat import LilyToxicosis
from kittenbench.env import KittenEnv
from kittenbench.players import POLICIES
from kittenbench.sim import RULES
from kittenbench.scoring import score
from kittenbench.players import TOOLS

MONDAY = 2


def run(policy, seed, days=14, n_cats=1):
    """Play one full episode with a scripted policy and return the finished env."""
    env = KittenEnv(seed=seed, days=days, n_cats=n_cats)
    obs = env.reset()
    pol = POLICIES[policy](env, seed)
    while not env.done:
        obs = env.step(pol.act(obs.text))
    return env


def test_tools_match_env():
    """Every tool the LLM is told about has an implementation, and vice versa."""
    described = {t["name"] for t in TOOLS}
    implemented = {m[len("tool_"):] for m in dir(KittenEnv) if m.startswith("tool_")}
    assert described == implemented


def test_rules_are_labelled():
    """Every rules.toml section says whether it's sourced (with a URL) or an assumption."""
    for section, r in RULES.items():
        assert r.get("status") in ("sourced", "assumption"), section
        if r["status"] == "sourced":
            assert r.get("url", "").startswith("https://"), section


def test_deterministic():
    """Same seed + same actions = identical scores and an identical kitten timeline."""
    a, b = run("guideline", 4), run("guideline", 4)
    assert score(a) == score(b)
    assert [r for r in a.trace if r["type"] == "cat"] == [r for r in b.trace if r["type"] == "cat"]


def test_no_water_kills_in_days():
    """A kitten given no water dies of dehydration in a few days, not hours or weeks."""
    env = run("neglect", 2)
    assert not env.cats[0].alive and "dehydration" in env.cats[0].cause_of_death
    assert 1.5 < env.cats[0].metrics.death_t / MINUTES_PER_DAY < 5


def test_workday_wait_stops_for_work():
    """A long wait on a weekday morning is cut off when the owner leaves for work, who returns at 18:00."""
    env = KittenEnv(seed=0)
    env.reset()
    while env.t < MONDAY * MINUTES_PER_DAY + WAKE_TIME:
        env.step({"name": "sleep"} if env.t % MINUTES_PER_DAY >= 19 * 60 else {"name": "wait", "arguments": {"minutes": 600}})
    for _ in range(20):  # waits can also be cut short by the kitten crying
        obs = env.step({"name": "wait", "arguments": {"minutes": 600}})
        if "You leave for work" in obs.text:
            break
    assert "You leave for work" in obs.text and env.t == MONDAY * MINUTES_PER_DAY + WORK_END


def test_bad_arguments_are_agent_errors():
    """Wrong argument names or values are reported to the agent, not raised."""
    env = KittenEnv(seed=0)
    env.reset()
    assert "Error: bad arguments" in env.step({"name": "go_to", "arguments": {"where": "kitchen"}}).text
    assert "Error: minutes must be a number" in env.step({"name": "wait", "arguments": {"minutes": "a while"}}).text
    assert env.metrics.invalid_calls == 2


@pytest.mark.parametrize("seed", range(20))
def test_lily_timeline_matches_source(seed):
    """Sampled lily timelines respect the MSPCA stage order and ranges, and early treatment always works."""
    rng = random.Random(seed)
    c = LilyToxicosis.start(0, rng)
    r = RULES["lily"]
    assert all(r["vomit_hours"][0] * 60 <= v for v in c.vomit_times)
    assert c.polyuria_t < c.anuria_t <= r["anuria_hours"][1] * 60
    assert r["death_hours"][0] * 60 <= c.death_t <= (r["death_hours"][1] + 12) * 60
    assert c.treat(r["treat_window_hours"] * 60 - 1, rng) is True


def test_reference_ordering():
    """The simulator rewards care: oracle ~ guideline > minimal > neglect."""
    seeds = range(6)
    kws = {p: sum(score(run(p, s))["kws"] for s in seeds) / len(seeds) for p in ("oracle", "guideline", "minimal", "neglect")}
    assert kws["oracle"] >= kws["guideline"] - 0.02 > kws["minimal"] > kws["neglect"]


def test_meter_reaches_level_on_time():
    """A meter reports exactly when it will cross a level, and treats 'within rounding error' as already there."""
    from kittenbench.sim import Meter
    m = Meter(0.2, rate=0.1, high=1.0)
    assert abs(m.time_to_reach(0, 0.7) - 5.0) < 1e-9
    assert m.time_to_reach(8, 1.0) is None          # already at the cap
    assert Meter(1.0 - 1e-12, rate=0.1, high=1.0).time_to_reach(0, 1.0) is None


def test_pathfinding_respects_closed_doors():
    """The kitten can't walk through a closed door; the owner can."""
    from kittenbench.sim import Apartment
    world = Apartment()
    bedroom, hallway = world.room_center("bedroom"), world.room_center("hallway")
    assert world.path(hallway, bedroom) is not None
    world.objects["bedroom_door"].props["open"] = False
    assert world.path(hallway, bedroom, cat=True) is None
    assert world.path(hallway, bedroom, cat=False) is not None


def test_time_budget_matches_cat_data():
    """Pattern P5/P7: a cared-for kitten is inactive ~85% of the time and sleeps 16-20 h a day."""
    total, inactive, asleep = 0.0, 0.0, 0.0
    for seed in range(4):
        env = run("guideline", seed)
        for r in env.trace:
            if r["type"] == "cat":
                d = r["t1"] - r["t0"]
                total += d
                inactive += d * (r["action"].startswith("sleep") or r["action"] in ("rest", "hide", "groom"))
                asleep += d * r["action"].startswith("sleep")
    assert 0.78 < inactive / total < 0.92
    assert 16 <= 24 * asleep / total <= 20


# ── Several kittens ─────────────────────────────────────────────


def test_multi_cat_deterministic():
    """Three kittens: same seed + same actions = identical scores and timelines."""
    a, b = run("guideline", 2, days=5, n_cats=3), run("guideline", 2, days=5, n_cats=3)
    assert score(a) == score(b)
    assert [r for r in a.trace if r["type"] == "cat"] == [r for r in b.trace if r["type"] == "cat"]


def test_multi_cat_kittens_have_own_metrics_and_names():
    """Each kitten keeps its own measurements, and the score has one row per kitten."""
    env = run("guideline", 1, days=4, n_cats=3)
    names = [c.name for c in env.cats]
    assert len(set(names)) == 3
    assert len({id(c.metrics) for c in env.cats}) == 3
    s = score(env)
    assert s["cats"] == 3 and [c["name"] for c in s["per_cat"]] == names
    assert s["kws_worst"] == min(c["kws"] for c in s["per_cat"])
    assert {r["cat"] for r in env.trace if r["type"] == "cat"} == set(names)


def test_multi_cat_guideline_beats_neglect():
    """The checklist script keeps all three kittens alive; neglect does much worse."""
    for seed in range(3):
        good, bad = score(run("guideline", seed, n_cats=3)), score(run("neglect", seed, n_cats=3))
        assert good["survived"], seed
        assert good["kws_worst"] > bad["kws_worst"], seed


def test_multi_cat_actions_need_a_name():
    """With several kittens, kitten actions must say which kitten; a wrong name is an invalid call."""
    env = KittenEnv(seed=0, days=2, n_cats=2)
    env.reset()
    obs = env.step({"name": "pet", "arguments": {}})
    assert "cat=" in obs.text
    obs = env.step({"name": "pet", "arguments": {"cat": "Garfield"}})
    assert env.metrics.invalid_calls == 2
    env.step({"name": "find_cat", "arguments": {"cat": env.cats[1].name}})
    assert env.metrics.invalid_calls == 2


def _pass_time(env, minutes):
    """Let time pass whatever the owner's schedule (sleeping through nights)."""
    end = env.t + minutes
    while env.t < end and not env.done:
        night = env.t % MINUTES_PER_DAY >= 19 * 60 or env.t % MINUTES_PER_DAY < WAKE_TIME
        env.step({"name": "sleep"} if night else {"name": "wait", "arguments": {"minutes": min(600, end - env.t)}})


def test_extra_litter_box_is_delivered_and_used():
    """A bought litter box arrives by the front door; once put in a room, kittens can use it."""
    env = KittenEnv(seed=0, days=4, n_cats=2)
    env.reset()
    env.step({"name": "shop", "arguments": {"item": "litter_box"}})
    _pass_time(env, MINUTES_PER_DAY + 9 * 60)
    box = env.world.objects["litter_box_2"]
    assert box.room == "hallway" and box.props["urine"] == 0
    env.step({"name": "go_to", "arguments": {"room": "hallway"}})
    env.step({"name": "use", "arguments": {"verb": "take", "object": "litter_box_2"}})
    env.step({"name": "go_to", "arguments": {"room": "living_room"}})
    env.step({"name": "use", "arguments": {"verb": "put", "object": "litter_box_2"}})
    assert env.world.objects["litter_box_2"].room == "living_room"
    _pass_time(env, MINUTES_PER_DAY)
    uses = [r for r in env.trace if r["type"] == "cat" and r["action"] == "use_litter_box" and r["target"] == "litter_box_2"]
    assert uses


def test_rival_near_a_resource_puts_kitten_off():
    """An awake rival in the bowl's room makes eating there much less appealing; a friend doesn't."""
    from kittenbench.cat import Situation, not_guarded
    env = KittenEnv(seed=0, days=2, n_cats=2)
    a, b = env.cats
    bowl = env.world.objects["food_bowl"]
    b.tile, b.room = bowl.tile, bowl.room
    env.force_cat_step(b, "groom", 30)
    s = Situation(env.t, env.world, bowl, housemates=[b])
    env.change_bond(a, b, -2)
    assert not_guarded(a, s) == RULES["social"]["guarded_mult"]
    env.change_bond(a, b, 3)
    assert not_guarded(a, s) == 1.0


def test_cat_flu_spreads_unless_kittens_are_kept_apart(monkeypatch):
    """A kitten with cat flu passes it to a kitten it can reach, not to one behind a closed door."""
    from kittenbench.cat import URI
    monkeypatch.setitem(RULES["contagion"], "hourly_prob_same_room", 1.0)
    monkeypatch.setitem(RULES["contagion"], "hourly_prob_shared_home", 1.0)
    for door_open in (True, False):
        env = KittenEnv(seed=0, days=3, n_cats=2)
        sick, healthy = env.cats
        sick.conditions = [URI.start_at(env.t, env.cat_rng)]
        healthy.conditions = []
        sick.tile, sick.room = env.world.room_center("bedroom"), "bedroom"
        healthy.tile, healthy.room = env.world.room_center("kitchen"), "kitchen"
        env.world.objects["bedroom_door"].props["open"] = door_open
        env._on_household(None)
        caught = healthy.metrics.uri_caught_t is not None
        assert caught == door_open
        if caught:
            onset = healthy.conditions[0].onset_t - env.t
            low, high = RULES["contagion"]["incubation_days"]
            assert low * MINUTES_PER_DAY <= onset <= high * MINUTES_PER_DAY


def test_newcomer_arrives_in_carrier_and_can_be_kept_apart():
    """The newcomer isn't home until its day; carry() lets it out in a chosen room, and a closed door keeps it there."""
    env = KittenEnv(seed=0, days=6, n_cats=2, newcomer_day=2)
    obs = env.reset()
    newcomer = env.cats[1]
    assert "will bring Tofu" in obs.text and newcomer.away
    assert "hasn't arrived yet" in env.step({"name": "find_cat", "arguments": {"cat": "Tofu"}}).text
    env.step({"name": "go_to", "arguments": {"room": "kitchen"}})
    env.step({"name": "use", "arguments": {"verb": "fill", "object": "food_bowl", "item": "kibble", "amount": 200}})
    env.step({"name": "use", "arguments": {"verb": "fill", "object": "water_bowl"}})
    while "drops off Tofu" not in obs.text and env.t < 3 * MINUTES_PER_DAY:
        obs = env.step({"name": "sleep"} if env.t % MINUTES_PER_DAY >= 19 * 60 or env.t % MINUTES_PER_DAY < WAKE_TIME
                       else {"name": "wait", "arguments": {"minutes": 600}})
    assert newcomer.away_reason == "carrier"
    env.step({"name": "carry", "arguments": {"cat": "Tofu", "room": "bedroom"}})
    env.step({"name": "use", "arguments": {"verb": "close", "object": "bedroom_door"}})
    assert not newcomer.away and newcomer.arrived_t is not None
    rooms = set()
    for _ in range(10):
        env.step({"name": "wait", "arguments": {"minutes": 30}})
        rooms.add(newcomer.room)
    assert rooms == {"bedroom"}


def test_newcomer_lets_itself_out_if_left_in_carrier():
    env = KittenEnv(seed=0, days=6, n_cats=2, newcomer_day=2)
    env.reset()
    _pass_time(env, 2 * MINUTES_PER_DAY + 10 * 60)   # past 18:30 on day 2, plus the 30 minutes
    assert env.cats[1].away_reason is None and not env.cats[1].away
    assert env.cats[1].room is not None


def test_fights_happen_between_rivals_not_friends():
    """Hostile kittens left together fight; bonded kittens almost never do."""
    def fights(bond):
        env = KittenEnv(seed=3, days=4, n_cats=2)
        a, b = env.cats
        env.change_bond(a, b, -2)
        env.change_bond(a, b, bond + 1)
        env.reset()
        env.step({"name": "go_to", "arguments": {"room": "kitchen"}})
        env.step({"name": "use", "arguments": {"verb": "fill", "object": "food_bowl", "item": "kibble", "amount": 300}})
        env.step({"name": "use", "arguments": {"verb": "fill", "object": "water_bowl"}})
        _pass_time(env, 4 * MINUTES_PER_DAY)
        return env.metrics.fights
    assert fights(-0.9) > fights(0.9)


def test_untreated_flu_costs_more_the_longer_it_goes():
    """The flu score counts hours of untreated symptoms: a flu that started just before the end costs little, one
    ignored for days costs everything, whatever course it would have taken."""
    from kittenbench.cat import URI
    from kittenbench.scoring import _uri_score
    env = KittenEnv(seed=0, days=4)
    cat = env.cats[0]
    for severe in (True, False):
        onset = 3 * MINUTES_PER_DAY
        cat.conditions = [URI(onset, severe_t=onset + 2 * MINUTES_PER_DAY if severe else None,
                              recover_t=None if severe else onset + 6 * MINUTES_PER_DAY)]
        cat.metrics.uri_onset_t, cat.metrics.uri_treated_t = onset, None
        assert _uri_score(cat, onset + 4 * 60) == round(1 - 4 / 72, 3)       # 4 h untreated
        assert _uri_score(cat, onset + 36 * 60) == 0.5                       # 36 h untreated
        assert _uri_score(cat, onset + 4 * MINUTES_PER_DAY) == 0.0           # 96 h untreated
    cat.metrics.uri_treated_t = onset + 18 * 60
    assert _uri_score(cat, onset + 4 * MINUTES_PER_DAY) == 0.75              # treated after 18 h


def test_scent_towel_helps_kittens_kept_apart_accept_each_other():
    """A towel rubbed on one kitten, left with the other, raises their bond faster than separation alone."""
    def bond_after_a_day(with_towel):
        env = KittenEnv(seed=0, days=3, n_cats=2)
        a, b = env.cats
        env.change_bond(a, b, -2)
        env.change_bond(a, b, 0.4)                          # bond -0.6
        a.tile, a.room = env.world.room_center("bedroom"), "bedroom"
        b.tile, b.room = env.world.room_center("kitchen"), "kitchen"
        env.world.objects["bedroom_door"].props["open"] = False
        if with_towel:
            towel = env.world.objects["bath_towel"]
            towel.props.update(scent_of=b.idx, scented_t=env.t)
            env.world.place(towel, a.tile)
        for _ in range(24):
            env._on_household(None)
            env.t += 60
        return a.bond(b)
    apart, apart_with_scent = bond_after_a_day(False), bond_after_a_day(True)
    assert apart_with_scent > apart > -0.6
    assert apart_with_scent <= RULES["introductions"]["scent_cap"]


def test_supervised_sessions_short_and_calm_beat_long_and_tense():
    """Two kittens playing together: friends nearly always stay calm and grow closer; rivals in a long session
    usually end up hissing."""
    def sessions(bond, minutes):
        calm = 0
        for seed in range(20):
            env = KittenEnv(seed=seed, days=2, n_cats=2)
            a, b = env.cats
            env.change_bond(a, b, -2)
            env.change_bond(a, b, bond + 1)
            env.owner_room, env.owner_tile = "living_room", env.world.room_center("living_room")
            for k in (a, b):
                k.tile, k.room = env.owner_tile, "living_room"
                env.force_cat_step(k, "explore", 60)
            before = a.bond(b)
            text = env.tool_play(minutes=minutes, cat=a.name, with_cat=b.name)
            if "relaxed" in text:
                calm += 1
                assert a.bond(b) > before
        return calm
    assert sessions(0.5, 5) >= 17
    assert sessions(-0.8, 30) <= 6
