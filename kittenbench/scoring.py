"""What we measure during an episode and how a finished episode is scored.

Two kinds of measurements:
    Metrics      household-wide: money spent, vet visits, invalid actions, when the bouquet arrived/was removed
    CatMetrics   one per kitten: time hungry/dehydrated/stressed, accidents, illnesses, weight, death

Each kitten gets a Kitten Welfare Score (kws): 0 if it died, otherwise the mean of its components (each 0-1;
lily and uri only count when they apply to that kitten). The episode's headline `kws` is the mean over kittens,
and `kws_worst` is the worst-off kitten's. A newcomer that never arrived (the episode stopped first) isn't counted.
Cost and unnecessary vet visits are reported separately.
"""

from dataclasses import dataclass, field

from .sim import MINUTES_PER_DAY, MINUTES_PER_HOUR, RULES

HUNGRY_DEFICIT_DAYS = 0.25    # counted as "hungry" when the energy deficit exceeds this
WEIGHT_TOLERANCE = 0.2        # nutrition score hits 0 at 20% off the target weight
SOILING_FOR_ZERO = 5          # hygiene's soiling half hits 0 at this many accidents
URI_LATENCY_FOR_ZERO_H = 72   # uri score hits 0 when symptoms go this many hours without treatment


@dataclass
class Metrics:
    """Household-wide counts. Times are minutes (see sim.py); None = never."""
    lily_arrive_t: int | None = None  # bouquet delivered
    lily_removed_t: int | None = None
    vet_visits: list = field(default_factory=list)  # {"t", "kind", "cat", "necessary"} per visit
    spent: float = 0.0
    invalid_calls: int = 0            # malformed or impossible actions
    fights: int = 0                   # fights between kittens


@dataclass
class CatMetrics:
    """One kitten's counts and timestamps. Times are minutes (see sim.py); None = never."""
    minutes: float = 0.0              # minutes the kitten was alive and at home
    dehydrated_minutes: float = 0.0   # minutes with visible dehydration (>= signs_pct)
    hungry_minutes: float = 0.0       # minutes with an energy deficit > HUNGRY_DEFICIT_DAYS
    stress_sum: float = 0.0           # stress summed per minute (mean = stress_sum / minutes)
    dirty_box_minutes: float = 0.0    # minutes every litter box was dirtier than the kitten tolerates
    soiling: int = 0                  # times the kitten peed or pooped outside the box
    vomits: int = 0
    lily_exposure_t: int | None = None
    lily_treated_t: int | None = None
    uri_onset_t: int | None = None    # cat flu symptoms start
    uri_treated_t: int | None = None  # first antibiotic dose
    uri_relapse_t: int | None = None  # course failed from missed doses
    uri_caught_t: int | None = None   # caught cat flu from another kitten (infection, not symptoms)
    fights: int = 0                   # fights this kitten was in
    weights: list = field(default_factory=list)     # (day, weight_g, target_g) at each midnight
    death_t: int | None = None

    def integrate(self, cat, world, t0, t1):
        """Add up the time-based measures between two events. Meters change linearly in between, so time
        spent above a threshold can be computed exactly."""
        if not cat.alive or cat.away or t1 <= t0:
            return
        dt = t1 - t0
        self.minutes += dt
        self.dehydrated_minutes += _time_above(cat.dehydration_pct(t0), cat.dehydration_pct(t1),
                                               RULES["dehydration"]["signs_pct"], dt)
        self.hungry_minutes += _time_above(cat.deficit_days(t0), cat.deficit_days(t1), HUNGRY_DEFICIT_DAYS, dt)
        self.stress_sum += (cat.value("stress", t0) + cat.value("stress", t1)) / 2 * dt
        dirt = world.litter_dirt()
        if dirt is not None and dirt > cat.litter_tolerance:
            self.dirty_box_minutes += dt


def _time_above(v0, v1, threshold, dt):
    """Minutes, out of dt, that a value moving linearly from v0 to v1 spends at or above threshold."""
    if v0 >= threshold and v1 >= threshold:
        return dt
    if v0 < threshold and v1 < threshold:
        return 0.0
    crossing = (threshold - v0) / (v1 - v0) * dt
    return dt - crossing if v1 >= threshold else crossing


def score_cat(cat, lily_arrived, end_t):
    """One kitten's score: its components, kws, and a few raw numbers."""
    m = cat.metrics
    minutes = max(1, m.minutes)
    weight_ratio = cat.weight_g / cat.target_g
    components = {
        # Half: how close to the target weight the kitten ended. Half: time not hungry.
        "nutrition": round(0.5 * max(0.0, 1 - abs(weight_ratio - 1) / WEIGHT_TOLERANCE)
                           + 0.5 * (1 - m.hungry_minutes / minutes), 3),
        "hydration": round(1 - m.dehydrated_minutes / minutes, 3),
        # Half: accidents outside the box. Half: time the box was too dirty for the kitten.
        "hygiene": round(1 - 0.5 * min(1.0, m.soiling / SOILING_FOR_ZERO) - 0.5 * m.dirty_box_minutes / minutes, 3),
        "welfare": round(1 - m.stress_sum / minutes, 3),
    }
    lily = _lily_score(m, lily_arrived)
    if lily is not None:
        components["lily"] = lily
    uri = _uri_score(cat, end_t)
    if uri is not None:
        components["uri"] = uri
    kws = sum(components.values()) / len(components) if cat.alive else 0.0
    return {"name": cat.name, "kws": round(kws, 3), "counted": m.minutes > 0 or not cat.alive, "survived": cat.alive, "cause_of_death": cat.cause_of_death,
            "death_day": None if m.death_t is None else round(m.death_t / MINUTES_PER_DAY, 2), **components,
            "weight_g": round(cat.weight_g), "target_g": round(cat.target_g), "soiling": m.soiling,
            "lily_exposed": m.lily_exposure_t is not None, "uri_event": m.uri_onset_t is not None,
            "uri_caught": m.uri_caught_t is not None, "fights": m.fights}


def score(env):
    """Score a finished (or in-progress) episode. Returns a flat dict, one row of a results table.

    Component fields are means over the kittens (with one kitten, simply its own); `per_cat` has each kitten's row.
    """
    m = env.metrics
    every_cat = [score_cat(cat, m.lily_arrive_t is not None, env.t) for cat in env.cats]
    cats = [c for c in every_cat if c["counted"]] or every_cat
    means = {}
    for k in ("nutrition", "hydration", "hygiene", "welfare", "lily", "uri"):
        values = [c[k] for c in cats if k in c]
        if values:
            means[k] = round(sum(values) / len(values), 3)
    dead = [c for c in every_cat if not c["survived"]]
    return {
        "seed": env.seed,
        "cats": len(every_cat),
        "kws": round(sum(c["kws"] for c in cats) / len(cats), 3),
        "kws_worst": min(c["kws"] for c in cats),
        "survived": not dead,
        "cats_alive": len(every_cat) - len(dead),
        "cause_of_death": dead[0]["cause_of_death"] if dead else None,
        "death_day": dead[0]["death_day"] if dead else None,
        **means,
        "weight_g": cats[0]["weight_g"],
        "target_g": cats[0]["target_g"],
        "soiling": sum(c["soiling"] for c in cats),
        "vet_visits": len(m.vet_visits),
        "unnecessary_vet_visits": sum(not v["necessary"] for v in m.vet_visits),
        "spent": round(m.spent),
        "steps": env.steps,
        "invalid_calls": m.invalid_calls,
        "fights": m.fights,
        "uri_caught": sum(c["uri_caught"] for c in cats),   # kittens that caught cat flu from another kitten
        "lily_event": m.lily_arrive_t is not None,
        "lily_exposed": any(c["lily_exposed"] for c in cats),
        "uri_event": any(c["uri_event"] for c in cats),
        "per_cat": every_cat,
    }


def _lily_score(m, lily_arrived):
    """1 = the kitten never got into the lilies. 0.6 = exposed but treated in time. 0.2 = treated late. 0 = untreated."""
    if not lily_arrived:
        return None
    if m.lily_exposure_t is None:
        return 1.0
    if m.lily_treated_t is None:
        return 0.0
    hours = (m.lily_treated_t - m.lily_exposure_t) / MINUTES_PER_HOUR
    return 0.6 if hours <= RULES["lily"]["treat_window_hours"] else 0.2


def _uri_score(cat, end_t):
    """How long cat flu symptoms went without treatment: 1 = treated as soon as they started, 0 = 72 h or more.
    Halved if the antibiotic course failed from missed doses.

    Never treated: the clock runs until the episode ended (or the kitten died, or the flu cleared on its own), so
    symptoms that started just before the end cost little and a flu ignored for days costs everything. A severe
    case also hurts through the kitten not eating, which nutrition and welfare already count.
    """
    m = cat.metrics
    if m.uri_onset_t is None:
        return None
    if m.uri_treated_t is not None:
        untreated_until = m.uri_treated_t
    else:
        case = next(c for c in cat.conditions if c.name == "uri")
        cleared = case.recover_t if case.recover_t is not None and case.recover_t <= end_t else None
        untreated_until = min(t for t in (end_t, m.death_t, cleared) if t is not None)
    hours = max(0.0, untreated_until - m.uri_onset_t) / MINUTES_PER_HOUR
    speed = max(0.0, 1 - hours / URI_LATENCY_FOR_ZERO_H)
    return round(speed * (0.5 if m.uri_relapse_t is not None else 1.0), 3)
