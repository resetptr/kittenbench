"""Everything about the kitten: its illnesses, the activities it can choose from, and the kitten itself.

Sections:
    illnesses     lily poisoning and cat flu as state machines with stage-change times
    interactions  loads the activities from data/kitten.toml, plus the named requirements and multipliers
    kitten        meters, rates, choosing and scoring activities, and the "specials" (eat, drink, litter box...)

With several kittens, each kitten keeps a bond with every other kitten (-1 hostile .. 1 bonded) in `Cat.bonds`.
Activities that target another kitten ("cat" targets in kitten.toml) only exist when there is another kitten.
"""

import math
from dataclasses import dataclass, field

from .sim import (INTERACTION_SPECS, MINUTES_PER_DAY, MINUTES_PER_HOUR, NEEDS, RULES, VERY_LOUD, Meter,
                  energy_need_kcal, response_curve, target_gain_g_per_day)

SOCIAL = RULES["social"]


# ════ illnesses ═══════════════════════════════════════════════════════════════════════════════════
# Illnesses the kitten can have.
#
# Each illness is a small state machine over time: `stage(t)` says where it is at minute t, and the other methods
# say how that stage affects the kitten (appetite, pain, urine, water loss) and what an owner could notice (`signs`).
# Timelines and treatment outcomes come from rules.toml.

TREATMENT_TAKES_EFFECT_MIN = 6 * MINUTES_PER_HOUR   # assumption: antibiotics restore appetite after ~6 h


def _uniform(rng, bounds):
    low, high = bounds
    return rng.uniform(low, high)


class Condition:
    """Interface shared by all illnesses. Defaults mean "no effect"."""

    name = ""

    def stage(self, t):
        raise NotImplementedError

    def appetite(self, t, warm_wet=False):
        """Fraction of normal appetite (1.0 = normal). `warm_wet`: warmed wet food is in the bowl."""
        return 1.0

    def pain(self, t):
        return 0.0

    def urine_mult(self, t):
        """Multiplier on how fast the bladder fills."""
        return 1.0

    def water_loss_mult(self, t):
        """Multiplier on how fast the kitten dehydrates."""
        return 1.0

    def change_times(self):
        """Moments when this illness changes stage. The engine re-evaluates the kitten's body at each one."""
        return []

    def sneeze_rate_per_hour(self, t):
        return 0.0

    def is_fatal(self, t):
        """True once the illness itself is fatal (illnesses that kill through not eating return False)."""
        return False

    def signs(self, t):
        """Short phrases an owner could notice when inspecting the kitten."""
        return []


@dataclass
class LilyToxicosis(Condition):
    """Lily poisoning: acute kidney injury. Timeline from MSPCA-Angell; treatment window from ASPCApro.

    Stages: nausea (vomiting, not eating) -> polyuria (drinking and peeing a lot) -> anuria (kidneys stop,
    no urine) -> weak -> death. Treatment is 48 h of IV fluids at the hospital; whether it works depends
    on how many hours after exposure it starts.
    """
    t0: int                 # minute of exposure
    vomit_times: list       # minutes at which the kitten vomits
    polyuria_t: int         # stage start times (minutes)
    anuria_t: int
    weakness_t: int
    death_t: int            # death if untreated (or if treatment fails)
    treated_t: int | None = None
    will_survive: bool | None = None  # decided when treatment starts
    resolved: bool = False            # set when the kitten is discharged healthy

    name = "lily"

    @classmethod
    def start(cls, t0, rng):
        """Sample a timeline for an exposure at minute t0."""
        r = RULES["lily"]
        H = MINUTES_PER_HOUR
        n_vomits = rng.randint(1, 3)
        vomits = sorted(t0 + int(_uniform(rng, r["vomit_hours"]) * H) + i * rng.randint(20, 90) for i in range(n_vomits))
        anuria = t0 + int(_uniform(rng, r["anuria_hours"]) * H)
        # Keep the stages in order even when the sampled ranges overlap.
        polyuria = min(t0 + int(_uniform(rng, r["polyuria_hours"]) * H), anuria - 6 * H)
        weakness = max(t0 + int(_uniform(rng, r["weakness_hours"]) * H), anuria)
        death = max(t0 + int(_uniform(rng, r["death_hours"]) * H), weakness + 12 * H)
        return cls(t0, vomits, polyuria, anuria, weakness, death)

    def change_times(self):
        return [self.polyuria_t, self.anuria_t, self.weakness_t, self.death_t]

    def stage(self, t):
        if self.resolved:
            return "resolved"
        if t < self.polyuria_t:
            return "nausea"
        if t < self.anuria_t:
            return "polyuria"
        if t < self.weakness_t:
            return "anuria"
        return "weak"

    def treat(self, t, rng):
        """Start IV fluids at minute t. Decides (and returns) whether the kitten will survive."""
        r = RULES["lily"]
        self.treated_t = t
        hours_since_exposure = (t - self.t0) / MINUTES_PER_HOUR
        if hours_since_exposure <= r["treat_window_hours"]:
            self.will_survive = True
        elif t < self.anuria_t:
            self.will_survive = rng.random() < r["survive_late_pre_anuria"]
        else:
            self.will_survive = rng.random() < r["survive_post_anuria"]
        return self.will_survive

    def appetite(self, t, warm_wet=False):
        return {"nausea": 0.3, "polyuria": 0.5, "anuria": 0.1, "weak": 0.0}.get(self.stage(t), 1.0)

    def pain(self, t):
        return {"nausea": 0.4, "polyuria": 0.3, "anuria": 0.7, "weak": 0.9}.get(self.stage(t), 0.0)

    def urine_mult(self, t):
        return {"polyuria": 2.5, "anuria": 0.0, "weak": 0.0}.get(self.stage(t), 1.0)

    def water_loss_mult(self, t):
        return 2.0 if self.stage(t) == "polyuria" else 1.0

    def is_fatal(self, t):
        if self.resolved:
            return False
        if self.treated_t is not None and self.will_survive:
            return False
        return t >= self.death_t

    def signs(self, t):
        return {
            "nausea": ["drooling", "lethargic", "not interested in food"],
            "polyuria": ["drinking a lot", "large urine clumps"],
            "anuria": ["no urine in litter box", "lethargic", "hunched"],
            "weak": ["very weak", "barely moves", "hunched"],
        }.get(self.stage(t), [])


@dataclass
class URI(Condition):
    """Cat flu (feline upper respiratory infection), per Cornell / AMC.

    Stages: incubating -> mild (sneezing, discharge, eats less). Untreated, a mild case either clears on its
    own (`recover_t`) or turns severe (`severe_t`: barely eats, which can be fatal through starvation).
    Once the vet treats it the stage is "treated"; at the end of the antibiotic course it resolves if enough
    doses were given, otherwise it relapses to severe.
    """
    onset_t: int                 # minute symptoms start
    severe_t: int | None         # minute it turns severe (None for a case that clears on its own)
    recover_t: int | None        # minute it clears on its own (None for a case that turns severe)
    treated_t: int | None = None
    doses_given: list = field(default_factory=list)  # minutes at which antibiotic doses were given
    resolved: bool = False

    name = "uri"

    @classmethod
    def start(cls, onset_day, rng):
        """Sample a case whose symptoms start on `onset_day`."""
        return cls.start_at(int(onset_day * MINUTES_PER_DAY + rng.randint(6, 20) * MINUTES_PER_HOUR), rng)

    @classmethod
    def start_at(cls, onset, rng):
        """Sample a case whose symptoms start at minute `onset` (e.g. caught from another kitten)."""
        r = RULES["uri"]
        mild_end = onset + int(_uniform(rng, r["mild_days"]) * MINUTES_PER_DAY)
        if rng.random() < r["severe_prob"]:
            return cls(onset, severe_t=mild_end, recover_t=None)
        return cls(onset, severe_t=None, recover_t=onset + int(_uniform(rng, r["recover_days"]) * MINUTES_PER_DAY))

    def change_times(self):
        times = [self.onset_t, self.severe_t, self.recover_t]
        if self.treated_t is not None:
            times += [self.treated_t + TREATMENT_TAKES_EFFECT_MIN, self.course_end()]
        return [t for t in times if t is not None]

    def stage(self, t):
        if self.resolved:
            return "resolved"
        if t < self.onset_t:
            return "incubating"
        if self.treated_t is not None:
            return "treated"
        if self.recover_t is not None and t >= self.recover_t:
            return "resolved"
        if self.severe_t is not None and t >= self.severe_t:
            return "severe"
        return "mild"

    def treat(self, t, rng):
        """The vet starts antibiotics and gives the first dose."""
        self.treated_t = t
        self.doses_given = [t]
        return True

    def course_end(self):
        return self.treated_t + RULES["uri"]["course_days"] * MINUTES_PER_DAY

    def adherence(self, t):
        """Fraction of the doses due so far that were actually given."""
        r = RULES["uri"]
        elapsed = min(t, self.course_end()) - self.treated_t
        expected = max(1, int(elapsed / MINUTES_PER_DAY * r["doses_per_day"]))
        return min(1.0, len(self.doses_given) / expected)

    def check_course(self, t):
        """Call every minute. At course end: returns "resolved" or "relapse"; otherwise None."""
        if self.treated_t is None or t < self.course_end() or self.resolved:
            return None
        if self.adherence(t) >= RULES["uri"]["min_adherence"]:
            self.resolved = True
            return "resolved"
        # Too many missed doses: back to an untreated, severe infection.
        self.treated_t, self.severe_t, self.recover_t = None, t, None
        return "relapse"

    def appetite(self, t, warm_wet=False):
        r = RULES["uri"]
        s = self.stage(t)
        if s == "treated":
            # Antibiotics take a few hours before appetite comes back.
            working = t >= self.treated_t + TREATMENT_TAKES_EFFECT_MIN
            base = r["treated_appetite"] if working else r["appetite_severe"]
        else:
            base = {"mild": r["appetite_mild"], "severe": r["appetite_severe"]}.get(s, 1.0)
        if s in ("mild", "severe", "treated") and warm_wet:
            base = min(1.0, base + r["warm_wet_appetite_bonus"])
        return base

    def pain(self, t):
        return {"mild": 0.2, "severe": 0.5, "treated": 0.1}.get(self.stage(t), 0.0)

    def water_loss_mult(self, t):
        return 1.1 if self.stage(t) == "severe" else 1.0  # fever

    def sneeze_rate_per_hour(self, t):
        return {"mild": 1.5, "severe": 2.5, "treated": 0.5}.get(self.stage(t), 0.0)

    def shedding(self, t):
        """How contagious the kitten is to other kittens (1 = fully)."""
        return {"mild": 1.0, "severe": 1.0, "treated": RULES["contagion"]["treated_shedding"]}.get(self.stage(t), 0.0)

    def signs(self, t):
        return {
            "mild": ["watery eyes", "clear nasal discharge"],
            "severe": ["crusty eye discharge", "thick nasal discharge", "lethargic", "warm to the touch"],
            "treated": ["slight eye discharge"],
        }.get(self.stage(t), [])


# ════ interactions ════════════════════════════════════════════════════════════════════════════════
# Interactions: the kitten's possible activities, loaded from data/kitten.toml.
#
# The data file refers to requirements and multipliers by name; they are defined here as small functions.
# Each takes the kitten and a Situation (what's around the kitten right now) for one candidate target.

NOVELTY_DAYS = 2.0            # assumption: a new object stays very interesting for a couple of days
NOVELTY_FLOOR = 0.25          # assumption: plants stay somewhat interesting to cats indefinitely
FAMILIARITY_PER_VISIT = 0.3   # assumption: each investigation makes it a bit less interesting


@dataclass
class Interaction:
    name: str
    target: str
    minutes: tuple
    effects: dict = field(default_factory=dict)
    special: str | None = None
    requires: list = field(default_factory=list)
    base: float = 0.0
    multipliers: list = field(default_factory=list)
    hidden: bool = False
    sound: tuple | None = None
    animation: str = ""
    description: str = ""
    description_unwell: str | None = None
    seen: str = ""

    @property
    def mean_minutes(self):
        return sum(self.minutes) / 2


def load_interactions():
    return [Interaction(**{**spec, "minutes": tuple(spec["minutes"]),
                           "sound": tuple(spec["sound"]) if "sound" in spec else None})
            for spec in INTERACTION_SPECS]


@dataclass
class Situation:
    """What the kitten considers when scoring one option."""
    t: float
    world: object                 # the Apartment
    target: object = None         # the Obj the kitten would use, if any
    owner_tile: tuple | None = None   # where the owner is, if home and awake; else None
    reachable: set = field(default_factory=set)   # rooms the kitten can walk to
    housemates: list = field(default_factory=list)   # the other kittens at home (alive, not away)


# ── requirements: all must be true for the option to be considered ────────

def bowl_has_food(cat, s):
    return bowl_kcal(s.target, s.t) > 1


def bowl_has_water(cat, s):
    return s.target.props["ml"] > 2


def hungry(cat, s):
    return cat.wants_to_eat(s.t)


def thirsty(cat, s):
    return cat.meters["dehydration_pct"].value(s.t) > 0.5


def needs_toilet(cat, s):
    return cat.meters["bladder"].value(s.t) >= 0.7 or cat.meters["bowel"].value(s.t) >= 0.7


def no_food_reachable(cat, s):
    return not any(o.room in s.reachable and bowl_kcal(o, s.t) > 1 for o in s.world.of_kind("food_bowl"))


def not_lethargic(cat, s):
    return not cat.lethargic(s.t)


# For "cat" targets, s.target is the other kitten.

BUSY_WITH_BODY = ("eat", "drink", "litter_box", "soil_floor")   # specials a kitten won't drop to socialize


def partner_playful(cat, s):
    """The other kitten is awake, well, not busy eating or on the toilet, and not a rival."""
    other = s.target
    busy = other.step is not None and other.step.interaction.special in BUSY_WITH_BODY
    resting = other.activity.startswith("sleep") or other.activity in ("rest", "hide")
    return not busy and not resting and not other.lethargic(s.t) and cat.bond(other) > SOCIAL["hostile_below"]


def partner_resting(cat, s):
    """The other kitten is lying still, and the two are close enough to groom each other."""
    other = s.target
    lying = other.activity.startswith("sleep") or other.activity in ("rest", "groom")
    return lying and cat.bond(other) > SOCIAL["groom_above"]


def partner_sleeping(cat, s):
    return s.target.activity.startswith("sleep") and cat.bond(s.target) > SOCIAL["sleep_together_above"]


def partner_rival(cat, s):
    """The other kitten is a rival and awake."""
    return cat.bond(s.target) < SOCIAL["hostile_below"] and not s.target.activity.startswith("sleep")


def has_rival(cat, s):
    return any(cat.bond(other) < SOCIAL["hostile_below"] for other in s.housemates)


REQUIREMENTS = {f.__name__: f for f in (bowl_has_food, bowl_has_water, hungry, thirsty, needs_toilet,
                                        no_food_reachable, not_lethargic, partner_playful, partner_resting,
                                        partner_sleeping, partner_rival, has_rival)}


# ── multipliers: scale an option's score ──────────────────────────────────

def _box_cleanliness(cat, box):
    """1 while the box is within the kitten's tolerance; falls toward 0.15 as it gets dirtier (AAFP/ISFM: scoop daily)."""
    over = box_dirt(box) - cat.litter_tolerance
    return 1.0 if over < 0 else max(0.15, 1 - over * 0.25)


def litter_cleanliness(cat, s):
    return _box_cleanliness(cat, s.target)


def litter_unacceptable(cat, s):
    """How strongly the kitten would rather go elsewhere: 1 if no box is reachable, else 1 minus how appealing the
    best reachable box is (dirt, and rivals sitting next to it)."""
    boxes = [b for b in s.world.of_kind("litter_box") if b.room in s.reachable]
    if not boxes:
        return 1.0
    return 1.0 - max(_box_cleanliness(cat, b) * _guarded_mult(cat, s.housemates, b.room) for b in boxes)


def not_guarded(cat, s):
    """A resource with an awake rival kitten in the same room is much less appealing (rival guarding)."""
    return _guarded_mult(cat, s.housemates, s.target.room)


def _guarded_mult(cat, housemates, room):
    for other in housemates:
        awake = not other.hidden and not other.activity.startswith("sleep")
        if awake and other.room == room and cat.bond(other) < SOCIAL["hostile_below"]:
            return SOCIAL["guarded_mult"]
    return 1.0


def friendly(cat, s):
    return 0.5 + cat.bond(s.target)


def hostility(cat, s):
    return -cat.bond(s.target)


def rivalry(cat, s):
    """How hostile the kitten is toward its worst rival."""
    return max([-cat.bond(other) for other in s.housemates] + [0.0])


def bold(cat, s):
    return 0.3 + cat.traits["boldness"]


def cooled_down(cat, s):
    """0 right after a hiss or fight, back to 1 after conflict_cooldown_hours."""
    return min(1.0, (s.t - cat.last_conflict_t) / (SOCIAL["conflict_cooldown_hours"] * MINUTES_PER_HOUR))


def trust(cat, s):
    return cat.meters["trust"].value(s.t)


def timid(cat, s):
    return 1.2 - cat.traits["boldness"]


def unwell(cat, s):
    return 3.0 if cat.lethargic(s.t) else 1.0


def playful(cat, s):
    return 0.5 + cat.traits["activity"]


def novelty(cat, s):
    """How interesting an object is. New things are most interesting; the pull fades over a couple of days and
    a little with each investigation, but never below a floor (cats keep going back to plants)."""
    p = s.target.props
    age_days = (s.t - p.get("added_t", 0)) / MINUTES_PER_DAY
    fresh = max(NOVELTY_FLOOR, math.exp(-age_days / NOVELTY_DAYS))
    return fresh / (1 + FAMILIARITY_PER_VISIT * p.get("investigated", 0))


def curious(cat, s):
    return 0.3 + cat.traits["impulsiveness"]


def sociable(cat, s):
    return cat.traits["sociability"]


MULTIPLIERS = {f.__name__: f for f in (litter_cleanliness, litter_unacceptable, trust, timid, unwell,
                                       playful, novelty, curious, sociable, not_guarded, friendly, hostility,
                                       rivalry, bold, cooled_down)}


def box_dirt(box):
    """How dirty a litter box is: unscooped deposits plus leftover smell."""
    return box.props["urine"] + box.props["feces"] + box.props["residue"]


def bowl_kcal(bowl, t):
    """Edible kcal in a food bowl (wet food that has sat out too long doesn't count)."""
    f = RULES["food"]
    p = bowl.props
    wet_fresh = p["wet_g"] > 0 and t - p["wet_added_t"] <= f["wet_spoil_minutes"]
    return p["kibble_g"] * f["kibble_kcal_per_g"] + (p["wet_g"] * f["wet_kcal_per_g"] if wet_fresh else 0)


def check_names():
    """Fail fast if the data file names a requirement or multiplier that doesn't exist."""
    for i in load_interactions():
        for name in i.requires:
            assert name in REQUIREMENTS, f"{i.name}: unknown requirement {name}"
        for name in i.multipliers:
            assert name in MULTIPLIERS, f"{i.name}: unknown multiplier {name}"


check_names()


# ════ kitten ══════════════════════════════════════════════════════════════════════════════════════
# The kitten: its needs (meters), how it picks what to do next, and what each activity does to it.
#
# How the kitten works, game-style:
#   * Its needs are meters (sim.py) that change at steady rates. The rates depend on its current activity and
#     its health, so they're recomputed whenever either changes (`update_rates`).
#   * When an activity ends the kitten scores every option (data/kitten.toml) and picks the best (`choose`).
#   * An activity is a Step: walk to the target, then do it until it ends. Its effects are extra rates on the kitten's
#     meters while it lasts; activities whose result depends on an object (eating from a bowl, using the litter
#     box) also run a "special" when they finish.
#
# Body rates come from rules.toml (vet sources); behavior rates and curves come from kitten.toml (tuning).

TRAITS = ("boldness", "activity", "impulsiveness", "food_motivation", "sociability")
INTERACTIONS = load_interactions()


@dataclass
class Step:
    """One activity: walk along `path`, then do the interaction at the end of it."""
    interaction: object      # an Interaction
    target: object           # the Obj used, or None
    path: list               # tiles from the kitten's position to where the activity happens
    t_start: float
    t_arrive: float
    t_end: float
    spot: str = ""           # e.g. "under the bed" for hiding
    partner: object = None   # the other kitten, for activities with another kitten
    lead: bool = True        # False for the kitten that was drawn into another kitten's activity


class Cat:
    def __init__(self, name, weight_g, traits, litter_tolerance, age0_days=56):
        self.name = name
        self.weight_g = weight_g
        self.traits = traits                      # personality, each 0-1 (see TRAITS)
        self.litter_tolerance = litter_tolerance  # unscooped deposits the kitten accepts before avoiding the box
        self.age0_days = age0_days                # age at adoption: 8 weeks
        self.meters = {                           # see data/kitten.toml for what each one means
            "hunger": Meter(0.5, high=1.0),
            "energy_debt": Meter(0.0),
            "dehydration_pct": Meter(0.0),
            "bladder": Meter(0.3, high=1.0),
            "bowel": Meter(0.3, high=1.0),
            "sleepiness": Meter(0.4, high=1.0),
            "boredom": Meter(0.2, high=1.0),
            "stress": Meter(0.7, high=1.0),       # starts high: a new home
            "trust": Meter(0.15, high=1.0),       # in the owner
        }
        self.tile, self.room = None, None
        self.step = None
        self.conditions = []                      # illnesses, current and past (cat.py)
        self.alive, self.cause_of_death = True, None
        self.away = False                         # at the vet or hospital
        self.intake_today = 0.0                   # kcal eaten since midnight (for growth)
        self.target_g = weight_g                  # target weight; grows each midnight along the target curve
        self.idx = 0                              # position among the household's kittens (set by the env)
        self.metrics = None                       # this kitten's CatMetrics (set by the env)
        self.bonds = {}                           # other kitten's idx -> bond, -1 (hostile) to 1 (bonded)
        self.scratch_until = -1                   # a fight left a visible scratch until this minute
        self.last_conflict_t = -10 ** 9           # minute of the kitten's last hiss or fight
        self.away_reason = None                   # while away: "vet", "not_arrived" (a newcomer) or "carrier"
        self.arrived_t = None                     # minute the kitten came home (set by the env)

    @classmethod
    def sample(cls, rng, name="Miso"):
        """A random kitten: personality, litter tolerance and adoption weight."""
        traits = {k: round(rng.random(), 2) for k in TRAITS}
        tolerance = rng.uniform(*RULES["litter"]["tolerance_deposits"])
        weight = rng.uniform(*RULES["growth"]["start_weight_g"])
        return cls(name, weight, traits, tolerance)

    # ── reading the kitten's state ─────────────────────────────────────────
    def value(self, meter, t):
        return self.meters[meter].value(t)

    def bond(self, other):
        return self.bonds.get(other.idx, 0.0)

    @property
    def activity(self):
        if not self.alive:
            return "dead"
        return self.step.interaction.name if self.step else "none"

    @property
    def hidden(self):
        return bool(self.step and self.step.interaction.hidden)

    def age_days(self, t):
        return self.age0_days + t / MINUTES_PER_DAY

    def need_kcal(self, t):
        return energy_need_kcal(self.weight_g, self.age_days(t))

    def stomach_kcal(self, t):
        return self.need_kcal(t) * RULES["food"]["stomach_days"]

    def dehydration_pct(self, t):
        return self.value("dehydration_pct", t)

    def deficit_days(self, t):
        return self.value("energy_debt", t)

    def active_conditions(self, t):
        return [c for c in self.conditions if c.stage(t) not in ("resolved", "incubating")]

    def lethargic(self, t):
        """Too unwell to play or explore: underfed, dehydrated, or in a bad stage of an illness."""
        if self.deficit_days(t) > RULES["fasting"]["weak_deficit_days"]:
            return True
        if self.dehydration_pct(t) >= RULES["dehydration"]["signs_pct"]:
            return True
        return any(c.stage(t) in ("severe", "anuria", "weak", "nausea") for c in self.conditions)

    def appetite(self, t, warm_wet=False):
        """Fraction of normal appetite: the worst across all illnesses, and very low when critically underfed."""
        a = min([c.appetite(t, warm_wet) for c in self.conditions] + [1.0])
        if self.deficit_days(t) > RULES["fasting"]["critical_deficit_days"]:
            a = min(a, 0.2)
        return a

    def wants_to_eat(self, t, warm_wet=False):
        """Hungry enough for a meal. A low appetite means the kitten stops eating before its stomach is full."""
        appetite = self.appetite(t, warm_wet)
        room_for_food = self.value("hunger", t) - (1 - appetite)
        return room_for_food > 0.15 or (self.deficit_days(t) > 0 and appetite > 0.05)

    def urgency(self, meter, t):
        return response_curve(NEEDS["urgency"][meter], self.value(meter, t))

    # ── rates ──────────────────────────────────────────────────────────
    def update_rates(self, t, owner_room, dirty_box, rival_near=False):
        """Set every meter's rate from the kitten's current activity and health.

        owner_room: where the owner is if home and awake, else None.  dirty_box: every litter box is dirtier
        than the kitten tolerates.  rival_near: an awake rival kitten is in the same room.
        """
        if self.away or not self.alive:
            for m in self.meters.values():
                m.set_rate(t, 0.0)
            return
        b, stressors = NEEDS["base_rates"], NEEDS["stressors"]
        water_mult = _product(c.water_loss_mult(t) for c in self.conditions)
        urine_mult = _product(c.urine_mult(t) for c in self.conditions)
        stomach_empty = self.value("hunger", t) >= 1.0
        rates = {
            # A full stomach empties in `stomach_days`; after that the shortfall becomes energy debt.
            "hunger": 1 / (RULES["food"]["stomach_days"] * MINUTES_PER_DAY),
            "energy_debt": 1 / MINUTES_PER_DAY if stomach_empty else 0.0,
            # 55 ml water per kg per day = 5.5% of body weight per day (rules.toml [water]).
            "dehydration_pct": RULES["water"]["turnover_ml_per_kg_day"] / 10 / MINUTES_PER_DAY * water_mult,
            # Urine production / bladder size, both per kg (rules.toml [elimination]).
            "bladder": (RULES["elimination"]["urine_ml_per_kg_day"] / RULES["elimination"]["void_threshold_ml_per_kg"]
                        / MINUTES_PER_DAY * urine_mult),
            "bowel": 0.0,   # filled by eating instead
            "sleepiness": b["sleepiness"],
            "boredom": b["boredom"],
            "stress": (b["stress"]
                       + (stressors["dirty_litter_box"] if dirty_box else 0.0)
                       + (stressors["starving"] if self.deficit_days(t) > 0 or stomach_empty else 0.0)
                       + stressors["pain"] * max([c.pain(t) for c in self.conditions] + [0.0])
                       + (stressors["rival_near"] if rival_near else 0.0)),
            "trust": b["trust_with_owner"] if owner_room == self.room and not self.hidden else 0.0,
        }
        if self.step:
            for meter, extra in self.step.interaction.effects.items():
                rates[meter] += extra
        for meter, rate in rates.items():
            self.meters[meter].set_rate(t, rate)

    def upcoming(self, t):
        """Thresholds the kitten's meters are heading toward: (time, what, meter, level) for each."""
        f, d = RULES["fasting"], RULES["dehydration"]
        watches = [("hunger", 1.0, "stomach empty"),   # each: (meter, level, what to tell the engine)
                   ("bladder", 1.0, "urgent"), ("bowel", 1.0, "urgent"),
                   ("dehydration_pct", d["signs_pct"], "body changed"), ("dehydration_pct", d["death_pct"], "body changed"),
                   ("energy_debt", f["weak_deficit_days"], "body changed"),
                   ("energy_debt", f["critical_deficit_days"], "body changed"),
                   ("energy_debt", f["death_deficit_days"], "body changed")]
        out = []
        for meter, level, what in watches:
            when = self.meters[meter].time_to_reach(t, level)
            if when is not None:
                out.append((when, what, meter, level))
        return out

    # ── choosing what to do ────────────────────────────────────────────
    def choose(self, t, world, owner_tile, rng, housemates=()):
        """Score every option the kitten can reach and return the best as (interaction, target, path).

        housemates: the other kittens at home. For "cat" interactions the target is the other kitten.
        """
        reachable = world.reachable_rooms(self.tile)
        housemates = list(housemates)
        lo, hi = NEEDS["choice"]["jitter"]
        best = None
        for inter in INTERACTIONS:
            for target, tile in self._targets(inter, world, owner_tile, reachable, rng, housemates):
                s = Situation(t, world, target, owner_tile, reachable, housemates)
                if not all(REQUIREMENTS[name](self, s) for name in inter.requires):
                    continue
                path = world.path(self.tile, tile)
                if path is None:
                    continue
                score = self.score(inter, s, len(path) - 1) * rng.uniform(lo, hi)
                if best is None or score > best[0]:
                    best = (score, inter, target, path)
        return best[1:]

    def score(self, inter, s, distance):
        """How appealing an option is: urgency of the needs it relieves, plus `base`, times multipliers,
        discounted with distance."""
        relief = SPECIAL_RELIEF[inter.special](self, s) if inter.special in SPECIAL_RELIEF else 0.0
        for meter, extra_rate in inter.effects.items():
            current = self.value(meter, s.t)
            if extra_rate < 0 and current > 0:
                fraction_relieved = min(1.0, -extra_rate * inter.mean_minutes / current)
                relief += self.urgency(meter, s.t) * fraction_relieved
        score = inter.base + relief
        for name in inter.multipliers:
            score *= MULTIPLIERS[name](self, s)
        return score / (1 + distance / NEEDS["choice"]["distance_halving_tiles"])

    def _targets(self, inter, world, owner_tile, reachable, rng, housemates):
        """Where an interaction could happen: list of (target, tile). The target is an object, another kitten
        (for "cat"), or None."""
        if inter.target == "cat":
            return [(other, other.tile) for other in housemates if other.room in reachable and not other.hidden]
        if inter.target == "self":
            return [(None, self.tile)]
        if inter.target == "owner":
            ok = owner_tile is not None and world.tile_room[owner_tile] in reachable
            return [(None, owner_tile)] if ok else []
        if inter.target == "room":
            others = sorted(reachable - {self.room})
            return [(None, world.random_tile(rng.choice(others), rng))] if others else []
        if inter.target == "floor":
            rooms = sorted(r for r in reachable if r != "kitchen")
            return [(None, world.random_tile(rng.choice(rooms), rng))] if rooms else []
        return [(o, o.tile) for o in world.of_kind(inter.target) if o.room in reachable]

    def planned_minutes(self, inter, target, t, rng):
        """How long the activity lasts: random within its range, but eating and drinking stop when the kitten is done."""
        minutes = rng.uniform(*inter.minutes)
        if inter.special == "eat":
            minutes = min(minutes, max(0.5, _eat_plan(self, target, t)[1]))
        elif inter.special == "drink":
            minutes = min(minutes, max(0.5, _drink_plan(self, target, t)[1]))
        return minutes

    # ── once a day ─────────────────────────────────────────────────────
    def end_of_day(self, t, day_fraction=1.0):
        """At midnight: grow (or shrink) based on what the kitten ate today. Returns the weight change in grams.

        Eating the full daily need gives the target gain; less scales it down, and below `stall_ratio` the kitten
        loses weight. (The stomach can't hold more than the daily need, so overeating isn't modeled.)
        `day_fraction` is how much of the day the kitten was home (the first day starts at 10:00); need and gain
        are scaled by it.
        """
        g = RULES["growth_model"]
        ratio = self.intake_today / (self.need_kcal(t) * day_fraction)
        scale = max(-1.5, min(1.0, (ratio - g["stall_ratio"]) / (1 - g["stall_ratio"])))
        gain = target_gain_g_per_day(self.age_days(t)) * scale * day_fraction
        self.weight_g += gain
        self.intake_today = 0.0
        return gain

    def check_death(self, t):
        """Mark the kitten dead if it's past a fatal threshold, recording why."""
        if not self.alive:
            return
        if self.dehydration_pct(t) >= RULES["dehydration"]["death_pct"]:
            cause = "dehydration"
        elif self.deficit_days(t) >= RULES["fasting"]["death_deficit_days"]:
            cause = "starvation"
        else:
            fatal = next((c for c in self.conditions if c.is_fatal(t)), None)
            cause = None if fatal is None else "lily toxicosis (acute kidney injury)" if fatal.name == "lily" else fatal.name
        if cause:
            behind = [c.name for c in self.active_conditions(t)]
            if cause in ("dehydration", "starvation") and behind:
                cause += " (not eating: " + ", ".join(behind) + ")"
            self.alive, self.cause_of_death = False, cause


# ── specials: activities whose result depends on the object ───────────────
# Each has a relief function (used when scoring) and a finish function (applied when the activity ends, with
# the minutes actually spent on it). `listener` is the environment: it hears sounds and records events.

def _eat_plan(cat, bowl, t):
    """(kcal the kitten wants, minutes it would take) for a meal from this bowl."""
    f = RULES["food"]
    warm = bowl.props["warm_until"] > t and bowl.props["wet_g"] > 1
    target_hunger = 1 - cat.appetite(t, warm)
    want = cat.deficit_days(t) * cat.need_kcal(t) + max(0.0, cat.value("hunger", t) - target_hunger) * cat.stomach_kcal(t)
    want = min(want, bowl_kcal(bowl, t))
    kcal_per_min = f["eat_kibble_g_per_min"] * f["kibble_kcal_per_g"]
    return want, want / kcal_per_min


def _drink_plan(cat, bowl, t):
    """(ml the kitten wants, minutes it would take)."""
    want = min(bowl.props["ml"], cat.dehydration_pct(t) / 100 * cat.weight_g + 5)
    return want, want / RULES["water"]["drink_ml_per_min"]


def finish_eat(cat, step, t, minutes, world, listener):
    """Eat fresh wet food first, then kibble. Food repays any energy debt, then fills the stomach."""
    f = RULES["food"]
    p = step.target.props
    want, _ = _eat_plan(cat, step.target, step.t_arrive)
    kcal_left = min(want, minutes * f["eat_kibble_g_per_min"] * f["kibble_kcal_per_g"])
    eaten = 0.0
    wet_fresh = t - p["wet_added_t"] <= f["wet_spoil_minutes"]
    foods = [("wet", f["wet_kcal_per_g"], f["wet_moisture"]), ("kibble", f["kibble_kcal_per_g"], f["kibble_moisture"])]
    for food, kcal_per_g, moisture in foods:
        if food == "wet" and not wet_fresh:
            continue
        grams = min(p[food + "_g"], kcal_left / kcal_per_g)
        p[food + "_g"] -= grams
        kcal_left -= grams * kcal_per_g
        eaten += grams * kcal_per_g
        cat.meters["dehydration_pct"].add(t, -grams * moisture / cat.weight_g * 100)  # water in the food
        cat.meters["bowel"].add(t, grams * RULES["elimination"]["stool_per_g_" + food])
    repay = min(eaten, cat.deficit_days(t) * cat.need_kcal(t))
    cat.meters["energy_debt"].add(t, -repay / cat.need_kcal(t))
    cat.meters["hunger"].add(t, -(eaten - repay) / cat.stomach_kcal(t))
    cat.intake_today += eaten


def finish_drink(cat, step, t, minutes, world, listener):
    want, _ = _drink_plan(cat, step.target, step.t_arrive)
    ml = min(want, minutes * RULES["water"]["drink_ml_per_min"], step.target.props["ml"])
    step.target.props["ml"] -= ml
    cat.meters["dehydration_pct"].add(t, -ml / cat.weight_g * 100)


def _empty_bladder_and_bowel(cat, t):
    """Returns what the kitten produced: a list containing 'urine' and/or 'feces'."""
    out = []
    if cat.value("bladder", t) >= 0.5:
        out.append("urine")
        cat.meters["bladder"].set(t, 0.0)
    if cat.value("bowel", t) >= 0.5:
        out.append("feces")
        cat.meters["bowel"].set(t, 0.0)
    return out


def finish_litter_box(cat, step, t, minutes, world, listener):
    box = step.target.props
    kidney_flush = any(c.name == "lily" and c.stage(t) == "polyuria" for c in cat.conditions)
    for kind in _empty_bladder_and_bowel(cat, t):
        box[kind] += 1
        if kind == "urine" and kidney_flush:
            box["big_clumps"] += 1


def finish_soil_floor(cat, step, t, minutes, world, listener):
    for kind in _empty_bladder_and_bowel(cat, t):
        world.add("mess", cat.tile, {"type": kind, "where": SOILING_PLACE.get(cat.room, "on the floor"), "t": t})
        listener.event("soiling", cat)


def finish_investigate_flowers(cat, step, t, minutes, world, listener):
    """Sniffing and pawing at lilies may expose the kitten (pollen, a chewed leaf) and may knock the vase over."""
    vase = step.target
    vase.props["investigated"] = vase.props.get("investigated", 0) + 1
    if not vase.props.get("lilies"):
        return
    low, high = RULES["lily"]["exposure_prob_per_visit"]
    exposed = listener.cat_rng.random() < low + (high - low) * cat.traits["impulsiveness"]
    if exposed and not any(c.name == "lily" for c in cat.conditions):
        listener.condition_started(LilyToxicosis.start(t, listener.cat_rng), cat)
    if exposed and not vase.props.get("knocked") and listener.cat_rng.random() < RULES["lily"]["crash_prob"]:
        vase.props["knocked"] = True
        vase.props["flowers"] += ", knocked over, petals and orange pollen scattered on the table"
        listener.sound("crash", cat.room, VERY_LOUD)


# ── specials with another kitten ──────────────────────────────────────────
# Only the kitten that started the activity (step.lead) applies the result, so it happens once per pair.

def finish_social_play(cat, step, t, minutes, world, listener):
    """Playing together builds the bond. Rough play between kittens that don't get along can turn into a fight."""
    other = step.partner
    if not step.lead or not other.alive or other.away:
        return
    listener.change_bond(cat, other, SOCIAL["play_bond_gain"])
    escalation = SOCIAL["play_escalation_base"] + SOCIAL["play_escalation_per_bond"] * max(0.0, 0.2 - cat.bond(other))
    if listener.cat_rng.random() < escalation:
        listener.fight(cat, other)


def finish_groom_kitten(cat, step, t, minutes, world, listener):
    other = step.partner
    if other.alive and not other.away:
        listener.change_bond(cat, other, SOCIAL["groom_bond_gain"])
        other.meters["stress"].add(t, -0.03)


def finish_sleep_with_kitten(cat, step, t, minutes, world, listener):
    if step.partner.alive and not step.partner.away:
        listener.change_bond(cat, step.partner, SOCIAL["sleep_bond_gain"])


def finish_hiss(cat, step, t, minutes, world, listener):
    """A hiss stresses the other kitten. If the other kitten is bold and the two really dislike each other, it
    turns into a fight."""
    other = step.partner
    if not other.alive or other.away or other.room != cat.room:
        return
    other.meters["stress"].add(t, 0.05)
    cat.last_conflict_t = other.last_conflict_t = t
    listener.change_bond(cat, other, -SOCIAL["hiss_bond_loss"])
    fight_chance = min(0.8, 0.15 + 0.5 * -cat.bond(other)) * (0.5 + other.traits["boldness"])
    if listener.cat_rng.random() < fight_chance:
        listener.fight(cat, other)


SOILING_PLACE = {"living_room": "behind the couch", "bedroom": "on the bed", "hallway": "on the doormat",
                 "bathroom": "on the bath mat"}

SPECIAL_RELIEF = {
    "eat": lambda cat, s: max(cat.urgency("hunger", s.t), min(1.0, cat.deficit_days(s.t))),
    "drink": lambda cat, s: cat.urgency("dehydration_pct", s.t),
    "litter_box": lambda cat, s: max(cat.urgency("bladder", s.t), cat.urgency("bowel", s.t)),
    "soil_floor": lambda cat, s: max(cat.urgency("bladder", s.t), cat.urgency("bowel", s.t)),
    "beg": lambda cat, s: 0.8 * max(cat.urgency("hunger", s.t), min(1.0, cat.deficit_days(s.t))),
}

SPECIAL_FINISH = {"eat": finish_eat, "drink": finish_drink, "litter_box": finish_litter_box,
                  "soil_floor": finish_soil_floor, "investigate_flowers": finish_investigate_flowers,
                  "social_play": finish_social_play, "groom_kitten": finish_groom_kitten,
                  "sleep_with_kitten": finish_sleep_with_kitten, "hiss": finish_hiss}


def _product(values):
    out = 1.0
    for v in values:
        out *= v
    return out
