"""The game the agent plays: what it reads, what its actions do, vet visits, and the engine loop.

Sections:
    text         every sentence the agent reads (fixed templates, no LLM)
    vet          a vet visit: diagnose, treat, bill
    actions      one method per agent tool (mixed into KittenEnv)
    engine       KittenEnv: the event loop, the owner's schedule, one or more kittens, the replay trace

Several kittens: they share the apartment, bowls and litter boxes, get along or don't (cat.py: bonds), can pass cat
flu to each other, and one of them can be a newcomer that arrives partway through (`newcomer_day`).
"""

import inspect
import json
import random
from copy import deepcopy
from dataclasses import dataclass, field

from .cat import INTERACTIONS, SOCIAL, SPECIAL_FINISH, URI, Cat, Step
from .scoring import CatMetrics, Metrics
from .sim import (APARTMENT, EARLIEST_BEDTIME, EPISODE_START, LOUD, MINUTES_PER_DAY, MINUTES_PER_HOUR,
                  MOVABLE, NEEDS, NORMAL, QUIET, REMOVABLE, RULES, VERY_LOUD, WAKE_TIME, WORK_END, WORK_START,
                  Apartment, EventQueue, day_name, day_of, hm, is_workday, minute_of_day, target_gain_g_per_day)


# ════ text ════════════════════════════════════════════════════════════════════════════════════════
# Everything the agent reads: how objects, sounds and the kitten are described in plain English.
#
# Observations are rendered from state with fixed templates (no LLM), so the same state always produces the
# same text. Nothing here changes the simulation.

def pretty(name):
    """'living_room' -> 'living room'."""
    return name.replace("_", " ")


# ── objects ──────────────────────────────────────────────────────────────

def describe(obj, t, detail=False):
    """One-line description of an object. `detail=True` (from examine) adds exact amounts and finer clues."""
    p, kind = obj.props, obj.kind
    if kind == "food_bowl":
        return _describe_food_bowl(p, t, detail)
    if kind == "water_bowl":
        return _describe_water_bowl(p["ml"], detail)
    if kind == "litter_box":
        return _describe_litter_box(p, detail)
    if kind == "door":
        return f"the {pretty(obj.id)} ({'open' if p['open'] else 'closed'})"
    if kind == "vase":
        return f"a vase of {p['flowers']}"
    if kind == "mess":
        noun = {"vomit": "a puddle of vomit", "urine": "a urine puddle", "feces": "a pile of cat poop"}[p["type"]]
        return f"{noun} {p['where']}"
    if kind == "toy":
        return f"a {p['name']}"
    if kind == "towel":
        return f"a {p['name']}"   # what it smells of is told by examine(); see _describe_towel
    if kind == "carrier":
        return "a cat carrier"
    if kind in ("bed", "couch", "window"):
        return {"bed": "your bed", "couch": "a couch", "window": "a window with a wide sill"}[kind]
    return obj.id


def _describe_towel(p, t, cats):
    """examine() on a towel: whose smell it carries, if it was rubbed on a kitten in the last day."""
    if p["scent_of"] >= 0 and t - p["scented_t"] < RULES["introductions"]["scent_fresh_hours"] * MINUTES_PER_HOUR:
        return f"A {p['name']} that smells of {cats[p['scent_of']].name}"
    return f"A clean {p['name']}"


def _describe_food_bowl(p, t, detail):
    parts = []
    if p["kibble_g"] > 0.5:
        parts.append(_grams(p["kibble_g"], "kibble") if detail else _rough_amount(p["kibble_g"], "kibble"))
    if p["wet_g"] > 0.5:
        spoiled = t - p["wet_added_t"] > RULES["food"]["wet_spoil_minutes"]
        warm = " (warm)" if p["warm_until"] > t else ""
        wet = _grams(p["wet_g"], "wet food") if detail else _rough_amount(p["wet_g"], "wet food")
        parts.append(wet + warm + (" that has dried out and smells off" if spoiled else ""))
    return "a food bowl with " + " and ".join(parts) if parts else "an empty food bowl"


def _describe_water_bowl(ml, detail):
    if ml < 5:
        return "an empty water bowl"
    fraction = ml / RULES["water"]["bowl_capacity_ml"]
    level = "full" if fraction > 0.8 else "about half full" if fraction > 0.35 else "nearly empty"
    return f"a water bowl, {level}" + (f" (~{ml:.0f} ml)" if detail else "")


def _describe_litter_box(p, detail):
    deposits = p["urine"] + p["feces"]
    if deposits == 0 and p["residue"] < 1:
        text = "a clean litter box"
    elif deposits == 0:
        text = "a scooped litter box; the litter looks worn and smells a bit"
    else:
        text = (f"a litter box with {p['urine']} urine clump{'s' * (p['urine'] != 1)} "
                f"and {p['feces']} stool{'s' * (p['feces'] != 1)}")
        if detail and p["big_clumps"]:
            text += f"; {p['big_clumps']} of the urine clumps are unusually large"
    if detail and p["residue"] >= RULES["litter"]["replace_residue_threshold"]:
        text += "; the litter is saturated and needs replacing"
    return text


def _grams(g, what):
    return f"about {round(g / 5) * 5:.0f} g of {what}"


def _rough_amount(g, what):
    if g < 10:
        return f"a few bites of {what}"
    if g < 30:
        return f"a small amount of {what}"
    return f"a generous pile of {what}"


def affordances(obj):
    """The verbs shown by examine(): what the owner can do with this object via use()."""
    return {
        "food_bowl": ["fill (item=kibble|wet_food, amount=grams)", "empty", "warm (wet food, kitchen only)", "take", "put"],
        "water_bowl": ["fill (with fresh water; no item or amount needed)", "empty", "take", "put"],
        "litter_box": ["scoop", "replace_litter", "take", "put"],
        "door": ["open", "close"],
        "vase": ["take", "put", "throw_away"],
        "mess": ["clean"],
        "toy": ["take", "put", "throw_away"],
        "towel": ["rub (on a kitten, with cat=<name>: the towel picks up that kitten's smell)", "take", "put"],
    }.get(obj.kind, [])


# ── sounds ───────────────────────────────────────────────────────────────

def sound_text(kind, room):
    where = f"from the {pretty(room)}"
    return {
        "retching": f"retching and a wet splat {where}",
        "crash": f"a crash {where}",
        "crying": f"loud, insistent crying {where}",
        "meowing": f"meowing {where}",
        "sneezing": f"sneezing {where}",
        "crunching": f"crunching {where}",
        "thumping": f"small thumps and skittering {where}",
        "litter_scratching": f"litter being scratched {where}",
        "hissing": f"hissing {where}",
        "fight": f"hissing, yowling and a scuffle {where}",
    }.get(kind, kind)


# ── the kitten ───────────────────────────────────────────────────────────

def describe_activity(cat, t):
    """What the owner sees the kitten doing (from the interaction's `description` in kitten.toml)."""
    if not cat.alive:
        return f"{cat.name} is lying still. She is not breathing."
    inter = cat.step.interaction
    template = inter.description_unwell if inter.description_unwell and cat.lethargic(t) else inter.description
    target, partner = cat.step.target, cat.step.partner
    obj_name = target.props.get("name", pretty(target.kind)) if target else ""
    return template.format(name=cat.name, spot=cat.step.spot, object=obj_name, partner=partner.name if partner else "")


INSPECT_REGIONS = ("eyes", "nose", "mouth", "gums", "skin", "coat", "belly", "abdomen", "rear", "overall")


def inspect_text(cat, region, t, bcs):
    """What the owner finds when checking one part of the kitten. None for an unknown region.

    Clues come from the kitten's illness signs, hydration (gums, skin tent) and body condition (bcs, 1-9).
    """
    dehydration = cat.dehydration_pct(t)
    signs = [s for cond in cat.conditions for s in cond.signs(t)]
    lethargic = cat.lethargic(t)
    scratched = cat.scratch_until > t   # from a fight with another kitten
    if region == "eyes":
        eye = [s for s in signs if "eye" in s]
        if eye:
            return f"Her eyes: {eye[0]}."
        return "Her eyes look a little dull." if lethargic else "Her eyes are bright and clear."
    if region == "nose":
        nose = [s for s in signs if "nasal" in s]
        return f"Her nose: {nose[0]}." if nose else "Her nose is clean and slightly moist."
    if region in ("mouth", "gums"):
        if "drooling" in signs:
            return "She's drooling, and her gums are pink."
        return "Her gums feel tacky and dry." if dehydration >= 5 else "Her gums are pink and moist."
    if region == "skin":
        # The skin-tent test for dehydration.
        if dehydration >= 8:
            return "When you gently lift the skin on her shoulders it stays tented."
        if dehydration >= 5:
            return "The skin over her shoulders returns slowly when lifted."
        return "The skin over her shoulders springs right back when lifted."
    if region == "coat":
        scratch = " There's a fresh scratch on her ear." if scratched else ""
        return ("Her coat looks unkempt." if lethargic else "Her coat is soft and clean.") + scratch
    if region in ("belly", "abdomen"):
        if any(s in signs for s in ("hunched", "very weak")):
            return "She tenses and flinches when you touch her belly."
        thin = " You can feel her ribs and spine easily." if bcs <= 3 else ""
        return "Her belly is soft and doesn't seem sore." + thin
    if region == "rear":
        return "Her rear end is clean."
    if region == "overall":
        bits = []
        if "warm to the touch" in signs:
            bits.append("she feels hot")
        if any("nasal" in s for s in signs):
            bits.append("her breathing sounds congested")
        if cat.deficit_days(t) > RULES["fasting"]["critical_deficit_days"] or "very weak" in signs:
            bits.append("she's weak and floppy")
        elif lethargic:
            bits.append("she's quiet and listless")
        elif cat.activity.startswith("sleep"):
            bits.append("she was asleep; she blinks at you, relaxed, then closes her eyes again")
        else:
            bits.append("she's alert and wriggly")
        if scratched:
            bits.append("she has a fresh scratch on her ear")
        return "Overall: " + ", ".join(bits) + "."
    return None


# ════ vet ═════════════════════════════════════════════════════════════════════════════════════════
# A vet visit: the owner takes one kitten in, the vet diagnoses and treats whatever it has.
#
# The vet is deliberately optimistic: it always finds the real problem, whatever the owner says. What the agent
# controls is *whether* and *how fast* the kitten gets to the vet, which decides whether treatment still works
# (lilies) and how long the kitten went without care (cat flu).

CATCH_MINUTES = 5                 # getting the kitten into the carrier
CATCH_MINUTES_HIDING = 15         # ... when it's hiding
HOSPITAL_DEATH_HOURS = (12, 44)   # when a failed lily treatment ends in death, hours after treatment starts


def vet_visit(env, kind, cat):
    """Run a whole visit ("emergency" or "appointment") for one kitten: diagnose, treat, bill, advance the clock.

    Returns the text the owner reads about the visit.
    """
    v, t_start = RULES["vet"], env.t
    if not cat.alive or cat.away:
        return "The vet visit is cancelled." if len(env.cats) == 1 else f"The vet visit for {cat.name} is cancelled."

    catch_minutes = CATCH_MINUTES_HIDING if cat.hidden else CATCH_MINUTES
    t = env.t + catch_minutes  # when the exam (and any treatment) starts
    trip_minutes = v["er_trip_minutes"] if kind == "emergency" else v["appointment_minutes"]
    env.spend(v["er_cost"] if kind == "emergency" else v["exam_cost"])

    findings, necessary, admitted = [], False, False
    for cond in cat.conditions:
        stage = cond.stage(t)
        if cond.name == "lily" and stage != "resolved" and cond.treated_t is None:
            necessary = admitted = True
            findings.append(_treat_lily(env, cat, cond, stage, t))
        elif cond.name == "uri" and stage in ("mild", "severe"):
            necessary = True
            findings.append(_treat_uri(env, cat, cond, stage, t))

    underfed = cat.deficit_days(env.t) > RULES["fasting"]["weak_deficit_days"]
    dehydration = cat.dehydration_pct(env.t)
    if dehydration >= RULES["dehydration"]["signs_pct"] or underfed:
        necessary = True
        env.spend(v["fluids_cost"])
        findings.append(f"She is dehydrated (~{dehydration:.0f}%) and underfed; given fluids and a feeding.")
        cat.meters["dehydration_pct"].set(env.t, 0.0)
        cat.meters["energy_debt"].set(env.t, 0.0)
        cat.meters["hunger"].set(env.t, 0.3)

    bcs = env.body_condition(cat)
    verdict = "ideal" if 4 <= bcs <= 5 else "underweight" if bcs < 4 else "overweight"
    findings.append(f"Weight {cat.weight_g / 1000:.2f} kg, body condition {bcs}/9 ({verdict}).")
    if not necessary:
        findings.insert(0, "Exam normal: a healthy kitten. No treatment needed.")
    env.metrics.vet_visits.append({"t": t_start, "kind": kind, "cat": cat.name, "necessary": necessary})

    # The trip. The kitten stays at the hospital if admitted (the discharge event brings it home). Any other
    # kittens stay home and carry on.
    env.cat_leaves(cat, "vet")
    env.mode, env.owner_room = "vet", None
    env.advance(catch_minutes + trip_minutes)
    env.come_home()
    if not admitted and cat.alive:
        env.cat_returns(cat)
    label = "emergency vet" if kind == "emergency" else "vet appointment"
    return f"({hm(t_start)}) You take {cat.name} to the {label}. " + " ".join(findings) + f" You're back home at {hm(env.t)}."


def _treat_lily(env, cat, cond, stage, t):
    """Admit the kitten for 48 h of IV fluids. Whether it survives was decided by how late this is."""
    r = RULES["lily"]
    cat.metrics.lily_treated_t = t
    survives = cond.treat(t, env.cat_rng)
    env.queue.invalidate(f"lily_vomits:{cat.idx}")   # no more vomiting at home
    env.spend(RULES["vet"]["lily_hospital_cost"])
    discharge_t = t + r["hospital_hours"] * MINUTES_PER_HOUR
    if survives:
        env.queue.push(discharge_t, "hospital_discharge", {"cat": cat.idx})
    else:
        env.queue.push(t + env.cat_rng.randint(*HOSPITAL_DEATH_HOURS) * MINUTES_PER_HOUR, "hospital_death", {"cat": cat.idx})
    return (f"Bloodwork and history point to lily poisoning (stage: {stage}). {cat.name} is admitted for "
            f"{r['hospital_hours']} hours of IV fluids.")


def _treat_uri(env, cat, cond, stage, t):
    """Give the first antibiotic dose and send the rest of the course home with the owner."""
    r = RULES["uri"]
    cat.metrics.uri_treated_t = cat.metrics.uri_treated_t or t
    cond.treat(t, env.cat_rng)
    env.schedule_condition(cat, cond)     # appetite returns, and the course ends, at new times
    doses_home = r["doses_per_day"] * r["course_days"] - 1
    env.inventory["antibiotic_doses"] += doses_home
    env.spend(RULES["vet"]["uri_meds_cost"])
    return (f"Diagnosis: upper respiratory infection ({stage}). First antibiotic dose given; {doses_home} doses sent "
            f"home, give one twice a day for {r['course_days']} days. Warmed wet food will help her appetite while "
            f"her nose is blocked.")


# ════ actions ═════════════════════════════════════════════════════════════════════════════════════
# Everything the agent (the owner) can do. One method per tool; players.py describes them to the LLM.
#
# Each `tool_<name>` method:
#   - checks the action is possible (raising ActionError with a helpful message if not),
#   - advances the clock by how long the action takes (the kittens keep living meanwhile),
#   - returns the text the agent reads.
#
# Actions on a kitten take an optional `cat` (its name). It's only required when there's more than one kitten.
# These are mixed into KittenEnv (below), which provides the clock, the world and the helpers used here.

SHOP = RULES["shop"]
SHOP_PRICES = {"kibble_bag": SHOP["kibble_bag"]["cost"], "wet_food": SHOP["wet_food"]["cost"],
               "litter": SHOP["litter_bag"]["cost"], "toy_wand": SHOP["toy_wand"]["cost"],
               "litter_box": SHOP["litter_box"]["cost"], "food_bowl": SHOP["food_bowl"]["cost"],
               "water_bowl": SHOP["water_bowl"]["cost"]}
SHOP_MENU = ("kibble_bag (1.5 kg, $25), wet_food (85 g can, $2), litter (one full refill, $15), toy_wand ($10), "
             "litter_box (an extra box with litter, $30), food_bowl ($8), water_bowl ($8)")
DELIVERED_OBJECTS = ("litter_box", "food_bowl", "water_bowl")   # delivered items that are objects in the apartment
MAX_WAIT_MINUTES = 600
SCALE_NOISE_G = 3             # a kitchen/baby scale reads within a few grams (standard deviation)
MAX_PLAY_MINUTES = 30
LISTEN_WINDOW_MINUTES = 15    # listen() reports sounds from the last this-many minutes
WARM_FOOD_MINUTES = 40        # warmed wet food stays warm (and smelly) this long


def _rooms(n):
    return f"{n} room" + ("s" if n != 1 else "")


def _number(value, name):
    """An agent-supplied number, or an ActionError the agent can read."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ActionError(f"{name} must be a number, not {value!r}") from None


class ActionError(Exception):
    """An action that can't be done. The message is shown to the agent as 'Error: ...'.

    Usually raised before any time passes, so a failed action costs no game time."""


class Actions:
    # ── perceive ───────────────────────────────────────────────────────
    def tool_look(self):
        self.advance(1)
        return self.look_text()

    def tool_listen(self):
        """Sounds from other rooms in the last few minutes, plus any kitten's current activity if it makes noise."""
        self.advance(1)
        recent = [s for s in self.sounds if s.t > self.t - LISTEN_WINDOW_MINUTES and s.room != self.owner_room]
        heard = sorted({sound_text(s.kind, s.room) for s in recent})
        for cat in self.cats:
            ongoing = cat.step.interaction.sound if cat.step else None
            if ongoing and not cat.away and cat.room != self.owner_room:
                heard.append(sound_text(ongoing[0], cat.room))
        return "You listen. You hear " + ("; ".join(sorted(set(heard))) if heard else "nothing but quiet") + "."

    def _room_arg(self, room):
        room = str(room).strip().lower().replace(" ", "_")
        if room not in self.world.rooms:
            raise ActionError(f"no room called '{room}'. Rooms you know of are reachable via exits.")
        return room

    def tool_go_to(self, room):
        room = self._room_arg(room)
        if room == self.owner_room:
            return "You're already here.\n" + self.look_text()
        self.owner_room, self.owner_tile = room, self.world.room_center(room)
        self.advance(1)
        return self.look_text()

    def tool_find_cat(self, cat=None):
        """Walk room to room (1 min each, starting here) until the kitten is found. With several kittens and no
        name given, search until every kitten has been seen and report where each one is."""
        if cat is not None or len(self.cats) == 1:
            return self._find_one(self.resolve_cat(cat))
        home = [c for c in self.cats if c.alive and not c.away]
        lines = [f"{c.name} {AWAY_TEXT[c.away_reason]}" for c in self.cats if c.alive and c.away]
        if not home:
            return ". ".join(lines) + "." if lines else "There are no kittens at home."
        search_order = [self.owner_room] + [r for r in self.world.rooms if r != self.owner_room]
        last_room = max((c.room for c in home), key=search_order.index)
        minutes = search_order.index(last_room) + 1
        self.owner_room, self.owner_tile = last_room, self.world.room_center(last_room)
        self.advance(minutes)
        found = [f"{c.name} in the {pretty(c.room)} ({describe_activity(c, self.t).removeprefix(c.name + ' is ')})"
                 for c in self.cats if c.alive and not c.away]
        return (f"You search the apartment ({_rooms(minutes)}, {minutes} min) and find " + "; ".join(found) + "."
                + "".join(f" {line}." for line in lines) + "\n" + self.look_text())

    def _find_one(self, cat):
        if not cat.alive:
            return f"{cat.name} has died."
        if cat.away:
            return f"{cat.name} {AWAY_TEXT[cat.away_reason]}."
        search_order = [self.owner_room] + [r for r in self.world.rooms if r != self.owner_room]
        minutes = search_order.index(cat.room) + 1
        self.owner_room, self.owner_tile = cat.room, self.world.room_center(cat.room)
        self.advance(minutes)
        if cat.room != self.owner_room:
            return (f"You search the apartment ({_rooms(minutes)}, {minutes} min), but {cat.name} slips past you "
                    f"into the {pretty(cat.room)}.\n" + self.look_text())
        return f"You find {cat.name} in the {pretty(cat.room)} ({_rooms(minutes)} searched, {minutes} min).\n" + self.look_text()

    def tool_examine(self, object):
        obj = self.find_object(object)
        self.advance(1)
        # Looking at the bed or couch finds a kitten hiding under it.
        for cat in self.cats:
            if self.cat_here(cat) and cat.hidden and cat.step.target is not None and cat.step.target.tile == obj.tile:
                return f"You look {cat.step.spot}. {cat.name} is hiding there, eyes wide."
        verbs = affordances(obj)
        text = _describe_towel(obj.props, self.t, self.cats) if obj.kind == "towel" else describe(obj, self.t, detail=True).capitalize()
        return text + "." + (f" You could: {', '.join(verbs)}." if verbs else "")

    def tool_inspect_cat(self, region, cat=None):
        cat = self.resolve_cat(cat)
        if not self.catch(cat):
            self.advance(2)
            return f"{cat.name} is wedged {cat.step.spot}, out of reach."
        self.advance(2)
        region = region.lower()
        result = inspect_text(cat, region, self.t, self.body_condition(cat))
        if result is None:
            raise ActionError("region must be one of " + ", ".join(r for r in INSPECT_REGIONS if r not in ("gums", "abdomen")))
        return result

    def tool_weigh(self, cat=None):
        cat = self.resolve_cat(cat)
        if not self.catch(cat):
            self.advance(2)
            return f"{cat.name} is wedged {cat.step.spot}, out of reach."
        self.advance(2)
        reading = round(cat.weight_g + self.cat_rng.gauss(0, SCALE_NOISE_G))
        return f"You weigh {cat.name}: {reading} g."

    # ── interact with objects: use(verb, object, ...) ──────────────────
    def tool_use(self, verb, object, item=None, amount=None, cat=None):
        verb = verb.lower().strip()
        obj = self.find_object(object)
        handler = {"fill": self._use_fill, "empty": self._use_empty, "warm": self._use_warm,
                   "scoop": self._use_scoop, "replace_litter": self._use_replace_litter, "clean": self._use_clean,
                   "throw_away": self._use_throw_away, "take": self._use_take, "put": self._use_put,
                   "open": self._use_door, "close": self._use_door, "rub": self._use_rub}.get(verb)
        result = handler(obj, verb=verb, item=item, amount=amount, cat=cat) if handler else None
        if result is None:
            raise ActionError(f"you can't '{verb}' the {obj.id}. Try examine('{obj.id}') to see what you can do with it.")
        return result

    # Each _use_* handler returns the result text, or None if the verb doesn't apply to this object.

    def _use_fill(self, obj, item, amount, **_):
        inv = self.inventory
        if obj.kind == "water_bowl":
            obj.props["ml"] = RULES["water"]["bowl_capacity_ml"]
            self.advance(1)
            return "You fill the water bowl with fresh water."
        if obj.kind != "food_bowl":
            return None
        if item in ("kibble", None):
            grams = _number(amount or 0, "amount")
            if grams <= 0:
                raise ActionError("say how many grams, e.g. amount=30")
            grams = min(grams, inv["kibble_g"])
            if grams <= 0:
                raise ActionError("you're out of kibble.")
            inv["kibble_g"] -= grams
            obj.props["kibble_g"] += grams
            self.advance(1)
            return f"You pour {grams:.0f} g of kibble into the bowl. {inv['kibble_g']:.0f} g left in the bag."
        if item in ("wet_food", "wet"):
            cans = max(1, int(_number(amount or 1, "amount")))
            if inv["wet_food_cans"] < cans:
                raise ActionError(f"you have {inv['wet_food_cans']} can(s) of wet food.")
            inv["wet_food_cans"] -= cans
            grams = cans * SHOP["wet_food"]["grams"]
            obj.props["wet_g"] += grams
            obj.props["wet_added_t"] = self.t
            self.advance(1)
            return f"You spoon {cans} can(s) of wet food ({grams} g) into the bowl."
        raise ActionError("item must be 'kibble' or 'wet_food'")

    def _use_empty(self, obj, **_):
        if obj.kind == "food_bowl":
            obj.props.update({"kibble_g": 0.0, "wet_g": 0.0, "wet_added_t": -1})
        elif obj.kind == "water_bowl":
            obj.props.update({"ml": 0.0})
        else:
            return None
        self.advance(1)
        return f"You empty the {pretty(obj.id)}."

    def _use_warm(self, obj, **_):
        """Microwave the wet food: a sick kitten with a blocked nose eats more when it can smell the food."""
        if obj.kind != "food_bowl":
            return None
        if obj.props["wet_g"] < 1 or self.owner_room != "kitchen":
            raise ActionError("you can warm wet food in the microwave; the bowl needs wet food and you need to be in the kitchen.")
        obj.props["warm_until"] = self.t + WARM_FOOD_MINUTES
        self.advance(2)
        return "You warm the wet food for a few seconds in the microwave. It smells strongly now."

    def _use_scoop(self, obj, **_):
        """Remove the deposits. Each scoop leaves a little smell behind, so the litter needs replacing eventually."""
        if obj.kind != "litter_box":
            return None
        p = obj.props
        deposits, big = p["urine"] + p["feces"], p["big_clumps"]
        p.update(urine=0, feces=0, big_clumps=0)
        p["residue"] += RULES["litter"]["residue_per_scoop"] * (deposits > 0)
        self.advance(3)
        if not deposits:
            return "Nothing to scoop."
        extra = f" {big} of the urine clumps were unusually large." if big else ""
        return f"You scoop out {deposits} clump(s).{extra}"

    def _use_replace_litter(self, obj, **_):
        if obj.kind != "litter_box":
            return None
        if self.inventory["litter_refills"] < 1:
            raise ActionError("you have no fresh litter. shop(item='litter').")
        self.inventory["litter_refills"] -= 1
        obj.props.update(urine=0, feces=0, big_clumps=0, residue=0.0)
        self.advance(10)
        return "You dump the old litter, wash the box and fill it with fresh litter."

    def _use_clean(self, obj, **_):
        if obj.kind != "mess":
            return None
        del self.world.objects[obj.id]
        self.advance(3)
        return f"You clean up the {obj.props['type']}."

    def _use_throw_away(self, obj, **_):
        if obj.kind not in REMOVABLE:
            return None
        if obj.kind == "vase" and obj.props.get("lilies") and self.metrics.lily_removed_t is None:
            self.metrics.lily_removed_t = self.t
        del self.world.objects[obj.id]
        if self.carrying == obj.id:
            self.carrying = None
        self.advance(2)
        return f"You take the {pretty(obj.id)} outside to the trash."

    def _use_take(self, obj, **_):
        if obj.kind not in MOVABLE:
            return None
        if self.carrying:
            raise ActionError(f"you're already carrying {self.carrying}.")
        self.carrying = obj.id
        self.world.place(obj, None)
        self.advance(1)
        return f"You pick up the {pretty(obj.id)}."

    def _use_put(self, obj, **_):
        if self.carrying != obj.id:
            return None
        self.carrying = None
        self.world.place(obj, self.owner_tile)
        self.advance(1)
        return f"You put the {pretty(obj.id)} down in the {pretty(self.owner_room)}."

    def _use_rub(self, obj, cat, **_):
        """Rub a towel over a kitten's cheeks and chin: the towel carries that kitten's smell for a day, and another
        kitten that can smell it gets used to it (rules.toml [introductions])."""
        if obj.kind != "towel":
            return None
        if cat is None and len(self.cats) > 1:
            raise ActionError("say which kitten to rub the towel on, e.g. cat=\"" + self.cats[0].name + "\"")
        kitten = self.resolve_cat(cat)
        if not self.catch(kitten):
            self.advance(1)
            return f"{kitten.name} is wedged {kitten.step.spot}, out of reach."
        obj.props.update(scent_of=kitten.idx, scented_t=self.t)
        self.advance(2)
        return f"You rub the {obj.props['name']} gently over {kitten.name}'s cheeks and chin. It smells of {kitten.name} now."

    def _use_door(self, obj, verb, **_):
        if obj.kind != "door":
            return None
        obj.props["open"] = verb == "open"
        self.advance(1)
        return f"You {verb} the {pretty(obj.id)}."

    # ── interact with a kitten ─────────────────────────────────────────
    def tool_give_medication(self, cat=None):
        cat, inv = self.resolve_cat(cat), self.inventory
        if inv["antibiotic_doses"] < 1:
            raise ActionError("you don't have any medication.")
        if not self.catch(cat):
            self.advance(3)
            return f"{cat.name} is wedged {cat.step.spot}; you can't get hold of her."
        inv["antibiotic_doses"] -= 1
        for cond in cat.conditions:
            if cond.name == "uri" and cond.treated_t is not None:
                cond.doses_given.append(self.t)
        cat.meters["trust"].add(self.t, -0.02)  # the kitten dislikes it
        self.advance(3)
        return f"You give {cat.name} a dose of antibiotics. She complains but swallows it. {inv['antibiotic_doses']} doses left."

    def tool_play(self, minutes=10, cat=None, with_cat=None):
        """Play lowers boredom and stress and builds trust. Only works if the kitten is here, awake and not unwell.
        With `with_cat`, play with two kittens together (see _play_together)."""
        minutes = max(1, min(MAX_PLAY_MINUTES, _number(minutes, "minutes")))
        cat = self.resolve_cat(cat)
        if with_cat is not None:
            return self._play_together(cat, self.resolve_cat(with_cat), minutes)
        refusal = self._why_not_now(cat, wants="play")
        if refusal:
            self.advance(2)
            return f"You try to play, but {refusal}."
        toys = [o for o in self.world.of_kind("toy") if o.room == self.owner_room] or self.world.of_kind("toy")
        self.force_cat_step(cat, "play_with_toy", minutes, toys[0] if toys else None)  # its play effects apply
        self.advance(minutes)
        cat.meters["trust"].add(self.t, 0.02 * minutes / 10)
        cat.meters["stress"].add(self.t, -0.05)
        return f"You play with {cat.name} for {minutes} minutes. She chases and pounces enthusiastically."

    def _play_together(self, a, b, minutes):
        """A supervised play session with two kittens (AAFP: short sessions with toys build positive associations;
        stop at the first hiss). The longer the session and the more the two dislike each other, the likelier it
        turns tense. Calm: the bond grows. Tense: the session stops at a hiss, sometimes a scuffle."""
        if a is b:
            raise ActionError("with_cat must be a different kitten")
        for kitten in (a, b):
            refusal = self._why_not_now(kitten, wants="play")
            if refusal:
                self.advance(2)
                return f"You try to play with {a.name} and {b.name} together, but {refusal}."
        intro = RULES["introductions"]
        hostility = max(0.0, -a.bond(b))
        per_10 = intro["session_tension_base"] + intro["session_tension_per_10min"] * hostility
        tense = self.cat_rng.random() < 1 - (1 - min(1.0, per_10)) ** (minutes / 10)
        played = self.cat_rng.uniform(1, minutes) if tense else minutes
        toys = [o for o in self.world.of_kind("toy") if o.room == self.owner_room] or self.world.of_kind("toy")
        for kitten in (a, b):
            self.force_cat_step(kitten, "play_with_toy", played, toys[0] if toys else None)
        self.advance(played)
        for kitten in (a, b):
            kitten.meters["trust"].add(self.t, 0.02 * played / 10)
        together = f"You play with {a.name} and {b.name} together, dangling the toy between them"
        if not tense:
            self.change_bond(a, b, intro["session_calm_gain"])
            for kitten in (a, b):
                kitten.meters["stress"].add(self.t, -0.05)
            return f"{together} for {round(played)} minutes. They stay relaxed, and by the end they're batting at the same toy."
        self.change_bond(a, b, -intro["session_tense_bond_loss"])
        a.last_conflict_t = b.last_conflict_t = self.t
        for kitten in (a, b):
            kitten.meters["stress"].add(self.t, 0.05)
        self.sound("hissing", self.owner_room, NORMAL)
        text = f"{together}. After {round(played)} minutes {b.name} flattens her ears and hisses at {a.name}"
        if self.cat_rng.random() < intro["session_fight_chance"] * hostility:
            self.fight(a, b)
            return text + ", and the two tumble into a brief scuffle before they split up. You stop the session."
        return text + ". You stop the session."

    def tool_pet(self, cat=None):
        """Builds trust if the kitten already trusts the owner a little; a wary kitten may back away."""
        cat = self.resolve_cat(cat)
        if not self.cat_here(cat):
            self.advance(1)
            return f"{cat.name} isn't here."
        if cat.hidden:
            self.advance(1)
            return f"{cat.name} is hiding {cat.step.spot}, out of reach."
        if cat.activity.startswith("sleep"):
            self.advance(2)
            return f"{cat.name} is asleep. She stirs at your touch, then settles back down."
        self.advance(2)
        if cat.value("trust", self.t) > 0.3 or self.cat_rng.random() < 0.4:
            cat.meters["trust"].add(self.t, 0.03)
            cat.meters["stress"].add(self.t, -0.03)
            return f"{cat.name} leans into your hand and purrs."
        cat.meters["stress"].add(self.t, 0.02)
        return f"{cat.name} flattens her ears and backs away."

    def tool_carry(self, cat, room):
        """Pick a kitten up and carry it to a room (2 min), e.g. to keep it behind a closed door. Also lets a newcomer
        out of its carrier. The kitten may walk straight back out if the door is open."""
        kitten = self.resolve_cat(cat)
        room = self._room_arg(room)
        if kitten.away_reason == "carrier":
            self.owner_room, self.owner_tile = room, self.world.room_center(room)
            self.release_newcomer(kitten, room)
            self.force_cat_step(kitten, "explore", SETTLE_MINUTES)   # sniffs around the new room first
            self.advance(2)
            return f"You carry the carrier to the {pretty(room)} and open it. {kitten.name} creeps out.\n" + self.look_text()
        if not self.catch(kitten):
            self.advance(1)
            return f"{kitten.name} is wedged {kitten.step.spot}, out of reach."
        self._finish_cat_step(kitten)
        kitten.meters["trust"].add(self.t, -0.02)   # most kittens dislike being carried
        kitten.meters["stress"].add(self.t, 0.03)
        self.owner_room, self.owner_tile = room, self.world.room_center(room)
        kitten.tile, kitten.room = self.owner_tile, room
        self.force_cat_step(kitten, "explore", SETTLE_MINUTES)   # sniffs around the new room first
        self.advance(2)
        return f"You pick {kitten.name} up, carry {kitten.name} to the {pretty(room)} and put {kitten.name} down.\n" + self.look_text()

    def _why_not_now(self, cat, wants):
        """Why the kitten won't join in right now (as text the owner reads), or None if it will."""
        if not self.cat_here(cat):
            return f"{cat.name} isn't here"
        if cat.hidden:
            return f"{cat.name} is hiding {cat.step.spot} and won't come out"
        if cat.activity.startswith("sleep"):
            return f"{cat.name} is asleep"
        if cat.lethargic(self.t):
            return f"{cat.name} is too listless to {wants}"
        return None

    # ── the outside world ──────────────────────────────────────────────
    def tool_shop(self, item, quantity=1):
        """Order supplies. Paid now, delivered tomorrow at the delivery hour."""
        item, qty = item.lower().strip(), max(1, int(_number(quantity, "quantity")))
        if item not in SHOP_PRICES:
            raise ActionError("items: " + SHOP_MENU)
        cost = SHOP_PRICES[item] * qty
        self.spend(cost)
        arrives = (day_of(self.t) + 1) * MINUTES_PER_DAY + SHOP["delivery_hour"] * 60
        self.queue.push(arrives, "delivery", {"item": item, "qty": qty})
        self.advance(3)
        return f"Ordered {qty} x {item} for ${cost}. Delivery tomorrow at {SHOP['delivery_hour']}:00. ${self.money:.0f} left."

    def tool_book_vet(self, urgency, reason="", cat=None):
        """'emergency' goes now; 'appointment' books the next 08:00 slot (the visit then happens automatically)."""
        cat = self.resolve_cat(cat)
        if urgency == "emergency":
            self.advance(2)
            return vet_visit(self, "emergency", cat)
        if urgency == "appointment":
            booked = next((t for t, c in self.appointments if c is cat), None)
            if booked is not None:
                raise ActionError(f"you already have an appointment on day {day_of(booked)} at {hm(booked)}.")
            slot = RULES["vet"]["appointment_hour"] * 60
            day = day_of(self.t) + (1 if minute_of_day(self.t) >= slot else 0)
            when = day * MINUTES_PER_DAY + slot
            self.appointments.append((when, cat))
            self.advance(5)
            who = f" for {cat.name}" if len(self.cats) > 1 else ""
            return f"Booked a vet appointment{who} for day {day} at {hm(when)}."
        raise ActionError("urgency must be 'emergency' (go to the emergency vet now) or 'appointment' (next 08:00 slot)")

    # ── time ───────────────────────────────────────────────────────────
    def tool_wait(self, minutes):
        """Stay where you are and let time pass. Stops early for a loud sound, the doorbell, work, bedtime or an
        appointment. Reports what the kittens did in this room meanwhile, and what any kitten here is doing now."""
        minutes = max(1, min(MAX_WAIT_MINUTES, _number(minutes, "minutes")))
        start = self.t
        self.advance(minutes, interruptible=True)
        cut = " (interrupted)" if self.interrupted and not self.done else ""
        here = self.kitten_lines()
        now = " Now: " + " ".join(here) if here else ""
        return f"You wait {round(self.t - start)} min{cut}." + self.seen_text() + now

    def tool_wait_until(self, time):
        try:
            hours, mins = (int(x) for x in time.split(":"))
        except ValueError:
            raise ActionError(f"time must look like '18:00', not {time!r}") from None
        target = day_of(self.t) * MINUTES_PER_DAY + hours * 60 + mins
        if target <= self.t:
            target += MINUTES_PER_DAY
        return self.tool_wait(min(MAX_WAIT_MINUTES, target - self.t))

    def tool_sleep(self):
        if WAKE_TIME <= minute_of_day(self.t) < EARLIEST_BEDTIME:
            raise ActionError("it's too early to go to bed. Use wait() to pass time.")
        self.owner_room = "bedroom"
        self.sleep_until_morning()
        return "You go to bed."

    # ── memory and bookkeeping (no game time) ──────────────────────────
    def tool_write_note(self, text):
        self.notes.append(f"[day {day_of(self.t)} {hm(self.t)}] {text}")
        return "Noted."

    def tool_read_notes(self):
        return "\n".join(self.notes) if self.notes else "You have no notes."

    def tool_check_inventory(self):
        inv = self.inventory
        prescribed = any(c.metrics.uri_treated_t is not None for c in self.cats)
        meds = f"Antibiotic doses: {inv['antibiotic_doses']}. " if prescribed else ""
        text = (f"Kibble: {inv['kibble_g']:.0f} g. Wet food: {inv['wet_food_cans']} can(s). "
                f"Fresh litter refills: {inv['litter_refills']}. {meds}Money: ${self.money:.0f}.")
        return text + (f" Carrying: {self.carrying}." if self.carrying else "")


# ════ engine ══════════════════════════════════════════════════════════════════════════════════════
# KittenEnv: the simulation engine. The agent plays the owner, one tool call per step.
#
#     env = KittenEnv(seed=0, n_cats=1)
#     obs = env.reset()
#     while not env.done:
#         obs = env.step({"name": "look", "arguments": {}})
#
# Time moves from event to event (sim.py), never minute by minute. Things that happen at known times are
# events in a queue: a kitten finishing an activity, a need reaching a threshold (sim.py works out when),
# an illness changing stage, a delivery, midnight. Between events nothing changes except meters moving at
# steady rates, so jumping straight to the next event loses nothing. Each kitten's events carry its index, so
# one kitten's plans never cancel another's.
#
# A step:
#   1. The tool runs (actions above). It moves time forward with `advance(minutes)`, which handles every event
#      due in that window.
#   2. The owner's schedule takes over if due (`_settle`): work, bedtime, a vet appointment.
#   3. The agent gets back the result plus anything it heard or was told meanwhile.
#
# Everything that happens is also written to a timeline (where the kittens and owner were, what they did, which
# animation) that the game viewer plays back. The agent never sees the hidden state, only text.

CAT_NAMES = ["Miso", "Tofu", "Pepper", "Biscuit", "Mochi", "Olive"]
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
START_INVENTORY = {"kibble_g": 400.0, "wet_food_cans": 0, "litter_refills": 1, "antibiotic_doses": 0}
START_MONEY = 1000.0
NIGHT_WAKE_MINUTES = 60      # woken at night, the owner may act this long before going back to bed
SOUND_HISTORY = 300          # recent sounds remembered for listen()
HIDDEN_REACH_TRUST = 0.5     # a hiding kitten lets the owner reach it only once its trust is at least this
SEEN_LOG_LENGTH = 8          # how many recent kitten activities the "What you saw ... do" summary lists
BY_NAME = {i.name: i for i in INTERACTIONS}
AWAY_TEXT = {"vet": "is at the animal hospital", "not_arrived": "hasn't arrived yet",
             "carrier": "is in the carrier in the hallway"}
SETTLE_MINUTES = 10           # a kitten carried into a room sniffs around it this long before wandering off
HOUSEHOLD_CHECK_MINUTES = 60   # with several kittens: how often cat flu spreading and bond drift are applied
VOMIT_PLACE = {"kitchen": "on the kitchen floor", "living_room": "on the rug", "bedroom": "on the bedroom floor",
               "hallway": "in the hallway", "bathroom": "on the bath mat"}


@dataclass
class Observation:
    text: str                                   # what the agent reads
    data: dict = field(default_factory=dict)    # time and room, for scripts; not shown to LLM agents
    done: bool = False


@dataclass
class Sound:
    t: float
    kind: str        # e.g. "retching", "crash", "sneezing" (see text.sound_text)
    room: str
    loudness: int    # QUIET .. VERY_LOUD (sim.py)


class KittenEnv(Actions):
    def __init__(self, seed=0, days=14, n_cats=1, cat_name="Miso", max_steps=6000, trace=True, newcomer_day=None):
        """n_cats: kittens in total. newcomer_day: if set, the last kitten isn't there at the start; a friend brings
        it on this day, and it starts out wary of the others."""
        if not 1 <= n_cats <= len(CAT_NAMES):
            raise ValueError(f"n_cats must be between 1 and {len(CAT_NAMES)}")
        if newcomer_day is not None and not (n_cats >= 2 and 1 <= newcomer_day < days):
            raise ValueError("a newcomer needs n_cats >= 2 and 1 <= newcomer_day < days")
        self.seed, self.days, self.max_steps, self.tracing = seed, days, max_steps, trace
        self.n_cats, self.newcomer_day = n_cats, newcomer_day
        # `rng` fixes the seed's setup (the kittens, which events happen when), the same whatever the agent
        # does. `cat_rng` drives everything during play.
        self.rng = random.Random(seed)
        self.cat_rng = random.Random(seed * 7919 + 1)
        self.world = Apartment()
        self.cat_names = [cat_name] + [n for n in CAT_NAMES if n != cat_name][:n_cats - 1]
        self.cats = [self._new_cat(self.cat_names[0])]   # the rest are added in _schedule_setup
        self.t = EPISODE_START
        self.queue = EventQueue()

        # The owner.
        self.owner_room = "hallway"              # None while out (work, vet)
        self.owner_tile = self.world.room_center("hallway")
        self.mode = "awake"                      # awake | asleep | work | vet
        self.carrying = None                     # id of the object in the owner's hands
        self.night_awake_until = -1              # woken at night: may act until this time
        self.worked_day = -1                     # last day the owner went to work
        self.inventory = dict(START_INVENTORY)
        self.money = START_MONEY
        self.notes = []
        self.appointments = []                   # booked regular vet appointments: (time, kitten)

        # What's happening, for observations.
        self.sounds = []                         # recent Sounds
        self.messages = []                       # news to give the agent at the end of this step
        self.seen_log = []                       # (t, room, what it did, kitten) the owner witnessed this step
        self.interrupted = False                 # something should cut a wait() short
        self._interrupt_level = LOUD
        self._deaths_announced = set()

        self.steps = 0
        self.trace = []                          # replay log for the viewer: agent steps and timeline entries
        self.metrics = Metrics()
        self._schedule_setup()
        self._initial_state = self._replay_state()
        self._log_state("initial")

    def _new_cat(self, name):
        cat = Cat.sample(self.rng, name)
        cat.idx, cat.metrics, cat.trace_id = len(getattr(self, "cats", [])), CatMetrics(), None
        return cat

    # ── setup ──────────────────────────────────────────────────────────
    def _schedule_setup(self):
        """Put the episode's fixed events in the queue and start the kittens off.

        The first kitten and the episode's events are drawn in the same order as a one-kitten episode, so adding
        kittens doesn't change what happens to the first one's scenario.
        """
        for day in range(1, self.days + 1):
            self.queue.push(day * MINUTES_PER_DAY, "midnight")
        lily = RULES["lily"]
        if self.rng.random() < lily["event_prob"]:
            day = self.rng.randint(*lily["event_day"])
            # The neighbor visits at 11:00 on weekends, 18:30 on weekdays (after work).
            self.queue.push(day * MINUTES_PER_DAY + (18 * 60 + 30 if is_workday(day) else 11 * 60), "bouquet")
        self._maybe_shelter_flu(self.cats[0])
        for name in self.cat_names[1:]:
            self.cats.append(self._new_cat(name))
            newcomer = self.newcomer_day is not None and len(self.cats) == self.n_cats
            if newcomer:
                self.cats[-1].age0_days -= self.newcomer_day   # 8 weeks old when it arrives
            self._maybe_shelter_flu(self.cats[-1], first_day=self.newcomer_day if newcomer else 0)
        if len(self.cats) > 1:
            self._set_up_household()
        # The kittens bolt out of their carriers and pick their first activities.
        carrier = self.world.objects["carrier"]
        for cat in self.cats:
            if cat.away_reason == "not_arrived":
                continue
            cat.tile, cat.room, cat.arrived_t = carrier.tile, carrier.room, self.t
            self._start_cat_step(cat)

    def _maybe_shelter_flu(self, cat, first_day=0):
        uri = RULES["uri"]
        if self.rng.random() < uri["event_prob"]:
            self.condition_started(URI.start(first_day + self.rng.randint(*uri["onset_day"]), self.rng), cat)

    def _set_up_household(self):
        """Several kittens: starting bonds, the newcomer's arrival, and the hourly household check."""
        social, nc = RULES["social"], RULES["newcomer"]
        newcomer = self.cats[-1] if self.newcomer_day is not None else None
        for i, a in enumerate(self.cats):
            for b in self.cats[i + 1:]:
                mean, noise = ((nc["start_bond_mean"], nc["start_bond_noise"]) if newcomer in (a, b)
                               else (social["start_bond_mean"], social["start_bond_noise"]))
                sociability = a.traits["sociability"] + b.traits["sociability"] - 1
                bond = mean + social["start_bond_per_sociability"] * sociability + self.rng.gauss(0, noise)
                a.bonds[b.idx] = b.bonds[a.idx] = max(-1.0, min(1.0, bond))
        if newcomer is not None:
            newcomer.away, newcomer.away_reason = True, "not_arrived"
            day = self.newcomer_day
            self.queue.push(day * MINUTES_PER_DAY + (18 * 60 + 30 if is_workday(day) else 11 * 60), "newcomer")
        for t in range(EPISODE_START + HOUSEHOLD_CHECK_MINUTES, self.days * MINUTES_PER_DAY, HOUSEHOLD_CHECK_MINUTES):
            self.queue.push(t, "household")

    def reset(self):
        """The opening observation."""
        days = f"{self.days} day{'s' if self.days != 1 else ''}"
        if len(self.cats) == 1:
            name = self.cats[0].name
            text = (f"It's Saturday 10:00. You've just brought home {name}, an 8-week-old kitten from the shelter. "
                    f"You set the carrier down in the hallway and open it, and she bolts out of sight.\n"
                    f"You live alone in a small apartment. On weekdays you leave for work at 08:30 and get home at 18:00, "
                    f"and you must be asleep between midnight and 07:00. You have ${self.money:.0f} to spend on her care.\n"
                    f"Every action takes time, and the world keeps running while you act. Your job: raise {name} well "
                    f"for the next {days}.\n\n" + self.look_text())
        else:
            here = [c for c in self.cats if c.away_reason != "not_arrived"]
            if len(here) == 1:
                adopted = (f"You've just brought home {here[0].name}, an 8-week-old kitten from the shelter. You set the "
                           f"carrier down in the hallway and open it, and {here[0].name} bolts out of sight.")
            else:
                names = ", ".join(c.name for c in here[:-1]) + " and " + here[-1].name
                adopted = (f"You've just brought home {NUMBER_WORDS[len(here)]} 8-week-old kittens from the shelter: "
                           f"{names}. You set the carriers down in the hallway and open them, and they bolt out of sight.")
            later = ""
            if self.newcomer_day is not None:
                nc, day = self.cats[-1].name, self.newcomer_day
                when = "around 18:30" if is_workday(day) else "around 11:00"
                later = (f" A friend who can't keep their 8-week-old kitten, {nc}, will bring {nc} to you on "
                         f"{day_name(day)} (day {day}) {when}.")
            text = (f"It's Saturday 10:00. {adopted}{later}\n"
                    f"You live alone in a small apartment. On weekdays you leave for work at 08:30 and get home at 18:00, "
                    f"and you must be asleep between midnight and 07:00. You have ${self.money:.0f} to spend on their care.\n"
                    f"Every action takes time, and the world keeps running while you act. When an action involves a "
                    f"kitten, say which one with cat=\"<name>\". Your job: raise all of them well for the next {days}.\n\n"
                    + self.look_text())
        full = self._header() + text
        self._log({"type": "step", "t": self.t, "step": 0, "call": None, "obs": full})
        return Observation(full, {"room": self.owner_room, "t": self.t})

    # ── the step loop ──────────────────────────────────────────────────
    @property
    def done(self):
        all_dead = not any(c.alive for c in self.cats)
        return all_dead or self.t >= self.days * MINUTES_PER_DAY or self.steps >= self.max_steps

    def step(self, call):
        """Run one tool call {"name": ..., "arguments": {...}} and return what the agent observes."""
        if self.done:
            return Observation("The episode is over.", done=True)
        self.steps += 1
        self.seen_log = []
        self.interrupted = False   # only interruptions during this action count
        t_before = self.t
        from_tile = self.owner_tile
        target = self._replay_target(call)
        text = self._run_tool(call.get("name"), call.get("arguments") or {})
        self._log({"type": "owner", "t0": t_before, "t1": self.t, "action": call.get("name"),
                   "room": self.owner_room, "tile": self.owner_tile, "from_tile": from_tile,
                   "step": self.steps, "call": deepcopy(call), "target": target, "result": text})
        self._settle()

        news = list(self.messages)
        self.messages = []
        for cat in self.cats:
            if not cat.alive and cat.idx not in self._deaths_announced:
                self._deaths_announced.add(cat.idx)
                news.append(f"{cat.name} has died ({cat.cause_of_death}).")
        if self.done:
            ending = "The episode is over." if not any(c.alive for c in self.cats) else \
                     f"{self.days} days have passed. The episode is over."
            if news and not any(c.alive for c in self.cats):
                news[-1] += " " + ending
            else:
                news.append(ending)
        full = self._header() + text + ("\n\n" + "\n".join(news) if news else "")
        self._log_state("step")
        self._log({"type": "step", "t": self.t, "step": self.steps, "call": call, "obs": full})
        return Observation(full, {"room": self.owner_room, "t": self.t, "call": call}, self.done)

    def _run_tool(self, name, args):
        """Call tool_<name>(**args). Agent mistakes come back as "Error: ..." text; simulator bugs still raise."""
        tool = getattr(self, f"tool_{name}", None)
        if tool is None:
            self.metrics.invalid_calls += 1
            return f"Error: unknown action '{name}'."
        try:
            inspect.signature(tool).bind(**args)
        except TypeError as e:
            self.metrics.invalid_calls += 1
            return f"Error: bad arguments for {name}: {e}"
        try:
            return tool(**args)
        except ActionError as e:
            self.metrics.invalid_calls += 1
            return f"Error: {e}"

    def _header(self):
        day = day_of(self.t)
        return f"[{day_name(day)} {hm(self.t)} · day {day} · {pretty(self.owner_room)}]\n"

    # ── time ───────────────────────────────────────────────────────────
    def advance(self, minutes, interruptible=False, interrupt_level=LOUD):
        """Move time forward up to `minutes`, handling every event due on the way.

        Stops early if `interruptible` and something interrupts (a sound at least `interrupt_level` loud, the
        doorbell), and always stops when the owner's schedule needs to take over (work, bedtime, appointment).
        """
        end = min(self.t + minutes, self.days * MINUTES_PER_DAY)
        if self.mode == "awake":
            end = min(end, self._next_schedule_time())
        self.interrupted = False
        self._interrupt_level = interrupt_level
        for cat in self.cats:
            self._refresh_cat(cat)  # the owner may have just moved or changed something
        self._log_state("advance_start")
        while not self.done:
            next_t = self.queue.next_time()
            if next_t is None or next_t > end:
                break
            event = self.queue.pop()
            self._move_clock(event.t)
            self._handle(event)
            if interruptible and self.interrupted:
                return
        if not self.done:
            self._move_clock(end)
        self._log_state("advance_end")

    def _move_clock(self, t):
        if t > self.t:
            for cat in self.cats:
                cat.metrics.integrate(cat, self.world, self.t, t)
            self.t = t

    def _next_schedule_time(self):
        """When the owner's schedule next takes over: an appointment, bedtime, or leaving for work."""
        t, day = self.t, day_of(self.t)
        times = [when for when, _ in self.appointments]
        if minute_of_day(t) < WAKE_TIME:
            times.append(max(t, self.night_awake_until))
        else:
            times.append((day + 1) * MINUTES_PER_DAY)  # midnight
        for d in (day, day + 1):
            if is_workday(d) and self.worked_day != d and t < d * MINUTES_PER_DAY + WORK_END:
                times.append(max(t, d * MINUTES_PER_DAY + WORK_START))
                break
        return min(times)

    def _settle(self):
        """Apply the owner's schedule until the agent is free to act again."""
        for _ in range(20):  # schedule events can chain (appointments, then late for work); 20 is plenty
            if self.done or self.t < self._next_schedule_time():
                return
            m, day = minute_of_day(self.t), day_of(self.t)
            due = [a for a in self.appointments if a[0] <= self.t]
            if due:
                appointment = min(due, key=lambda a: a[0])
                self.appointments.remove(appointment)
                self.messages.append(vet_visit(self, "appointment", appointment[1]))
            elif m < WAKE_TIME:
                self.messages.append("It's past midnight. You're exhausted and go to bed." if m < 60 else "You go back to bed.")
                self.sleep_until_morning()
            else:
                self._go_to_work(day, late=m >= WORK_START + 15)

    def _go_to_work(self, day, late):
        self.worked_day = day
        self.messages.append(f"You leave for work{' (late)' if late else ''}.")
        self.mode, self.owner_room = "work", None
        self.advance(day * MINUTES_PER_DAY + WORK_END - self.t)
        self.come_home()
        self.messages.append("You get home from work.")

    def come_home(self):
        self.mode, self.owner_room = "awake", "hallway"
        self.owner_tile = self.world.room_center("hallway")

    def sleep_until_morning(self):
        """Sleep until the alarm. A very loud sound wakes the owner, who may then act for a while."""
        self.mode, self.owner_room = "asleep", "bedroom"
        self.owner_tile = self.world.room_center("bedroom")
        tomorrow = 1 if minute_of_day(self.t) >= WAKE_TIME else 0
        alarm = (day_of(self.t) + tomorrow) * MINUTES_PER_DAY + WAKE_TIME
        self.advance(alarm - self.t, interruptible=True, interrupt_level=VERY_LOUD)
        self.mode = "awake"
        if self.t < alarm and not self.done:
            self.night_awake_until = self.t + NIGHT_WAKE_MINUTES
            last = next((s for s in reversed(self.sounds) if s.loudness >= VERY_LOUD), None)
            cause = sound_text(last.kind, last.room) if last else "a noise"
            self.messages.append(f"You're woken at {hm(self.t)} by {cause}. "
                                 f"You can act for up to an hour before going back to bed.")
        else:
            self.messages.append("Your alarm goes off. You get up.")

    # ── events ─────────────────────────────────────────────────────────
    def _handle(self, event):
        handlers = {
            "midnight": self._on_midnight,
            "bouquet": self._on_bouquet,
            "delivery": self._on_delivery,
            "cat_step_end": self._on_cat_step_end,
            "cat_body": self._on_cat_body,
            "condition_change": self._on_condition_change,
            "vomit": self._on_vomit,
            "sneeze": self._on_sneeze,
            "hospital_discharge": self._on_discharge,
            "hospital_death": self._on_hospital_death,
            "household": self._on_household,
            "newcomer": self._on_newcomer,
            "carrier_release": self._on_carrier_release,
        }
        handlers[event.kind](event)
        self._check_deaths()
        self._log_state(event.kind)

    def _cat_of(self, event):
        return self.cats[event.data["cat"]]

    def _on_midnight(self, event):
        """Daily growth update and the day's weight record, for each kitten."""
        for cat in self.cats:
            if not cat.alive or cat.arrived_t is None or cat.away_reason == "carrier":
                continue
            # The first day starts when the kitten arrives (10:00, or later for a newcomer), so it's only part of a day.
            day_fraction = (self.t - max(self.t - MINUTES_PER_DAY, cat.arrived_t)) / MINUTES_PER_DAY
            if cat.away_reason == "vet":
                cat.intake_today = cat.need_kcal(self.t) * day_fraction   # hospital staff feed the kitten
            cat.end_of_day(self.t, day_fraction)
            cat.target_g += target_gain_g_per_day(cat.age_days(self.t)) * day_fraction
            cat.metrics.weights.append((day_of(self.t), round(cat.weight_g), round(cat.target_g)))

    def _on_bouquet(self, event):
        """The neighbor brings flowers, but only when the owner is home and awake to answer the door."""
        if not self._owner_present():
            self.queue.push(self.t + 30, "bouquet")
            return
        self.world.add("vase", tuple(APARTMENT["spawn"]["vase"]),
                       {"flowers": "white lilies and pink roses", "lilies": True, "added_t": self.t})
        self.metrics.lily_arrive_t = self.t
        who = "the new kitten" if len(self.cats) == 1 else "the new kittens"
        self.messages.append(f"The doorbell rings. Your neighbor drops by with a bouquet as a welcome-home gift for "
                             f"{who}. You put the flowers in a vase on the living room coffee table.")
        self.interrupted = True

    def _on_delivery(self, event):
        item, qty = event.data["item"], event.data["qty"]
        inv = self.inventory
        if item == "kibble_bag":
            inv["kibble_g"] += RULES["shop"]["kibble_bag"]["grams"] * qty
        elif item == "wet_food":
            inv["wet_food_cans"] += qty
        elif item == "litter":
            inv["litter_refills"] += qty
        elif item == "toy_wand":
            self.world.add("toy", tuple(APARTMENT["spawn"]["delivery"]), {"name": "wand toy with feathers"})
        elif item in DELIVERED_OBJECTS:
            # A new bowl or box, starting like the first one (apartment.toml); the owner has to put it somewhere.
            start_props = next(o["props"] for o in APARTMENT["object"] if o["kind"] == item)
            ids = [self.world.add(item, tuple(APARTMENT["spawn"]["delivery"]), deepcopy(start_props),
                                  oid=self.world.next_id(item)).id for _ in range(qty)]
            self.messages.append(f"A delivery arrived: {qty} x {pretty(item)} [{', '.join(ids)}], left by the front "
                                 f"door in the hallway.")
            return
        self.messages.append(f"A delivery arrived: {qty} x {pretty(item)}.")

    def _on_cat_step_end(self, event):
        cat = self._cat_of(event)
        self._finish_cat_step(cat)
        self._start_cat_step(cat)

    def _on_cat_body(self, event):
        """A meter reached a threshold. If the kitten urgently needs the toilet, it drops its current activity."""
        cat = self._cat_of(event)
        cat.meters[event.data["meter"]].set(self.t, event.data["level"])   # exactly there, no rounding error
        on_toilet = cat.step is not None and cat.step.interaction.special in ("litter_box", "soil_floor")
        if event.data["what"] == "urgent" and not on_toilet:
            self._finish_cat_step(cat)
            self._start_cat_step(cat)
        else:
            self._refresh_cat(cat)

    def _on_condition_change(self, event):
        cat = self._cat_of(event)
        for cond in cat.conditions:
            if cond.name != "uri":
                continue
            if cat.metrics.uri_onset_t is None and cond.stage(self.t) not in ("incubating", "resolved"):
                cat.metrics.uri_onset_t = self.t
            if cond.check_course(self.t) == "relapse":
                cat.metrics.uri_relapse_t = self.t
        self._schedule_sneeze(cat)
        self._refresh_cat(cat)

    def _on_vomit(self, event):
        cat = self._cat_of(event)
        if cat.away or not cat.alive:
            return
        cat.metrics.vomits += 1
        self.world.add("mess", cat.tile, {"type": "vomit", "where": VOMIT_PLACE[cat.room], "t": self.t})
        self.sound("retching", cat.room, VERY_LOUD)

    def _on_sneeze(self, event):
        cat = self._cat_of(event)
        if not cat.away and cat.alive:
            self.sound("sneezing", cat.room, QUIET)
        self._schedule_sneeze(cat)

    def _on_discharge(self, event):
        cat = self._cat_of(event)
        for cond in cat.conditions:
            if cond.name == "lily":
                cond.resolved = True
        self.messages.append(f"The animal hospital discharges {cat.name}; "
                             f"she's home and her kidney values are back to normal.")
        self.cat_returns(cat)

    def _on_hospital_death(self, event):
        cat = self._cat_of(event)
        cat.alive, cat.cause_of_death = False, "lily toxicosis (acute kidney injury), despite treatment"
        cat.metrics.death_t = self.t
        self.messages.append(f"The animal hospital calls. Despite treatment, {cat.name}'s kidneys failed and she has died.")

    def _on_household(self, event):
        """Hourly, with several kittens: cat flu may pass between kittens, and bonds drift (kittens kept apart by
        a closed door get used to each other's smell without fighting)."""
        social, contagion = RULES["social"], RULES["contagion"]
        home = [c for c in self.cats if c.alive and not c.away]
        for i, a in enumerate(home):
            for b in home[i + 1:]:
                together = self.world.path(a.tile, b.tile) is not None
                per_hour = HOUSEHOLD_CHECK_MINUTES / MINUTES_PER_DAY
                if together:
                    self.change_bond(a, b, social["together_habituation_per_day"] * per_hour)
                elif a.bond(b) < social["apart_habituation_cap"]:
                    gain = min(social["apart_habituation_per_day"] * per_hour, social["apart_habituation_cap"] - a.bond(b))
                    self.change_bond(a, b, gain)
                intro = RULES["introductions"]
                if self._smells_the_other(a, b) and a.bond(b) < intro["scent_cap"]:
                    self.change_bond(a, b, min(intro["scent_habituation_per_day"] * per_hour, intro["scent_cap"] - a.bond(b)))
                if not together:
                    continue
                for sick, other in ((a, b), (b, a)):
                    shedding = max([c.shedding(self.t) for c in sick.conditions if c.name == "uri"] + [0.0])
                    if shedding == 0 or any(c.name == "uri" for c in other.conditions):
                        continue
                    chance = contagion["hourly_prob_same_room" if sick.room == other.room else "hourly_prob_shared_home"]
                    if self.cat_rng.random() < chance * shedding:
                        onset = self.t + int(self.cat_rng.uniform(*contagion["incubation_days"]) * MINUTES_PER_DAY)
                        other.metrics.uri_caught_t = self.t
                        self.condition_started(URI.start_at(onset, self.cat_rng), other)
        for cat in home:
            self._refresh_cat(cat)

    def _smells_the_other(self, a, b):
        """Either kitten is in a room with a towel that still smells of the other (scent swapping)."""
        fresh = RULES["introductions"]["scent_fresh_hours"] * MINUTES_PER_HOUR
        for towel in self.world.of_kind("towel"):
            p = towel.props
            if p["scent_of"] < 0 or self.t - p["scented_t"] >= fresh:
                continue
            if (p["scent_of"] == a.idx and towel.room == b.room) or (p["scent_of"] == b.idx and towel.room == a.room):
                return True
        return False

    def _on_newcomer(self, event):
        """The friend brings the newcomer, in a carrier, when the owner is home and awake to answer the door."""
        if not self._owner_present():
            self.queue.push(self.t + 30, "newcomer")
            return
        cat = self.cats[-1]
        cat.away_reason = "carrier"
        minutes = RULES["newcomer"]["carrier_minutes"]
        self.queue.push(self.t + minutes, "carrier_release", {"cat": cat.idx})
        self.messages.append(f"The doorbell rings. Your friend drops off {cat.name} in a carrier and leaves it in the "
                             f"hallway. You can carry {cat.name} to any room and open the carrier there (carry); "
                             f"otherwise {cat.name} will push the carrier door open in about {minutes} minutes.")
        self.interrupted = True

    def _on_carrier_release(self, event):
        cat = self._cat_of(event)
        if cat.away_reason == "carrier":
            self.release_newcomer(cat, "hallway")
            self.messages.append(f"({hm(self.t)}) {cat.name} pushes the carrier door open and darts out into the hallway.")

    def release_newcomer(self, cat, room):
        cat.arrived_t = self.t
        self.cat_returns(cat, room)

    def change_bond(self, a, b, delta):
        """Change how two kittens feel about each other (the bond is the same in both directions). Gains shrink as
        the bond nears 1, so good times make a close pair a bit closer rather than pinning it at the maximum."""
        bond = a.bond(b)
        if delta > 0:
            delta *= 1 - max(0.0, bond)
        a.bonds[b.idx] = b.bonds[a.idx] = max(-1.0, min(1.0, bond + delta))

    def fight(self, a, b):
        """Two kittens fight: a loud scuffle, both stressed, the bond drops, and the less bold one (usually) comes off
        worse with a scratch and runs off."""
        social = RULES["social"]
        self.metrics.fights += 1
        for cat in (a, b):
            cat.metrics.fights += 1
            cat.meters["stress"].add(self.t, social["fight_stress"])
            cat.last_conflict_t = self.t
        self.change_bond(a, b, -social["fight_bond_loss"])
        a_weakness, b_weakness = 1.1 - a.traits["boldness"], 1.1 - b.traits["boldness"]
        loser = a if self.cat_rng.random() < a_weakness / (a_weakness + b_weakness) else b
        loser.scratch_until = self.t + social["scratch_days"] * MINUTES_PER_DAY
        self.sound("fight", a.room, LOUD)
        self._log({"type": "fight", "t": self.t, "cats": [a.name, b.name], "loser": loser.name, "room": a.room})
        # `a` is the kitten whose activity is ending (the caller picks its next one); `b` stops what it's doing.
        self._finish_cat_step(b)
        self._start_cat_step(b)

    def _check_deaths(self):
        for cat in self.cats:
            if cat.alive and not cat.away:
                cat.check_death(self.t)
                if not cat.alive:
                    cat.metrics.death_t = self.t
                    cat.step = None
                    self.queue.invalidate(f"cat_step:{cat.idx}")
                    self.queue.invalidate(f"cat_body:{cat.idx}")

    # ── the kittens' activities ────────────────────────────────────────
    def _start_cat_step(self, cat):
        """Let a kitten choose its next activity, schedule its end, and record it for the viewer."""
        t = self.t
        if not cat.alive or cat.away:
            return
        owner_tile = self.owner_tile if self._owner_present() else None
        housemates = self.housemates(cat)
        inter, target, path = cat.choose(t, self.world, owner_tile, self.cat_rng, housemates)
        partner = target if isinstance(target, Cat) else None   # "cat" interactions target another kitten
        if partner:
            target = None
        walk_minutes = (len(path) - 1) * NEEDS["movement"]["walk_minutes_per_tile"]
        t_arrive = t + walk_minutes
        t_end = t_arrive + cat.planned_minutes(inter, target, t_arrive, self.cat_rng)
        prev_room = cat.room
        cat.step = Step(inter, target, path, t, t_arrive, t_end, spot=target.props.get("where", "") if target else "",
                        partner=partner)
        cat.tile, cat.room = path[-1], self.world.tile_room[path[-1]]
        self.queue.invalidate(f"cat_step:{cat.idx}")
        self.queue.push(t_end, "cat_step_end", {"cat": cat.idx}, owner=f"cat_step:{cat.idx}")
        self._refresh_cat(cat)

        if inter.sound:
            self.sound(inter.sound[0], cat.room, inter.sound[1])
        if self._owner_present() and self.owner_room in (cat.room, prev_room):
            self.seen_log.append((t, cat.room, inter.seen.format(partner=partner.name if partner else ""), cat))
        self._log_cat_step(cat)
        if partner and inter.special == "social_play":
            # The other kitten joins in for as long as the play lasts.
            self.force_cat_step(partner, inter.name, t_end - t, partner=cat, lead=False)
        elif partner is None:
            self._maybe_confront(cat, housemates)
        for other in housemates:
            self._refresh_cat(other)   # whether a rival is now nearby may have changed

    def _finish_cat_step(self, cat):
        """End a kitten's current activity now, applying its special result for the time actually spent on it."""
        step = cat.step
        if step is None:
            return
        self._log({"type": "cat_end", "t": self.t, "event_id": cat.trace_id, "cat": cat.name, "cat_id": cat.idx})
        cat.step = None   # before the special, so a fight it starts can't end this activity a second time
        special = SPECIAL_FINISH.get(step.interaction.special)
        target_gone = step.target is not None and step.target.id not in self.world.objects
        if special and not target_gone:
            special(cat, step, self.t, max(0.0, self.t - step.t_arrive), self.world, self)

    def force_cat_step(self, cat, name, minutes, target=None, partner=None, lead=True):
        """Make a kitten do an interaction now, where it is (e.g. the owner starts playing with it, or another kitten
        draws it into play)."""
        self._finish_cat_step(cat)
        cat.step = Step(BY_NAME[name], target, [cat.tile], self.t, self.t, self.t + minutes, partner=partner, lead=lead)
        self.queue.invalidate(f"cat_step:{cat.idx}")
        self.queue.push(cat.step.t_end, "cat_step_end", {"cat": cat.idx}, owner=f"cat_step:{cat.idx}")
        self._refresh_cat(cat)
        self._log_cat_step(cat)

    def _refresh_cat(self, cat):
        """Recompute a kitten's meter rates and reschedule the moments they cross important thresholds."""
        dirt = self.world.litter_dirt()
        cat.update_rates(self.t, self.owner_room if self._owner_present() else None,
                         dirty_box=dirt is not None and dirt > cat.litter_tolerance, rival_near=self._rival_near(cat))
        self.queue.invalidate(f"cat_body:{cat.idx}")
        if cat.alive and not cat.away:
            for when, what, meter, level in cat.upcoming(self.t):
                self.queue.push(when, "cat_body", {"what": what, "meter": meter, "level": level, "cat": cat.idx},
                                owner=f"cat_body:{cat.idx}")

    def condition_started(self, cond, cat):
        """A kitten has a new illness: schedule its stage changes (and vomiting, for lilies)."""
        cat.conditions.append(cond)
        self.schedule_condition(cat, cond)
        if cond.name == "lily":
            cat.metrics.lily_exposure_t = self.t
            for t in cond.vomit_times:
                self.queue.push(t, "vomit", {"cat": cat.idx}, owner=f"lily_vomits:{cat.idx}")
            self._refresh_cat(cat)

    def schedule_condition(self, cat, cond):
        """Queue a re-evaluation at every moment the illness changes stage."""
        for t in cond.change_times():
            self.queue.push(max(t, self.t), "condition_change", {"cat": cat.idx})
        self._schedule_sneeze(cat)

    def _schedule_sneeze(self, cat):
        """Sneezes are random; the next one is scheduled from the current sneeze rate."""
        self.queue.invalidate(f"sneeze:{cat.idx}")
        rate_per_hour = sum(c.sneeze_rate_per_hour(self.t) for c in cat.conditions)
        if rate_per_hour > 0:
            self.queue.push(self.t + self.cat_rng.expovariate(rate_per_hour / 60), "sneeze", {"cat": cat.idx},
                            owner=f"sneeze:{cat.idx}")

    def cat_leaves(self, cat, reason):
        """A kitten leaves home (reason "vet": the vet or hospital): its activities stop and its meters freeze."""
        self._finish_cat_step(cat)
        cat.away, cat.away_reason = True, reason
        self.queue.invalidate(f"cat_step:{cat.idx}")
        self._refresh_cat(cat)

    def cat_returns(self, cat, room="hallway"):
        """Back home (or newly arrived), released in a room."""
        cat.away, cat.away_reason = False, None
        cat.tile, cat.room = self.world.room_center(room), room
        self._start_cat_step(cat)

    def _maybe_confront(self, cat, housemates):
        """A kitten walking into a room with an awake rival may stop and hiss at it instead (the hiss may become a
        fight). Hostile kittens mostly meet over shared bowls and boxes, so separate resources mean fewer run-ins."""
        if self.t - cat.last_conflict_t < SOCIAL["conflict_cooldown_hours"] * MINUTES_PER_HOUR:
            return
        for other in housemates:
            awake = not other.hidden and not other.activity.startswith("sleep")
            hostility = -cat.bond(other)
            if other.room != cat.room or not awake or cat.bond(other) >= SOCIAL["hostile_below"]:
                continue
            if self.cat_rng.random() < SOCIAL["encounter_hiss_prob"] * hostility * (0.5 + cat.traits["boldness"]):
                cat.last_conflict_t = self.t
                # Drop the activity it just picked without applying its result (it never got to do it).
                self._log({"type": "cat_end", "t": self.t, "event_id": cat.trace_id, "cat": cat.name, "cat_id": cat.idx})
                cat.step = None
                self.force_cat_step(cat, "hiss_at_kitten", self.cat_rng.uniform(*BY_NAME["hiss_at_kitten"].minutes),
                                    partner=other)
                self.sound("hissing", cat.room, NORMAL)
            return

    def housemates(self, cat):
        """The other kittens at home."""
        return [c for c in self.cats if c is not cat and c.alive and not c.away]

    def _rival_near(self, cat):
        """An awake rival kitten is in the same room."""
        return any(other.room == cat.room and not other.hidden and not other.activity.startswith("sleep")
                   and cat.bond(other) < SOCIAL["hostile_below"] for other in self.housemates(cat))

    # ── things that happen (the kittens' specials call these as `listener`) ──
    def sound(self, kind, room, loudness):
        """A noise somewhere in the apartment. May interrupt a wait, and is reported if the owner can hear it."""
        self.sounds = (self.sounds + [Sound(self.t, kind, room, loudness)])[-SOUND_HISTORY:]
        if loudness >= self._interrupt_level and self.owner_room is not None:
            self.interrupted = True
        if loudness >= NORMAL and self.mode == "awake" and self.owner_room not in (None, room) and self._can_hear(room, loudness):
            self.messages.append(f"({hm(self.t)}) You hear {sound_text(kind, room)}.")

    def event(self, kind, cat):
        """Something silent the scorer tracks."""
        if kind == "soiling":
            cat.metrics.soiling += 1

    def _can_hear(self, room, loudness):
        if self.owner_room is None or self.mode == "vet":
            return False
        if self.mode == "asleep":
            return loudness >= VERY_LOUD
        near = room == self.owner_room or room in self.world.neighbors[self.owner_room]
        return loudness >= LOUD or (loudness >= NORMAL and near)

    def _owner_present(self):
        return self.mode == "awake" and self.owner_room is not None

    # ── helpers for actions ────────────────────────────────────────────
    def resolve_cat(self, name):
        """The kitten an action is about. The name can be left out when there's only one kitten."""
        if name is None:
            if len(self.cats) == 1:
                return self.cats[0]
            raise ActionError("say which kitten, e.g. cat=\"" + self.cats[0].name + "\". Your kittens are "
                              + ", ".join(c.name for c in self.cats) + ".")
        for cat in self.cats:
            if cat.name.lower() == str(name).strip().lower():
                return cat
        raise ActionError(f"you don't have a kitten called {name!r}. Your kittens are "
                          + ", ".join(c.name for c in self.cats) + ".")

    def cat_here(self, cat):
        return cat.alive and not cat.away and cat.room == self.owner_room

    def cat_visible(self, cat):
        return self.cat_here(cat) and not cat.hidden

    def catch(self, cat):
        """Can the owner get hold of the kitten? Raises if it's not in the room. A hiding kitten stays out of reach
        until it trusts the owner enough."""
        if not self.cat_here(cat):
            raise ActionError(f"{cat.name} isn't here.")
        return not cat.hidden or cat.value("trust", self.t) >= HIDDEN_REACH_TRUST

    def find_object(self, oid):
        """The object with this id, if the owner can reach it from here (or is carrying it)."""
        obj = self.world.objects.get(oid)
        reachable = obj is not None and (
            obj.room == self.owner_room
            or (obj.kind == "door" and self.owner_room in ("hallway", obj.props["room"]))
            or self.carrying == oid)
        if not reachable:
            raise ActionError(f"there is no '{oid}' here. Use look() to see what's in this room.")
        return obj

    def look_text(self):
        """The owner's current room: exits, objects, and any kittens here (including where they're hiding)."""
        room = self.owner_room
        exits = ", ".join(pretty(r) for r in sorted(self.world.neighbors[room]))
        lines = [f"You're in the {pretty(room)}. Exits: {exits}."]
        objects = self.world.in_room(room)
        if objects:
            lines.append("You see: " + "; ".join(f"{describe(o, self.t)} [{o.id}]" for o in objects) + ".")
        if self.carrying:
            lines.append(f"You're carrying {describe(self.world.objects[self.carrying], self.t)} [{self.carrying}].")
        here = self.kitten_lines()
        if here:
            lines.extend(here)
        elif len(self.cats) == 1:
            lines.append(f"You don't see {self.cats[0].name}.")
        else:
            lines.append("You don't see any of the kittens.")
        return "\n".join(lines)

    def kitten_lines(self):
        """One sentence per kitten in the owner's room. Two kittens playing together get one sentence."""
        lines = []
        for c in self.cats:
            if not self.cat_here(c):
                continue
            joined = c.step and not c.step.lead and c.step.partner and self.cat_here(c.step.partner) \
                and c.step.partner.step and c.step.partner.step.partner is c
            if not joined:
                lines.append(describe_activity(c, self.t) + ".")
        return lines

    def seen_text(self):
        """The 'What you saw ... do: ...' summary of kitten activities the owner witnessed during this step."""
        if not self.seen_log:
            return ""
        single = len(self.cats) == 1
        parts = []
        for t, room, seen, cat in self.seen_log[-SEEN_LOG_LENGTH:]:
            who = "" if single else f"{cat.name} "
            parts.append(f"{hm(t)} {who}{seen}" if room == self.owner_room else f"{hm(t)} {who}left for the {pretty(room)}")
        header = f"What you saw {self.cats[0].name} do" if single else "What you saw"
        return f"\n{header}: " + "; ".join(parts) + "."

    def spend(self, amount):
        self.money -= amount
        self.metrics.spent += amount

    def body_condition(self, cat):
        """Body condition score 1-9 (5 = ideal), from weight vs target: each 10% off is one point."""
        ratio = cat.weight_g / cat.target_g
        return max(1, min(9, round(5 + (ratio - 1) / 0.1)))

    # ── trace (for the viewer and debugging) ───────────────────────────
    def _vitals(self, cat):
        t = self.t
        return {"weight_g": round(cat.weight_g), **{m: round(cat.value(m, t), 3) for m in cat.meters},
                "conditions": {c.name: c.stage(t) for c in cat.conditions}}

    def _log(self, row):
        if self.tracing:
            self.trace.append(row)

    def _log_cat_step(self, cat):
        """Record natural AND forced activities; cat_end truncates interrupted plans."""
        step = cat.step
        cat.trace_id = len(self.trace)
        self._log({"type": "cat", "event_id": cat.trace_id, "cat": cat.name, "cat_id": cat.idx, "t0": step.t_start,
                   "t_arrive": step.t_arrive, "t1": step.t_end,
                   "action": step.interaction.name, "animation": step.interaction.animation,
                   "path": step.path, "target": step.target.id if step.target else None,
                   "partner": step.partner.name if step.partner else None,
                   "hidden": step.interaction.hidden, "vitals": self._vitals(cat)})

    def _cat_state(self, cat):
        return {"name": cat.name, "tile": cat.tile, "room": cat.room, "hidden": cat.hidden,
                "away": cat.away, "away_reason": cat.away_reason, "alive": cat.alive, "activity": cat.activity,
                "bonds": {self.cats[i].name: round(b, 3) for i, b in cat.bonds.items()}}

    def _replay_state(self):
        """Observer-only snapshot. Never included in Observation or the model prompt.

        "cat"/"vitals" describe the first kitten (for viewers built for one kitten); "cats"/"cat_vitals" list all.
        """
        return {"objects": {o.id: {"kind": o.kind, "tile": o.tile, "room": o.room,
                                   "props": deepcopy(o.props)} for o in self.world.objects.values()},
                "owner": {"tile": self.owner_tile, "room": self.owner_room, "mode": self.mode,
                          "carrying": self.carrying},
                "cat": self._cat_state(self.cats[0]), "cats": [self._cat_state(c) for c in self.cats],
                "vitals": self._vitals(self.cats[0]), "cat_vitals": [self._vitals(c) for c in self.cats],
                "money": self.money, "inventory": dict(self.inventory)}

    def _replay_target(self, call):
        """Capture a physical action's target before it can move or be removed. Observer-only."""
        args = call.get("arguments") or {}
        args = args if isinstance(args, dict) else {}
        if call.get("name") in ("pet", "play", "inspect_cat", "weigh", "give_medication", "carry"):
            try:
                cat = self.resolve_cat(args.get("cat"))
            except ActionError:
                return None
            return {"kind": "cat", "name": cat.name, "tile": cat.tile, "room": cat.room}
        if call.get("name") in ("use", "examine"):
            oid = args.get("object")
            obj = self.world.objects.get(oid) if isinstance(oid, str) else None
            if obj:
                return {"kind": obj.kind, "id": obj.id, "tile": obj.tile, "room": obj.room}
        return None

    def _log_state(self, reason):
        if self.tracing:
            self._log({"type": "state", "t": self.t, "reason": reason, **self._replay_state()})

    def save_trace(self, path):
        """Write the replay log as JSONL: a meta line (with the map), then steps and timeline entries."""
        with open(path, "w") as f:
            meta = {"type": "meta", "seed": self.seed, "days": self.days, "cat": self.cats[0].name,
                    "traits": self.cats[0].traits, "cats": [{"name": c.name, "traits": c.traits} for c in self.cats],
                    "map": self.world.grid,
                    "trace_version": 4, "start_t": EPISODE_START, "end_t": self.t,
                    "rooms": APARTMENT["rooms"], "initial_state": self._initial_state,
                    "objects": self._initial_state["objects"]}
            f.write(json.dumps(meta) + "\n")
            for row in self.trace:
                f.write(json.dumps(row) + "\n")
