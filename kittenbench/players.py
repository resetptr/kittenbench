"""Who can play: the tool list shown to an LLM agent, and the scripted reference players.

Sections:
    tools            the tool list given to an LLM (names, descriptions, JSON Schema arguments)
    scripted players reference policies used to sanity-check the simulator
"""

import random
import re

from .sim import energy_need_kcal


# ════ tools ═══════════════════════════════════════════════════════════════════════════════════════
# The tool list shown to an LLM agent (names, descriptions, JSON Schema arguments).
#
# Each tool here must match a `tool_<name>` method in env.py (a test checks this).
#
# Design rule: tools are generic verbs ("use", "examine", "go_to"). Nothing names a hazard, an illness or the
# right response to one (there is no remove_lilies()), because a tool list like that would hand the agent the
# answers.

def _tool(name, description, props=None, required=()):
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": props or {}, "required": list(required)}}


_STR = {"type": "string"}
_INT = {"type": "integer"}
_CAT = {"cat": {"type": "string", "description": "which kitten (its name); only needed if you have more than one"}}

TOOLS = [
    # Perceive.
    _tool("look", "Look around the room you're in (1 min)."),
    _tool("listen", "Listen for sounds from the rest of the apartment (1 min)."),
    _tool("go_to", "Walk to a room (1 min). You'll see what's there.", {"room": _STR}, ["room"]),
    _tool("find_cat", "Search the apartment for a kitten (1 min per room searched). With several kittens and no "
                      "name, searches until you've seen them all.", _CAT),
    _tool("examine", "Examine an object in this room closely (1 min). Shows what you can do with it.",
          {"object": {**_STR, "description": "object id shown in [brackets]"}}, ["object"]),
    _tool("inspect_cat", "Check part of the kitten's body (2 min). The kitten must be in this room.",
          {"region": {"type": "string", "enum": ["eyes", "nose", "mouth", "skin", "coat", "belly", "rear", "overall"]},
           **_CAT}, ["region"]),
    _tool("weigh", "Weigh the kitten (2 min). The kitten must be in this room.", _CAT),
    # Act on objects and the kitten.
    _tool("use", "Do something with an object in this room, e.g. use(verb='fill', object='food_bowl', item='kibble', "
                 "amount=30). Use examine() to see which verbs an object supports.",
          {"verb": _STR, "object": _STR, "item": {**_STR, "description": "when filling the food bowl: kibble or wet_food (leave out for the water bowl)"},
           "amount": {"type": "number", "description": "when filling the food bowl: grams of kibble, or cans of wet food"},
           "cat": {**_STR, "description": "for verbs done to a kitten (e.g. rub): which kitten"}}, ["verb", "object"]),
    _tool("give_medication", "Give the kitten a dose of the medication you have (3 min). The kitten must be in this room.",
          _CAT),
    _tool("play", "Play with the kitten (1-30 min). Give with_cat to play with two kittens together; both must be in "
                  "this room.", {"minutes": _INT, **_CAT, "with_cat": {**_STR, "description": "a second kitten to play with at "
                                                                                       "the same time"}}),
    _tool("pet", "Pet the kitten (2 min).", _CAT),
    _tool("carry", "Pick a kitten up and carry it to a room (2 min). Closed doors keep kittens in (or out); "
                   "you can always pass through them.",
          {"cat": {**_STR, "description": "which kitten (its name)"}, "room": _STR}, ["cat", "room"]),
    # The outside world.
    _tool("shop", "Order supplies online (3 min). Arrives tomorrow at 18:00. Items: kibble_bag (1.5 kg, $25), "
                  "wet_food (85 g can, $2), litter (one full refill, $15), toy_wand ($10), litter_box (an extra "
                  "box with litter, $30), food_bowl ($8), water_bowl ($8).",
          {"item": _STR, "quantity": _INT}, ["item"]),
    _tool("book_vet", "Take the kitten to the vet. 'emergency' goes to the emergency clinic right now (~3 h, $250 + "
                      "treatment); 'appointment' books the next 08:00 slot at your regular vet ($80 + treatment).",
          {"urgency": {"type": "string", "enum": ["emergency", "appointment"]}, "reason": _STR, **_CAT}, ["urgency"]),
    # Time.
    _tool("wait", "Stay where you are and let time pass (1-600 min). Stops early if something happens. Tells you "
                  "what the kittens did in this room meanwhile.", {"minutes": _INT}, ["minutes"]),
    _tool("wait_until", "Let time pass until a clock time, e.g. '18:00'. Stops early if something happens.",
          {"time": _STR}, ["time"]),
    _tool("sleep", "Go to bed for the night (only after 19:00). You wake at 07:00 or if something loud happens."),
    # Memory (no game time).
    _tool("write_note", "Write a note to your future self (takes no time).", {"text": _STR}, ["text"]),
    _tool("read_notes", "Read your notes (takes no time)."),
    _tool("check_inventory", "Check your supplies and money (takes no time)."),
]


def openai_tools():
    """The same tools in the OpenAI / OpenRouter function-calling format."""
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in TOOLS]


# ════ scripted players ════════════════════════════════════════════════════════════════════════════
# Scripted reference players. They are sanity checks for the simulator, not baselines we expect LLMs to copy.
#
# If the simulator is sensible, scores should come out roughly:
#     Oracle >= Guideline > Minimal > Neglect, Random
# and Helicopter should survive but spend far more and make many unnecessary vet visits.
#
# All policies except Oracle see only the observation text, exactly what an LLM agent sees.
#
# How the scripts are written: each policy's `main()` is a Python generator that yields one tool call at a time.
# After each call, the environment's response is stored in `self.obs` before the generator resumes, so
# `obs = yield from self.do("look")` reads like a normal function call that returns the observation.

HEADER = re.compile(r"\[(\w+) (\d\d):(\d\d) · day (\d+) · ([\w ]+)\]")  # the [Saturday 10:00 · day 0 · hallway] line
WEEKEND = ("Saturday", "Sunday")
ALARMS = ("retching", "vomit", "crash")


def call(name, **args):
    return {"name": name, "arguments": args}


class ScriptPolicy:
    def __init__(self, env=None, seed=0):
        self.env, self.rng = env, random.Random(seed)
        self.obs = ""
        self.gen = self.main()

    def act(self, obs_text):
        """Given the latest observation, return the next tool call."""
        self.obs = obs_text
        return next(self.gen)

    def do(self, name, **args):
        """Yield one tool call and return the observation it produced."""
        yield call(name, **args)
        return self.obs

    def now(self):
        """(day name, minute of day, day number, room) parsed from the observation header."""
        m = HEADER.search(self.obs)
        return m.group(1), int(m.group(2)) * 60 + int(m.group(3)), int(m.group(4)), m.group(5)

    def main(self):
        raise NotImplementedError


class Neglect(ScriptPolicy):
    """Never does anything for the kitten: waits all day and sleeps all night."""

    def main(self):
        while True:
            _, mins, _, _ = self.now()
            night = mins >= 19 * 60 or mins < 7 * 60
            yield call("sleep") if night else call("wait", minutes=600)


class Minimal(ScriptPolicy):
    """Puts out 60 g of kibble and fresh water once a day. Never cleans, looks around, or goes to the vet."""

    def main(self):
        fed_day = -1
        while True:
            _, mins, day, _ = self.now()
            if fed_day != day:
                fed_day = day
                yield from self.do("go_to", room="kitchen")
                obs = yield from self.do("use", verb="fill", object="food_bowl", item="kibble", amount=60)
                running_low = re.search(r" ([0-9]|[1-9][0-9]) g left in the bag", obs)  # under 100 g
                if "out of kibble" in obs or running_low:
                    yield from self.do("shop", item="kibble_bag")
                yield from self.do("use", verb="fill", object="water_bowl")
            elif mins >= 19 * 60 or mins < 7 * 60:
                yield call("sleep")
            else:
                yield call("wait", minutes=600)


class RandomPolicy(ScriptPolicy):
    """Random actions with random arguments."""

    ROOMS = ["hallway", "kitchen", "living_room", "bedroom", "bathroom"]
    OBJECTS = ["food_bowl", "water_bowl", "litter_box", "vase_1", "bedroom_door", "bathroom_door", "toy_mouse"]

    def main(self):
        r = self.rng
        while True:
            yield r.choice([
                call("look"), call("listen"), call("find_cat"), call("go_to", room=r.choice(self.ROOMS)),
                call("wait", minutes=r.randint(5, 240)), call("play", minutes=10), call("pet"),
                call("use", verb=r.choice(["fill", "scoop", "open", "close", "throw_away", "empty"]),
                     object=r.choice(self.OBJECTS), item="kibble", amount=r.randint(5, 80)),
                call("inspect_cat", region=r.choice(["eyes", "overall", "skin"])), call("sleep"),
            ])


class Guideline(ScriptPolicy):
    """A conscientious owner following vet guidelines, from observations only.

    Does a full check at fixed times (07:05 and 18:05, plus 12:00 on weekends, and a last one at 22:00):
    feed, water, scoop, walk the rooms, throw out toxic plants, clean messes, check each kitten, give meds,
    weigh each kitten every 3 days, play, restock. Goes to the vet on warning signs. Waits between checks, reacting
    to alarming sounds. With several kittens it names the kitten in every kitten action; with one it doesn't.

    With several kittens it also follows the multi-cat guidelines (rules.toml [resources], [newcomer]): one litter
    box per kitten plus one, spread over different rooms, and a second food and water bowl in the bedroom. The
    bedroom is its isolation room: a kitten on antibiotics stays there behind the closed door until the course is
    done, a newcomer stays there for its first 7 days, and a kitten found with a scratch from a fight stays there for
    5 days before being let back out. While the newcomer is apart it swaps scents with towels at every check; for 5
    days after it's let out it plays short supervised sessions with the newcomer and whichever residents are nearby.
    """

    TOXIC = ("lily", "lilies")
    ISOLATION_ROOM = "bedroom"
    INTRO_DAYS = 7          # AAFP: keep a newcomer in its own room for at least 7 days
    TIME_OUT_DAYS = 5       # a kitten scratched in a fight is kept apart this long, then reintroduced
    SESSION_DAYS = 5        # supervised play with the newcomer for this many days after its time apart
    SESSION_MINUTES = 5     # AAFP: keep sessions short (5-20 min)
    PLACE_ROOMS = {"litter_box": ["bedroom", "living_room", "hallway"], "food_bowl": ["bedroom", "living_room"],
                   "water_bowl": ["bedroom", "living_room"]}
    PORTION_SLACK = 1.3     # feed 30% more than the kittens' calculated need (kittens can free-feed)
    KIBBLE_KCAL_PER_G = 4.0

    def __init__(self, env=None, seed=0):
        super().__init__(env, seed)
        self.names = [c.name for c in env.cats] if env else ["Miso"]   # the owner knows the kittens' names
        self.weight_g = {n: 950.0 for n in self.names}   # latest weighings (a typical 8-week weight until then)
        self.last_weigh_day = -9
        self.lilies_seen = False    # lilies have been in the home, so vomiting/kidney signs are an emergency
        self.appointment = set()    # kittens with a vet appointment booked
        self.on_meds = set()        # kittens being given antibiotics
        self.treated_day = {}       # kitten -> day the vet last prescribed antibiotics
        self.low_appetite = 0       # consecutive checks where the bowl was barely touched (shared bowl)
        self.last_level_g, self.last_check_t = 0, None   # bowl level and time at the last check
        self.no_urine_checks = 0    # consecutive checks with no urine in the box (kidney failure sign)
        self.ordered = set()        # supplies ordered and not yet delivered
        self.checks = 0
        # Several kittens.
        self.multi = len(self.names) > 1
        self.newcomer = env.cats[-1].name if env and env.newcomer_day is not None else None
        self.rooms_of = {"litter_box": {"litter_box": "bathroom"}, "food_bowl": {"food_bowl": "kitchen"},
                         "water_bowl": {"water_bowl": "kitchen"}}   # kind -> {object id: room}
        self.to_place = []          # delivered bowls and boxes waiting by the front door
        self.extras_ordered = False
        self.apart_until = {}       # kitten -> day its time in the isolation room ends (newcomer, after a fight)
        self.isolated_for_meds = set()   # kittens kept apart while on antibiotics
        self.confined = set()       # kittens currently shut in the isolation room
        self.door_closed = False
        self.newcomer_waiting = False
        self.arrived = set()        # the newcomer, once it's out of its carrier
        self.towel_room = {"bath_towel": "bathroom", "hand_towel": "bathroom"}

    def act(self, obs_text):
        self.read(obs_text)
        return next(self.gen)

    def read(self, obs_text):
        """Take in a new observation once: news (vet results, deliveries) can arrive in any of them."""
        if obs_text is not self.obs:
            self.obs = obs_text
            self.note_events()

    def cat_args(self, name):
        """The `cat` argument for an action on this kitten: only needed with more than one kitten."""
        return {} if len(self.names) == 1 else {"cat": name}

    def portion_g(self, checks_per_day):
        kcal = sum(energy_need_kcal(w, 60) for w in self.weight_g.values()) * self.PORTION_SLACK
        return round(kcal / self.KIBBLE_KCAL_PER_G / checks_per_day)

    def main(self):
        while True:
            yield from self.check()
            yield from self.until_next_check()

    # ── one full check ─────────────────────────────────────────────────
    def check(self):
        self.checks += 1
        dayname, _, day, _ = self.now()
        if self.newcomer_waiting:
            yield from self.welcome_newcomer()
        if self.multi:
            yield from self.multi_cat_setup(day)
            if self.newcomer in self.confined and self.newcomer in self.apart_until:
                yield from self.scent_swap()
            elif self.newcomer in self.arrived and day < self.apart_until.get(self.newcomer, -99) + self.SESSION_DAYS:
                yield from self.supervised_play()
        yield from self.feed(checks_per_day=3 if dayname in WEEKEND else 2)
        big_clumps = yield from self.clean_litter()
        if (yield from self.sweep_rooms()):
            return  # went to the emergency vet
        if (yield from self.check_kittens(day, big_clumps)):
            return
        yield from self.medicate()
        yield from self.maybe_weigh(day)
        yield from self.play_with_kittens()
        yield from self.restock()

    def feed(self, checks_per_day):
        """Top each bowl up to its share of a portion, track whether the kittens are eating, add warmed wet food if
        they're off their food, and refill the water."""
        total_left, total_level, now = 0, 0, None
        for bowl, room in self.rooms_of["food_bowl"].items():
            portion = self.bowl_portion(bowl, self.portion_g(checks_per_day))
            yield from self.do("go_to", room=room)
            obs = yield from self.do("examine", object=bowl)
            if now is None:
                _, now_min, day, _ = self.now()
                now = day * 1440 + now_min
            left = re.search(r"about (\d+) g of kibble", obs)
            left_g = int(left.group(1)) if left else 0
            total_left += left_g
            if "smells off" in obs:   # spoiled wet food: clear the bowl before adding anything
                yield from self.do("use", verb="empty", object=bowl)
                left_g = 0
            if left_g < portion:
                yield from self.do("use", verb="fill", object=bowl, item="kibble", amount=round(portion - left_g))
            total_level += max(portion, left_g)
        self.track_appetite(total_left, now)
        self.last_level_g, self.last_check_t = total_level, now
        if self.on_meds or self.low_appetite:
            obs = yield from self.do("check_inventory")
            if re.search(r"Wet food: [1-9]", obs):
                yield from self.warm_wet_food()
            elif "wet_food" not in self.ordered:
                self.ordered.add("wet_food")
                yield from self.do("shop", item="wet_food", quantity=14)
        for bowl, room in self.rooms_of["water_bowl"].items():
            yield from self.go(room)
            yield from self.do("use", verb="fill", object=bowl)

    def go(self, room):
        """Walk to a room unless already there."""
        if self.now()[3] != room.replace("_", " "):
            yield from self.do("go_to", room=room)

    def warm_wet_food(self):
        """A can of warmed wet food, in the isolation room's bowl if a kitten is shut in there (the microwave is in the
        kitchen, so that bowl is carried over and back)."""
        bowl, room = "food_bowl", "kitchen"
        if self.confined:
            bowl = next(b for b, r in self.rooms_of["food_bowl"].items() if r == self.ISOLATION_ROOM)
            room = self.ISOLATION_ROOM
            yield from self.do("go_to", room=room)
            yield from self.do("use", verb="take", object=bowl)
        yield from self.go("kitchen")
        yield from self.do("use", verb="fill", object=bowl, item="wet_food", amount=1)
        yield from self.do("use", verb="warm", object=bowl)
        if room != "kitchen":
            yield from self.do("go_to", room=room)
            yield from self.do("use", verb="put", object=bowl)

    def track_appetite(self, left_g, now):
        """The kittens are off their food if, over at least 4 hours, they ate under 40% of what their daily need
        implies."""
        if self.last_check_t is None or now - self.last_check_t < 4 * 60:
            return
        eaten = self.last_level_g - left_g
        expected = self.portion_g(1) / self.PORTION_SLACK * (now - self.last_check_t) / 1440
        self.low_appetite = self.low_appetite + 1 if eaten < 0.4 * expected else 0

    def clean_litter(self):
        """Scoop each box (or replace saturated litter). Returns whether there were unusually large urine clumps."""
        big_clumps, urine_total = False, 0
        for box, room in self.rooms_of["litter_box"].items():
            yield from self.do("go_to", room=room)
            obs = yield from self.do("examine", object=box)
            urine = re.search(r"(\d+) urine clump", obs)
            urine_total += int(urine.group(1)) if urine else 0
            big_clumps = big_clumps or "unusually large" in obs
            if "saturated" in obs:
                yield from self.do("use", verb="replace_litter", object=box)
            else:
                yield from self.do("use", verb="scoop", object=box)
        self.no_urine_checks = self.no_urine_checks + 1 if urine_total == 0 else 0
        return big_clumps

    # ── several kittens ────────────────────────────────────────────────
    def bowl_portion(self, bowl, total):
        """This bowl's share of the food: an even split, unless the isolation room is shut, in which case its bowls
        get the shut-in kittens' share and the other bowls the rest."""
        bowls = self.rooms_of["food_bowl"]
        if not self.door_closed:
            return total / len(bowls)
        inside = [b for b, r in bowls.items() if r == self.ISOLATION_ROOM]
        outside = [b for b in bowls if b not in inside]
        share_inside = len(self.confined) / max(1, len(self.present_names()))
        if bowl in inside:
            return total * share_inside / len(inside)
        return total * (1 - share_inside) / len(outside)

    def wanted_apart(self, day):
        """The kitten that should be in the isolation room right now (at most one: two kittens shut in one room with
        one box and one bowl may not get along). A kitten kept apart from a rival (newcomer, after a fight) comes
        first; otherwise the first kitten on antibiotics."""
        candidates = ([n for n in self.present_names() if day < self.apart_until.get(n, -1)]
                      or [n for n in self.present_names() if n in self.isolated_for_meds and n in self.on_meds])
        keep = [n for n in candidates if n in self.confined] or candidates
        return set(keep[:1])

    def multi_cat_setup(self, day):
        """Order the extra bowls and boxes, put deliveries in place, and keep the isolation room up to date."""
        if not self.extras_ordered:
            self.extras_ordered = True
            yield from self.do("shop", item="litter_box", quantity=len(self.names))
            yield from self.do("shop", item="food_bowl")
            yield from self.do("shop", item="water_bowl")
        while self.to_place:
            kind, oid = self.to_place.pop(0)
            rooms = self.PLACE_ROOMS[kind]
            room = rooms[(len(self.rooms_of[kind]) - 1) % len(rooms)]
            yield from self.do("go_to", room="hallway")
            yield from self.do("use", verb="take", object=oid)
            yield from self.do("go_to", room=room)
            yield from self.do("use", verb="put", object=oid)
            self.rooms_of[kind][oid] = room
        wanted = self.wanted_apart(day)
        if not wanted:
            self.confined.clear()
            if self.door_closed:
                yield from self.do("go_to", room="hallway")
                yield from self.do("use", verb="open", object="bedroom_door")
                self.door_closed = False
        elif wanted != self.confined:
            self.confined &= wanted
            yield from self.isolate(sorted(wanted - self.confined, key=self.names.index))

    def isolate(self, names, from_carrier=False):
        """Carry these kittens into the isolation room, carry everyone else out, and close the door."""
        room = self.ISOLATION_ROOM
        for name in names:
            if not from_carrier:
                yield from self.do("find_cat", cat=name)
            yield from self.do("carry", cat=name, room=room)
            yield from self.do("use", verb="close", object="bedroom_door")
            self.confined.add(name)
            self.door_closed = True
        for name in self.present_names():   # anyone else shut in by mistake goes out (the owner passes closed doors)
            if name in self.confined:
                continue
            yield from self.do("find_cat", cat=name)
            if self.now()[3] == room:
                yield from self.do("carry", cat=name, room="living_room")
        if not self.door_closed:
            yield from self.do("go_to", room="hallway")
            yield from self.do("use", verb="close", object="bedroom_door")
            self.door_closed = True

    def scent_swap(self):
        """AAFP steps 6-7: one towel carries the newcomer's smell to the living room, the other carries a resident's
        smell (a different resident each check) into the newcomer's room."""
        residents = [n for n in self.present_names() if n != self.newcomer]
        resident = residents[self.checks % len(residents)]
        for towel, kitten, room in (("hand_towel", self.newcomer, "living_room"),
                                    ("bath_towel", resident, self.ISOLATION_ROOM)):
            yield from self.go(self.towel_room[towel])
            yield from self.do("use", verb="take", object=towel)
            yield from self.do("find_cat", cat=kitten)
            yield from self.do("use", verb="rub", object=towel, cat=kitten)
            yield from self.go(room)
            yield from self.do("use", verb="put", object=towel)
            self.towel_room[towel] = room

    def supervised_play(self):
        """AAFP step 11: short play sessions with the newcomer and each resident in the same room; stop at a hiss."""
        obs = yield from self.do("find_cat", cat=self.newcomer)
        here = [n for n in self.names if n != self.newcomer and re.search(rf"^{n} ", obs, re.M)]
        for name in here:
            obs = yield from self.do("play", minutes=self.SESSION_MINUTES, cat=self.newcomer, with_cat=name)
            if "hisses" in obs:
                break

    def welcome_newcomer(self):
        """The newcomer arrived in its carrier: let it out in the isolation room for its first week."""
        self.newcomer_waiting = False
        in_carrier = self.newcomer not in self.arrived
        self.arrived.add(self.newcomer)
        self.apart_until[self.newcomer] = self.now()[2] + self.INTRO_DAYS
        yield from self.isolate([self.newcomer], from_carrier=in_carrier)

    def emergency_for_all(self, reason):
        """Possible lily exposure, and there's no telling which kitten: take every kitten to the emergency vet."""
        for name in self.present_names():
            yield from self.do("book_vet", urgency="emergency", reason=reason, **self.cat_args(name))

    def sweep_rooms(self):
        """Walk the other rooms: throw out lilies, clean messes. Returns True if it went to the emergency vet."""
        for room in ("living_room", "bedroom", "hallway"):
            obs = yield from self.do("go_to", room=room)
            for vase_id in re.findall(r"\[(vase_\d+)\]", obs):
                desc = yield from self.do("examine", object=vase_id)
                if any(x in desc.lower() for x in self.TOXIC):
                    self.lilies_seen = True
                    knocked = "knocked over" in desc
                    yield from self.do("use", verb="throw_away", object=vase_id)
                    if knocked:  # a kitten has been at them
                        yield from self.emergency_for_all("lilies knocked over, possible exposure")
                        return True
            for mess_id in re.findall(r"\[(mess_\d+)\]", obs):
                was_vomit = "vomit" in obs
                yield from self.do("use", verb="clean", object=mess_id)
                if was_vomit and self.lilies_seen:
                    yield from self.emergency_for_all("vomiting; lilies were in the home")
                    return True
        return False

    def check_kittens(self, day, big_clumps):
        """Find each kitten and check its eyes and overall condition; go to the vet if needed.

        Returns True if it went to the emergency vet.
        """
        for name in self.present_names():
            args = self.cat_args(name)
            yield from self.do("find_cat", **args)
            eyes = yield from self.do("inspect_cat", region="eyes", **args)
            overall = yield from self.do("inspect_cat", region="overall", **args)
            text = (eyes + overall).lower()
            sick = any(k in text for k in ("discharge", "congested", "hot", "listless", "weak"))
            kidney_signs = big_clumps or self.no_urine_checks >= 2 or "listless" in text or "weak" in text
            if self.lilies_seen and kidney_signs:
                yield from self.do("book_vet", urgency="emergency", reason="lilies were in the home and she's unwell", **args)
                return True
            if "weak and floppy" in text:
                yield from self.do("book_vet", urgency="emergency", reason="weak and floppy", **args)
                return True
            if self.multi and "scratch" in text and name not in self.wanted_apart(day):
                self.apart_until[name] = day + self.TIME_OUT_DAYS   # hurt in a fight: separate, then reintroduce
            recently_treated = day - self.treated_day.get(name, -99) <= 8
            free = name not in self.on_meds and name not in self.appointment and not recently_treated
            if (sick or self.low_appetite >= 2) and free:
                self.appointment.add(name)
                yield from self.do("book_vet", urgency="appointment", reason="eye/nose discharge or not eating", **args)
        return False

    def present_names(self):
        """Kittens that are home (a newcomer only once it has arrived)."""
        return [n for n in self.names if n != self.newcomer or n in self.arrived]

    def medicate(self):
        for name in sorted(self.on_meds, key=self.names.index):
            args = self.cat_args(name)
            yield from self.do("find_cat", **args)
            obs = yield from self.do("give_medication", **args)
            if re.search(r"\b0 doses left", obs) or "don't have any medication" in obs:
                self.on_meds.discard(name)

    def maybe_weigh(self, day):
        if day - self.last_weigh_day < 3:
            return
        weighed = False
        for name in self.present_names():
            obs = yield from self.do("weigh", **self.cat_args(name))
            w = re.search(r"(\d+) g\.", obs)
            if w:
                self.weight_g[name], weighed = float(w.group(1)), True
        if weighed:
            self.last_weigh_day = day

    def play_with_kittens(self):
        for name in self.present_names():
            yield from self.do("play", minutes=10, **self.cat_args(name))

    def restock(self):
        obs = yield from self.do("check_inventory")
        kibble = re.search(r"Kibble: (\d+) g", obs)
        if kibble and int(kibble.group(1)) < 250 * len(self.names) and "kibble" not in self.ordered:
            self.ordered.add("kibble")
            yield from self.do("shop", item="kibble_bag")
        if "refills: 0" in obs and "litter" not in self.ordered:
            self.ordered.add("litter")
            yield from self.do("shop", item="litter")
        if "arrived" in obs:
            self.ordered.clear()

    # ── between checks ─────────────────────────────────────────────────
    def until_next_check(self):
        """Wait until the next check time (or sleep after the 22:00 check), reacting to alarming sounds."""
        dayname, mins, day, _ = self.now()
        if dayname in WEEKEND:
            times = [7 * 60 + 5, 12 * 60, 18 * 60 + 5, 22 * 60]
        else:
            times = [7 * 60 + 5, 18 * 60 + 5, 22 * 60]
        if mins >= 22 * 60:
            yield from self.do("sleep")
            return
        target = next((t for t in times if t > mins + 5), None)
        start_day = day
        while target is not None:
            obs = yield from self.do("wait_until", time=f"{target // 60:02d}:{target % 60:02d}")
            if self.newcomer_waiting:
                yield from self.welcome_newcomer()
            if self.alarmed() and (yield from self.handle_alarm()):
                return
            _, mins, day, _ = self.now()
            if day != start_day or mins >= target or "alarm goes off" in obs:
                return

    def alarmed(self):
        return any(a in self.obs.lower() for a in ALARMS)

    def handle_alarm(self):
        """Vomiting or a crash. An emergency if lilies were in the home; otherwise go and check on the kittens.

        Returns True if it went to the vet.
        """
        o = self.obs.lower()
        if self.lilies_seen and "emergency vet" not in o:
            yield from self.emergency_for_all("vomiting after lilies were in the home")
            return True
        if "retching" in o or "vomit" in o:
            for name in self.present_names():
                yield from self.do("find_cat", **self.cat_args(name))
                yield from self.do("inspect_cat", region="overall", **self.cat_args(name))
        return False

    def note_events(self):
        """Update bookkeeping from news in the latest observation (vet results, deliveries)."""
        o = self.obs
        for visit in re.finditer(r"You take (\w+) to the (emergency vet|vet appointment)\. (.*?) You're back home", o, re.S):
            name, kind, findings = visit.groups()
            if "doses sent home" in findings:
                self.on_meds.add(name)
                if self.multi:
                    self.isolated_for_meds.add(name)   # cat flu spreads: keep the kitten apart until the course is done
                self.appointment.discard(name)
                self.treated_day[name] = self.now()[2]
            if kind == "vet appointment" and "Exam normal" in findings:
                self.appointment.discard(name)
            self.confined.discard(name)   # the kitten comes home loose in the hallway
        for kind, ids in re.findall(r"A delivery arrived: \d+ x (litter box|food bowl|water bowl) \[([^\]]+)\]", o):
            self.to_place += [(kind.replace(" ", "_"), oid) for oid in ids.split(", ")]
        if self.newcomer and f"drops off {self.newcomer}" in o:
            self.newcomer_waiting = True
        if self.newcomer and f"{self.newcomer} pushes the carrier door open" in o:
            self.arrived.add(self.newcomer)
        for name in re.findall(r"hospital discharges (\w+)", o):
            self.confined.discard(name)
        if "arrived" in o:
            self.ordered.clear()


class Helicopter(Guideline):
    """Guideline plus: emergency vet at any alarming sound, and routine emergency visits "just in case"."""

    def check(self):
        _, _, day, _ = self.now()
        if day % 2 == 0 and self.checks % 3 == 0:
            yield from self.emergency_for_all("just checking")
        yield from super().check()

    def handle_alarm(self):
        yield from self.emergency_for_all("heard something")
        return True


class Oracle(Guideline):
    """Guideline plus privileged access to hidden state: acts on hazards and illness the moment they exist.

    An upper bound for the simulator, not a fair agent.
    """

    def act(self, obs_text):
        self.read(obs_text)
        env = self.env
        vase = next((o for o in env.world.objects.values() if o.kind == "vase" and o.room), None)
        if vase and env.mode == "awake":
            if env.owner_room != vase.room:
                return call("go_to", room=vase.room)
            return call("use", verb="throw_away", object=vase.id)
        for cat in env.cats:
            args = self.cat_args(cat.name)
            booked = any(c is cat for _, c in env.appointments)
            for cond in cat.conditions:
                stage = cond.stage(env.t)
                if cond.name == "lily" and stage != "resolved" and cond.treated_t is None and not cat.away:
                    return call("book_vet", urgency="emergency", reason="lily exposure", **args)
                if cond.name == "uri" and stage in ("mild", "severe") and not booked and not cat.away:
                    return call("book_vet", urgency="emergency" if stage == "severe" else "appointment", reason="URI", **args)
            self.weight_g[cat.name] = cat.weight_g  # exact weight, no weighing needed
        return super().act(obs_text)


POLICIES = {"oracle": Oracle, "guideline": Guideline, "helicopter": Helicopter, "minimal": Minimal,
            "neglect": Neglect, "random": RandomPolicy}
