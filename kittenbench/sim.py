"""The simulation engine's building blocks: the data files, time, meters, the event queue, and the apartment.

Sections:
    data         loads data/*.toml and the two formulas derived from the vet rules
    time         simulated time and the owner's schedule constants
    meters       needs as "value + rate", Tamagotchi-style
    event queue  the simulation jumps from one scheduled event to the next
    apartment    the tile map, rooms, doors, objects and pathfinding
"""

import heapq
import itertools
import math
import tomllib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


# ════ data ════════════════════════════════════════════════════════════════════════════════════════
# Loads the data files in data/ and provides the formulas derived from them.
#
# rules.toml         body facts from vet sources (energy, water, growth, illnesses, costs)
# kitten.toml         behavior tuning: need rates, stressors, urgency curves
# kitten.toml  everything the kitten can do
# apartment.toml     what's where in the apartment (with apartment.txt, the tile map)

DATA = Path(__file__).parent / "data"


def _load(name):
    return tomllib.loads((DATA / name).read_text())


RULES = _load("rules.toml")                          # body facts from vet sources
NEEDS = _load("kitten.toml")                         # behavior tuning: need rates, stressors, urgency curves...
INTERACTION_SPECS = NEEDS.pop("interaction")         # ...and the kitten's activities ([[interaction]] entries)
APARTMENT = _load("apartment.toml")
APARTMENT_MAP = (DATA / "apartment.txt").read_text().splitlines()

SIXTEEN_WEEKS_DAYS = 16 * 7


def energy_need_kcal(weight_g, age_days):
    """Daily energy need (kcal/day): resting energy requirement times a growth multiplier for the kitten's age."""
    e = RULES["energy"]
    rer = e["rer_coef"] * (weight_g / 1000) ** e["rer_exp"]
    multiplier = e["multiplier_under_16wk"] if age_days < SIXTEEN_WEEKS_DAYS else e["multiplier_16wk_plus"]
    return rer * multiplier


def target_gain_g_per_day(age_days):
    """How many grams per day a well-fed kitten of this age should gain: flat to 16 weeks, then tapering."""
    g = RULES["growth"]
    weeks = age_days / 7
    if weeks < 16:
        return g["gain_g_per_week_to_16wk"] / 7
    taper = min(1.0, (weeks - 16) / 8)  # 0 at 16 weeks, 1 at 24 weeks and after
    early, late = g["gain_g_per_week_to_16wk"], g["gain_g_per_week_at_24wk"]
    return (early + taper * (late - early)) / 7


# ════ time ════════════════════════════════════════════════════════════════════════════════════════
# Simulated time.
#
# Time is `t`: minutes (fractional) since midnight at the start of day 0. The engine jumps between events, so times are rarely whole minutes.
# Day 0 is a Saturday, so days 0, 1 are the first weekend and days 2-6 are Monday-Friday.

MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 24 * MINUTES_PER_HOUR

DAY_NAMES = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# The owner's schedule, in minutes after midnight.
EPISODE_START = 10 * MINUTES_PER_HOUR            # the kitten arrives at 10:00 on day 0
WAKE_TIME = 7 * MINUTES_PER_HOUR                 # alarm; the owner must be asleep from midnight until this
WORK_START = 8 * MINUTES_PER_HOUR + 30           # weekdays: leave for work at 08:30 ...
WORK_END = 18 * MINUTES_PER_HOUR                 # ... and get home at 18:00
EARLIEST_BEDTIME = 19 * MINUTES_PER_HOUR         # sleep() is refused before this


def day_of(t):
    return int(t // MINUTES_PER_DAY)


def minute_of_day(t):
    return t % MINUTES_PER_DAY


def is_workday(day):
    return day % 7 >= 2  # 0 = Saturday, 1 = Sunday


def day_name(day):
    return DAY_NAMES[day % 7]


def hm(t):
    """Format the clock time of `t` as HH:MM."""
    m = int(minute_of_day(t))
    return f"{m // MINUTES_PER_HOUR:02d}:{m % MINUTES_PER_HOUR:02d}"


# ════ meters ══════════════════════════════════════════════════════════════════════════════════════
# Meters: numbers that change at a steady rate between events.
#
# This is how Tamagotchi-style games track needs without updating every frame. A meter stores its value at
# one moment and its rate of change; its value at any later moment is simple arithmetic. Whenever the rate
# changes (the kitten starts eating, falls asleep, gets sick) the meter is "rebased": the current value is saved
# and the new rate applies from then on. The engine also asks a meter *when* it will reach a level, and
# puts that moment in the event queue, so nothing is missed between events.

CLOSE_ENOUGH = 1e-9   # floating-point slack: a meter this close to a level counts as having reached it


@dataclass
class Meter:
    value0: float           # value at time t0
    rate: float = 0.0       # change per minute
    t0: float = 0.0
    low: float = 0.0        # the value never goes outside [low, high]
    high: float = math.inf

    def value(self, t):
        return min(self.high, max(self.low, self.value0 + self.rate * (t - self.t0)))

    def set_rate(self, t, rate):
        """From time t on, change at `rate` per minute."""
        self.value0, self.t0, self.rate = self.value(t), t, rate

    def set(self, t, value):
        self.value0, self.t0 = min(self.high, max(self.low, value)), t

    def add(self, t, amount):
        self.set(t, self.value(t) + amount)

    def time_to_reach(self, t, level):
        """The moment (> t) it will reach `level` at the current rate, or None if it never will
        (or is already there, within rounding error)."""
        now = self.value(t)
        if abs(level - now) < CLOSE_ENOUGH:
            return None
        if self.rate > 0 and now < level <= self.high:
            return t + (level - now) / self.rate
        if self.rate < 0 and now > level >= self.low:
            return t + (level - now) / self.rate
        return None


def response_curve(points, x):
    """Piecewise-linear curve through [(x, y), ...]: 0 before the first point, the last y after the last."""
    if x <= points[0][0]:
        return points[0][1] if x == points[0][0] else 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


# ════ event queue ═════════════════════════════════════════════════════════════════════════════════
# The event queue: the simulation jumps from one scheduled event to the next instead of ticking every minute.
#
# Events are (time, kind, data). Some events become outdated, e.g. "the bladder will be full at 14:30" stops
# being true if the kitten pees at 14:00. Rather than hunting them down, each event can carry a `tag`; the code that
# scheduled it bumps its tag when plans change, and outdated events are skipped when they come up.

@dataclass(order=True)
class Event:
    t: float
    seq: int                                  # breaks ties: events at the same time run in the order scheduled
    kind: str = field(compare=False)
    data: dict = field(compare=False, default_factory=dict)
    tag: tuple | None = field(compare=False, default=None)  # (owner name, version); see EventQueue.valid


class EventQueue:
    def __init__(self):
        self._heap = []
        self._seq = itertools.count()
        self._versions = {}   # owner name -> current version; events with an older version are outdated

    def push(self, t, kind, data=None, owner=None):
        tag = (owner, self._versions.get(owner, 0)) if owner else None
        heapq.heappush(self._heap, Event(t, next(self._seq), kind, data or {}, tag))

    def invalidate(self, owner):
        """Mark every pending event scheduled under `owner` as outdated."""
        self._versions[owner] = self._versions.get(owner, 0) + 1

    def valid(self, event):
        return event.tag is None or self._versions.get(event.tag[0], 0) == event.tag[1]

    def next_time(self):
        """Time of the next valid event, or None."""
        while self._heap and not self.valid(self._heap[0]):
            heapq.heappop(self._heap)
        return self._heap[0].t if self._heap else None

    def pop(self):
        self.next_time()  # drop outdated events first
        return heapq.heappop(self._heap)


# ════ apartment ═══════════════════════════════════════════════════════════════════════════════════
# The apartment: a tile map (data/apartment.txt) with rooms, doors and objects, plus pathfinding.
#
# Positions are tiles: (row, column) in the map. The kitten and the owner walk tile by tile along shortest
# paths; a closed door blocks the kitten but not the owner. The game viewer draws the same map.

# How far a sound carries. The owner hears a sound if they're home and awake and:
QUIET = 0       # ... in the same room (eating, sneezing, litter scratching)
NORMAL = 1      # ... in the same or an adjacent room (meowing, playing)
LOUD = 2        # ... anywhere in the apartment; it also cuts a wait() short (crying)
VERY_LOUD = 3   # ... anywhere, and it wakes them if asleep (retching, a crash)

DOOR, DOORWAY = "D", "d"   # map characters (walls are "#", rooms are letters)
MOVABLE = {"food_bowl", "water_bowl", "litter_box", "vase", "toy", "towel"}   # the owner can pick these up
REMOVABLE = {"vase", "mess", "toy"}                                   # the owner can throw these away
NOT_LISTED = {"hiding_spot"}                                          # part of furniture; not shown by look()


@dataclass
class Obj:
    """Something in the apartment. `tile` and `room` are None while the owner is carrying it.

    `props` depends on `kind`: see apartment.toml for the starting values. Objects created later:
        vase: flowers (description), lilies (bool), knocked (bool)
        mess: type ("vomit" | "urine" | "feces"), where (description), t
        towel: name, scent_of (index of the kitten it was last rubbed on, -1 for none), scented_t
    """
    id: str
    kind: str
    tile: tuple | None
    room: str | None
    props: dict = field(default_factory=dict)


class Apartment:
    def __init__(self):
        room_names = APARTMENT["rooms"]
        self.grid = APARTMENT_MAP
        self.tile_room = {}   # every walkable tile -> its room (doors and doorways belong to the hallway)
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch in room_names:
                    self.tile_room[(r, c)] = room_names[ch]
                elif ch in (DOOR, DOORWAY):
                    self.tile_room[(r, c)] = "hallway"
        self.rooms = sorted(set(self.tile_room.values()))
        self.objects = {}     # id -> Obj, in creation order
        self._counter = {}
        for spec in APARTMENT["object"]:
            self.add(spec["kind"], tuple(spec["tile"]), dict(spec.get("props", {})), oid=spec["id"])
        self.door_at = {o.tile: o for o in self.objects.values() if o.kind == "door"}
        self.neighbors = self._room_adjacency()

    # ── objects ────────────────────────────────────────────────────────
    def add(self, kind, tile, props=None, oid=None):
        """Create an object on a tile. Without an explicit id it is numbered, e.g. mess_1, mess_2."""
        if oid is None:
            n = self._counter[kind] = self._counter.get(kind, 0) + 1
            oid = f"{kind}_{n}"
        self.objects[oid] = Obj(oid, kind, tile, self.tile_room.get(tile), props or {})
        return self.objects[oid]

    def place(self, obj, tile):
        """Put an object down (tile=None: the owner is holding it)."""
        obj.tile, obj.room = tile, (self.tile_room.get(tile) if tile else None)

    def in_room(self, room):
        """Objects the owner sees in `room`, including doors leading into it."""
        return [o for o in self.objects.values()
                if o.kind not in NOT_LISTED and (o.room == room or (o.kind == "door" and o.props["room"] == room))]

    def of_kind(self, kind):
        return [o for o in self.objects.values() if o.kind == kind and o.tile]

    def first(self, kind):
        """The first placed (not carried) object of a kind, e.g. the food bowl. None if there isn't one."""
        found = self.of_kind(kind)
        return found[0] if found else None

    def litter_dirt(self):
        """How dirty the cleanest litter box is: unscooped deposits plus leftover smell. None if there's no box out."""
        boxes = self.of_kind("litter_box")
        if not boxes:
            return None
        return min(b.props["urine"] + b.props["feces"] + b.props["residue"] for b in boxes)

    def next_id(self, kind):
        """An unused id for a new object of a kind: the first is `kind`, then kind_2, kind_3..."""
        if kind not in self.objects:
            return kind
        n = 2
        while f"{kind}_{n}" in self.objects:
            n += 1
        return f"{kind}_{n}"

    # ── movement ───────────────────────────────────────────────────────
    def walkable(self, tile, cat):
        if tile not in self.tile_room:
            return False
        door = self.door_at.get(tile)
        return not (cat and door and not door.props["open"])

    def path(self, start, goal, cat=True):
        """Shortest list of tiles from start to goal (both included), or None if it can't be reached."""
        if start == goal:
            return [start]
        came_from = {start: None}
        queue = deque([start])
        while queue:
            tile = queue.popleft()
            r, c = tile
            for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nxt not in came_from and self.walkable(nxt, cat):
                    came_from[nxt] = tile
                    if nxt == goal:
                        path = [nxt]
                        while came_from[path[-1]] is not None:
                            path.append(came_from[path[-1]])
                        return path[::-1]
                    queue.append(nxt)
        return None

    def reachable_rooms(self, tile):
        """Rooms the kitten can walk to from `tile` with the doors as they are."""
        return {room for room in self.rooms if self.path(tile, self.room_center(room)) is not None}

    def room_center(self, room):
        """A standard standing spot in a room: the floor tile closest to its middle."""
        tiles = [t for t, r in self.tile_room.items() if r == room and self.grid[t[0]][t[1]] not in (DOOR, DOORWAY)]
        mid_r = sum(t[0] for t in tiles) / len(tiles)
        mid_c = sum(t[1] for t in tiles) / len(tiles)
        return min(tiles, key=lambda t: (abs(t[0] - mid_r) + abs(t[1] - mid_c), t))

    def random_tile(self, room, rng):
        tiles = sorted(t for t, r in self.tile_room.items() if r == room and self.grid[t[0]][t[1]] not in (DOOR, DOORWAY))
        return rng.choice(tiles)

    def _room_adjacency(self):
        """Which rooms touch each other (through a door or doorway)."""
        adj = {room: set() for room in self.rooms}
        for (r, c), room in self.tile_room.items():
            for nxt in ((r + 1, c), (r, c + 1)):
                other = self.tile_room.get(nxt)
                if other and other != room:
                    adj[room].add(other)
                    adj[other].add(room)
        return adj
