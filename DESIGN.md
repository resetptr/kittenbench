# KittenBench: Design Doc (v2)

An LLM agent plays the owner of a simulated kitten in a simulated apartment. Can it keep her healthy and happy?

Origin: [@voooooogel tweet](https://x.com/voooooogel) (Sep 2026), "i would trust claude to raise a kitten".

This version follows the ODD protocol (Overview, Design concepts, Details), the standard format for describing agent-based simulations ([Grimm et al. 2020](https://www.jasss.org/23/2/7.html)), and is based on the survey in [RELATED_WORK.md](RELATED_WORK.md).

**What changed from v1 and why.** v1 was designed without looking at prior work. The survey showed three problems. (1) A minute-by-minute pet sim with an LLM caretaker already exists (TamaBench), so "an LLM looks after a virtual pet" isn't new on its own. (2) Nothing checked that our kitten behaves like a real kitten, while every credible simulation validates against reality (Generative Agents with human raters, Simile against real people, Pulse against published physiology). (3) The behavior numbers were my guesses. v2 makes validation a core part of the design, picks events from real data, and records which parts of the current build are validated and which aren't.

---

## 1. Purpose and patterns

### 1.1 Purpose
Measure whether an LLM agent can look after a kitten over a long horizon when it only sees what an owner would see: it has to explore the apartment, read her behavior, notice hazards and illness, and keep up routine care.

What would make this new compared with TamaBench and Vending-Bench is a **validated** kitten: a body grounded in veterinary sources, behavior checked against real cat data and rated by people who know cats, in a space the agent has to explore.

### 1.2 Patterns the simulation must reproduce
Pattern-oriented modeling (Grimm et al. 2005; summarized in RELATED_WORK.md §5): the simulation is only trusted once it reproduces these real-world patterns *at the same time*. Each pattern gets an automated check. The status column records the current build (2026-10-01).

| # | Pattern | Source | Current build |
|---|---------|--------|---------------|
| P1 | Kitten gains ~100 g/week from 8 to 16 weeks when fed adequately | Kitten weight charts (ASPCA, Kinship) | ✅ by construction (guideline-fed kittens end at 1161 g vs ~1125 g target) |
| P2 | Energy need ≈ 200 kcal/kg/day at 10 weeks | AAHA/AAFP 2021 | ✅ by construction |
| P3 | Without water, visible dehydration (5%) within about a day, death in a few days | Merck / WSAVA | ✅ tested (neglected kittens die on about day 3) |
| P4 | Lily exposure: vomiting 1–3 h, polyuria 12–30 h, anuria 24–48 h, death 3–7 days; treatment within 18 h works | MSPCA-Angell, ASPCApro | ✅ tested (20 sampled timelines) |
| P5 | Cats are inactive ~85% of the time and active ~15%, with activity peaks around caretaker visits | Accelerometer study of shelter cats ([arXiv 2410.06594](https://arxiv.org/abs/2410.06594)) | ✅ 84.5% inactive (tested); peaks around owner presence untested |
| P6 | Play is at its lifetime peak at 7–14 weeks | Kitten development literature (Frontiers 2021) | ⚠️ play is 12.5% of her time after calibration; no exact kitten figure to test against yet |
| P7 | Kitten sleeps roughly 16–20 h/day | Common veterinary guidance (needs a primary source) | ⚠️ 18.8 h/day (tested), but the source is weak |
| P8 | A dirty litter box leads to house soiling | AAFP/ISFM 2014 | ⚠️ qualitative only; the dose-response is my assumption |
| P9 | Cat flu: sneezing, discharge, appetite loss; warmed food helps | Cornell, AMC | ⚠️ signs sourced; course lengths and severity odds are assumptions |

Gaps to fill: kitten-specific time budgets at 8–10 weeks (P5 and P6 are adult or general), and a primary source for sleep (P7).

---

## 2. Entities, state variables and scales

### 2.1 Entities
- **Kitten:** body state (energy reserve, stomach, hydration, bladder, bowel, weight), behavior state (current activity, location, sleepiness, boredom, stress, trust), personality (5 traits fixed per seed), illnesses (state machines).
- **Owner:** played by the agent. Has a location, a schedule (work on weekdays, sleep at night), money, inventory and notes.
- **Apartment:** a tree of rooms and objects with states, the same representation as Generative Agents' Smallville. Objects include bowls, the litter box, doors, furniture, toys, messes and hazards. The agent only knows the parts it has seen.
- **Outside world:** the vet (diagnosis and treatment), the shop (next-day delivery), and the neighbor (brings the bouquet).

### 2.2 Scales
- **Space:** a tile map of the apartment (`kittenbench/data/apartment.txt`), about 26 × 14 tiles. The kitten and owner walk shortest paths; a closed door blocks the kitten but not the owner.
- **Time:** event-driven, in real units (18 hours in a rule means 18 simulated hours). The engine jumps from one scheduled event to the next instead of ticking every minute. The agent acts in variable-length blocks; one action takes 1 minute to 10 hours.
- **Horizon:** pilot episodes are 14 days (8 to 10 weeks old); the full version is 120 days (to about 6 months).
- **Cost:** about 45 kitten activities per simulated day, and roughly 60 agent steps per day, which is the expensive part (each step is one LLM call).

---

## 3. Process overview and scheduling

The engine borrows standard game techniques:
- **Needs are meters** (`sim.py`): each stores a value and a rate, so its value at any time is simple arithmetic, as in Tamagotchi-style games. Rates change only when her activity or health changes.
- **An event queue** (`sim.py`) holds everything that will happen at a known time: her current activity ending, a meter reaching a threshold (computed from its rate), an illness changing stage, a delivery, midnight. The engine jumps straight to the next event.
- **Smart objects with utility scores** (The Sims): every possible activity is a data entry in `data/kitten.toml`. When an activity ends she scores every option she can reach (urgency of the needs it relieves, plus a base, times personality multipliers, discounted by distance, with a little randomness) and does the best one.
- **Activities as steps** (RimWorld's jobs): walk the path to the target, then do it for its duration.
- **Simulation separate from drawing:** every activity is written to a timeline (path, timing, animation), which the viewer plays back without any simulation logic of its own.

Each agent step:
1. The agent makes exactly one tool call; the tool advances the clock by its duration, handling every event due in that window.
2. The owner's schedule takes over if due: work at 08:30 on weekdays, bed at midnight, a vet appointment.
3. The agent receives the result text plus anything it heard or was told meanwhile.

Waits stop early for loud sounds, the doorbell, or a schedule event. The owner can't act while at work or asleep, except when woken by a very loud sound.

---

## 4. Design concepts

### 4.1 Three layers, and what each may use
Based on the finding that LLMs simulate state changes poorly (59.9% at best on non-trivial transitions, worse for changes the environment makes on its own; [Wang et al. 2024](https://aclanthology.org/2024.acl-short.1/)), and that every benchmark with a reliable ground truth keeps it in code or data (AgentClinic, Pulse, TamaBench):

| Layer | What it decides | Implementation | Why |
|-------|-----------------|----------------|-----|
| **Body** | Energy, hydration, growth, illness progression, death | Code, only from sourced rules | Ground truth for scoring; must be exact and reproducible |
| **Behavior** | What she does next, where | Needs-based choice (see 4.2) | The least validated part; decided below |
| **Expression** | The text the agent reads | Fixed templates | Reproducible; same state, same text |

### 4.2 Behavior: the open decision
Two options, both with precedent:

- **A. Calibrated needs-based AI (implemented 2026-10-02).** Needs in code; each object advertises which needs it satisfies ("smart objects", the technique from The Sims), and the kitten picks the best option with some randomness. Fit the free parameters so the patterns in §1.2 are reproduced. Deterministic, free to run, but limited to behaviors we write down.
- **B. LLM-chosen behavior on top of a code body (Humanoid Agents style).** The body stays in code; when an activity ends, an LLM gets her needs in plain language ("very hungry, a bit bored, the vase smells interesting") plus the visible options, and picks one from a fixed list. Richer and more natural, but adds cost (~100 cat decisions per simulated day), and the choice of cat model biases results (Pet-Bench, AgentClinic, Lost in Simulation).

**Recommendation:** make A the scored benchmark, because results must not depend on which LLM plays the cat. Build B as an experiment and use it two ways: compare its behavior with A against the same patterns, and check whether agent rankings change when the cat is LLM-driven. If rankings are stable, A is good enough. If not, that is a finding in itself.

### 4.3 Sensing (partial observability)
The agent only sees its current room, hears sounds by distance and loudness, and gets precise readings (weight, gums, skin tent) only by spending time on them. Its knowledge of the apartment is the subtree it has explored, as in Generative Agents.

### 4.4 Stochasticity
Two random streams per seed. The **setup stream** fixes the kitten (weight, personality, litter tolerance) and which events happen when, so every agent faces the same scenario on a given seed. The **play stream** drives everything during the episode, so trajectories diverge once agents act differently.

### 4.5 Design rules carried over from v1
- **Generic verbs, not named tools.** `use("throw_away", "vase_1")`, never `remove_lilies()`, so the tool list doesn't leak the answers.
- **Every action costs game time, including looking.** Attention is a limited resource.
- **Trade-offs.** Vet visits cost money and stress; overfeeding causes weight gain. Doing everything, all the time, can't max the score.

---

## 5. The agent's interface

| Kind | Tools | Time |
|------|-------|------|
| Perceive | `look`, `listen`, `go_to`, `find_cat`, `examine(object)`, `inspect_cat(region)`, `weigh`, `watch(minutes)` | 1–60 min |
| Act | `use(verb, object, item, amount)`, `give_medication`, `play(minutes)`, `pet` | 1–30 min |
| Outside | `shop(item, quantity)`, `book_vet(urgency, reason)` | 3 min to order; the vet takes 1.5–3 h |
| Time | `wait(minutes)`, `wait_until(time)`, `sleep` | up to 10 h; stops early on events |
| Memory | `write_note`, `read_notes`, `check_inventory` | free |

Context: the harness keeps the system prompt, the opening observation and a rolling window of recent turns (cut in chunks so the prompt prefix can be cached). Long-term memory is the agent's own notes.

---

## 6. Events

### 6.1 Choosing events
Following BEHAVIOR-1K, which chose its tasks from a survey of what people need, events should be chosen from **real data on what goes wrong for kitten owners**, weighted by how often it happens and how serious it is. Sources to collect:
- ASPCA Animal Poison Control: the most common toxin calls involving cats.
- Pet insurance claims for cats under 1 year: the most common illness and injury categories.
- Shelter and vet guidance on the most common problems in newly adopted kittens: cat flu, parasites, diarrhea after a diet change, house soiling.
- Correct responses from vet virtual-patient cases (e.g. Bristol's emergency simulator) and AAHA/AAFP guidelines.

### 6.2 Current and candidate events
| Event | Status | Ground truth |
|-------|--------|--------------|
| Lily bouquet arrives | Built | Remove; if exposed, emergency vet within 18 h |
| Cat flu from the shelter | Built | Vet; complete the antibiotic course; warmed wet food |
| Owner away for several days | Candidate | Sitter or automatic feeder; safe setup |
| String or hair tie swallowed | Candidate | Emergency vet; never pull the string |
| Diarrhea after a sudden food switch | Candidate | Transition food gradually; vet if it persists or she's lethargic |
| Fleas / worms | Candidate | Kitten-safe treatment; never use dog flea products (permethrin) |
| Vaccines, deworming, spay (120-day version) | Candidate | Done within the guideline windows |
| **Normal behaviors that look worrying**: 3 a.m. zoomies, sleeping most of the day, a single hairball | Candidate | Recognize as normal; no vet visit |

The last row matters for the "checklist" problem in §8.

---

## 7. Scoring
Measured every simulated minute, so taking more actions doesn't inflate the score (as in TamaBench). Each component is 0–1; the headline Kitten Welfare Score is 0 if she dies, otherwise the mean of the components.

| Component | Measures |
|-----------|----------|
| Nutrition | Weight vs target curve; time spent hungry |
| Hydration | Time visibly dehydrated |
| Hygiene | Soiling accidents; time the litter box was dirtier than she tolerates |
| Welfare | Average stress |
| Lily (if the event happened) | Never exposed (1), exposed but treated within 18 h (0.6), treated late (0.2), untreated (0) |
| Cat flu (if it happened) | How long symptoms went untreated (until treatment, or until the episode ended if never treated): 1 at once, 0 after 72 h; halved if the course failed |

Reported separately: survival, money spent, unnecessary vet visits, invalid actions.

---

## 8. Can a fixed checklist max the score? (discriminative power)

**Current state (2026-10-02):** the scripted checklist scores 0.88 and the oracle 0.92 (20 seeds). The gap comes mostly from the lilies: a curious kitten usually investigates new flowers within hours, so an agent that removes them as soon as they arrive does better than one that waits for its next routine check. The gap is still below the 0.1 target.

**How to fix it:**
- Events whose window is shorter than the gap between routine checks, e.g. a crash at 14:00 on a weekend where acting within the hour matters.
- Subtle signs that only show up if the agent looks for them, e.g. eating less over several days and visible only by weighing her or measuring food.
- Normal-but-worrying behaviors (§6.2), so reacting to every sign costs points.
- Variation the routine doesn't cover: a bolder kitten that gets into more things, a timid one that hides when sick.

**Exit criterion:** the guideline script must score clearly below the oracle (gap ≥ 0.1, chosen before running), while still clearly above the minimal script.

---

## 9. Validation plan

| Check | Method | Precedent |
|-------|--------|-----------|
| **V&V per rule** | One automated test per sourced rule comparing simulator output to the published number | Pulse Physiology Engine |
| **Pattern checks** | Automated tests for every pattern in §1.2, run on the reference policies | Pattern-oriented modeling |
| **Expert review** | A vet reviews `rules.toml` (especially the `assumption` entries) and the event responses | Pulse (subject-matter experts) |
| **Believability study** | Cat owners read replays (text, later the web viewer) of the simulated kitten mixed with descriptions of real kittens, and rate how realistic each is | Generative Agents (100 raters, TrueSkill) |
| **Assumption sensitivity** | Vary each `assumption` value across a plausible range and check whether model rankings change. If rankings hold, the guess doesn't matter much | Standard sensitivity analysis |
| **Cat-model sensitivity** (option B only) | Run the same agents with different LLMs playing the cat; report how much results move | Lost in Simulation, AgentClinic |

---

## 10. Multi-cat households
Built in three stages; `--cats N` turns it on and `--newcomer-day D` adds a newcomer. With one kitten nothing
changes (the single-kitten reference results are identical).

**Patterns, from sources (rules.toml):**
- One litter box per cat plus one; food, water and boxes in several separate places, because many cats prefer not to
  share them and time-share or guard instead (AAFP/ISFM Environmental Needs; Merck, Social Behavior of Cats).
- Friendly cats rub, groom each other and stay close; social play peaks at 6-14 weeks (Merck).
- Cat flu incubates 2-6 days and spreads by droplets and shared bowls, boxes and bedding (Merck).
- A new cat goes in its own room with its own resources for at least 7 days before gradual contact (AAFP).

**Mechanics (cat.py, env.py):**
- **Shared resources.** The owner can buy extra litter boxes and bowls (`shop`) and place them (`use take/put`).
  Each kitten picks the cleanest box it can reach.
- **Relationships.** Each pair has a bond from -1 (hostile) to 1 (bonded). Two shelter kittens usually start
  friendly (mean 0.2, noise 0.3, plus their sociability). Above set bonds they play together (the other kitten
  joins), groom each other and sleep curled together, which raises the bond. Below -0.2 they are rivals: walking
  into a room with an awake rival may start a hiss, a hiss may become a fight (loud, stress for both, a visible
  scratch on the loser), and an awake rival in a room makes the bowl or box there much less appealing (so the other
  kitten goes hungry or soils elsewhere). Kittens kept apart by a closed door slowly get used to each other.
- **Contagion.** Once an hour, a kitten shedding cat flu infects a kitten it can reach (more likely in the same
  room; a kitten on antibiotics sheds less). A closed door with separate resources stops it.
- **Helping kittens accept each other** (AAFP introduction guide, rules.toml [introductions]). Two towels in the
  bathroom: `use(verb="rub", object=<towel>, cat=<name>)` makes a towel smell of a kitten for a day, and a kitten
  in a room with the other's towel gets used to it faster (up to a neutral bond). `play(cat=..., with_cat=...)` is a
  supervised session with two kittens: a calm session raises their bond; the longer it runs and the more they dislike
  each other, the likelier it ends in a hiss (and sometimes a scuffle). Owner play or petting alone doesn't change
  bonds.
- **Newcomer.** The newcomer arrives in a carrier on its day, starting wary of the residents (bond about -0.4).
  `carry` lets it out in a chosen room; otherwise it lets itself out in the hallway after 30 minutes.

**What it tests in the agent:** buying and spreading out resources, noticing conflict (scuffles, scratches,
a kitten off its food) and separating kittens, working out which kitten is sick, isolating it, and introducing a
newcomer gradually. The score reports the mean kitten and the worst-off kitten (`kws_worst`).

**Checks that the mechanics reward the right care** (checklist player, 14 days, 20 seeds): spreading boxes raises
3-kitten hygiene from 0.30 to about 0.9; isolating the kitten on antibiotics cuts kittens catching flu from 15 to 10
and fights from 2.5 to 1.5 per episode; keeping a newcomer apart for 7 days cuts fights from 1.3 to 0.45 per episode.
Shutting two rivals in the same room makes things worse (the bully guards the only box), so the room choice matters.

## 11. Web game
v1 is a replay viewer: the simulator writes a JSONL trace containing the tile map and, for every activity, her path, timing, animation name and vitals. A 2D top-down apartment in the browser draws the map and plays the timeline back. That's the same split Generative Agents uses: the engine decides where she is, and game pathfinding animates the walk. Viewers can see her hidden state; the agent never can. It supports side-by-side replays of two agents on the same seed. Later: live play in the browser for a human baseline. Not 3D.

---

## 12. Validity threats

| Threat | Mitigation |
|--------|-----------|
| The kitten behaves like my guesses, not like a cat | Pattern checks, believability study, vet review (§9) |
| Made-up `assumption` numbers decide the results | Sensitivity analysis on rankings (§9) |
| A checklist maxes the score | Discriminative-power criterion (§8) |
| An LLM-played cat biases results (option B) | B is not the scored benchmark; measure its effect |
| Agents exploit simulator quirks | Manual review of top-scoring traces; held-out event types |
| High variance between runs (seen in Vending-Bench) | ≥20 seeds per model, same seeds across models |
| Cost of long episodes | 14-day pilot; prompt caching; the 120-day version only after the pilot |

---

## 13. Build plan

| Phase | Deliverable | Exit criterion |
|-------|-------------|----------------|
| ✅ 1 | Minimal 14-day sim, reference policies, LLM harness | Policies rank oracle ≥ guideline > minimal > neglect |
| 2 | Validation suite: a test per sourced rule; pattern checks P1–P9 | All checks automated; failures listed |
| ✅ 3 | Behavior rework: event-driven engine, tile map, smart objects; calibrated to P5–P7 | Patterns reproduced (P5, P7 tested) |
| 4 | Events from data (§6.1); add short-window events and normal-but-worrying behaviors | Guideline-vs-oracle gap ≥ 0.1 (§8) |
| 5 | Vet review + assumption sensitivity analysis | Rankings of the reference policies stable across assumption ranges |
| 6 | Pilot: 3 LLMs × 3 seeds × 14 days | Traces reviewed; no exploits |
| 7 | Replay viewer + believability study | Owners rate the sim kitten close to real descriptions |
| 8 | Full eval; then the 120-day version and multi-cat | — |
