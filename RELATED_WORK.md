# KittenBench: Related Work

KittenBench is a simulation with an agent inside it. Prior work splits into four groups: LLM-driven simulations of living things, simulated environments used as agent benchmarks, rule-based simulations of bodies and creatures, and the methodology for checking that a simulation matches reality. For each work: what it simulates, how, how it checks realism, and what we should take from it.

Surveyed 2026-10-01.

---

## 1. LLM-driven simulations of living things

### Generative Agents (Park et al., 2023) — [arXiv 2304.03442](https://arxiv.org/abs/2304.03442)
- **What:** 25 LLM-driven characters living in "Smallville", a Sims-like town, for two game days.
- **How:** each agent has a memory stream (a log of everything it observed), periodically writes reflections (higher-level conclusions from those memories), and plans its day. The world is a tree of areas and objects ("stove" is a child of "kitchen"). Each agent only knows the part of the tree it has explored. When an agent acts on an object, the LLM is asked what happens to the object's state (a coffee machine goes from "off" to "brewing"). Movement is animated with ordinary game pathfinding.
- **Validation:** believability, not accuracy. 100 human raters ranked interview answers from four architectures and a human-written baseline. The full architecture beat the ablations (TrueSkill 29.89 vs 21.21 with no memory, planning or reflection) and also beat crowdworkers writing the answers (22.95).
- **Limitations they report:** two game days for 25 agents cost "thousands of dollars" and took days. Agents embellished facts and were overly cooperative.
- **What we take:** the tree-of-areas world with per-agent partial knowledge is the same as our exploration design, so we should adopt their representation. Their Phaser web front end is the model for our replay viewer. Their validation (human raters comparing variants) is a template for checking our cat's realism. Cost is a warning: an LLM per simulated character is expensive.

### Humanoid Agents (Wang et al., 2023) — [arXiv 2310.05418](https://arxiv.org/abs/2310.05418)
- **What:** Generative Agents plus basic needs (fullness, energy, health, social, fun), emotion and relationship closeness.
- **How:** needs are numbers on a 0–10 scale that decay at fixed rates (e.g. fullness −1 per 5 h, energy −5 per 5 h). When a need drops to 3 or below, that fact is put into the prompt in plain language ("very hungry") and the LLM revises its plan. One time step is 15 minutes; days run 06:00–24:00.
- **Validation:** 3 human annotators labeled 144 activities. Agreement with the model was F1 0.97 for fullness and 0.86 for energy, and lower (0.66–0.74) for fun, health and social, which the model over-predicted.
- **What we take:** this is the closest design to a "living kitten": numeric needs in code, with behavior chosen by an LLM that reads those needs. It shows the pattern works and where it's weak: needs with clear physical meaning are judged well, fuzzy ones poorly.

### Generative Agent Simulations of 1,000 People / Simile — [arXiv 2411.10109](https://arxiv.org/abs/2411.10109), [simile.com](https://www.simile.com/)
- **What:** agents built from two-hour interviews with 1,052 real people, then asked to answer surveys and experiments as those people.
- **Validation:** each agent's answers are compared with its real person's, normalized by how consistently the person repeats their own answers two weeks later. Agents reached 85% of that self-consistency on the General Social Survey. Simile validates weekly against real people (7,000+ evaluations).
- **What we take:** validate against the real thing, and use the real thing's own variability as the ceiling. For a cat, "the real thing" is published cat behavior data and the judgment of owners and vets, not our own intuition.

### Pet-Bench (2025) — [arXiv 2506.03761](https://arxiv.org/abs/2506.03761)
- **What:** a benchmark of LLMs *playing* virtual pets (scheduling, memory-based dialogue, emotional conversation), 7,500+ interactions, 28 models.
- **Finding:** quality varied a lot by model and size; the authors conclude general-purpose LLMs need specialized tuning to play pets well.
- **What we take:** if an LLM plays our kitten, which model plays her will change her behavior, and so the benchmark. This has to be fixed and measured, not assumed away.

---

## 2. Can an LLM be the simulator?

### Can Language Models Serve as Text-Based World Simulators? (Wang et al., ACL 2024) — [paper](https://aclanthology.org/2024.acl-short.1/)
- **What:** 76,369 state transitions from 31 text games. GPT-4 was given the current state and an action and asked to predict the next state.
- **Findings:** at most 59.9% accuracy on transitions where something actually changes. It did better on direct effects of actions (up to 77.1%) than on things the environment does by itself (at most 49.7%, e.g. water filling a cup under a running tap). Errors cluster where arithmetic, common sense or science is needed. Errors compound: 0.599¹⁰ is under 1% after ten steps. Humans scored 80% where GPT-4 scored 50%.
- **What we take:** this is the strongest argument for keeping the kitten's *body* in code. Hydration, energy, toxin timelines and growth are exactly the "environment-driven, arithmetic" transitions LLMs get wrong, and a kitten's state evolves over thousands of steps. An LLM can help with behavior and description, but not with the state that decides whether she lives.

### LLM-simulated people in agent benchmarks: τ-bench and its critics
- **τ-bench** ([arXiv 2406.12045](https://arxiv.org/abs/2406.12045)) uses an LLM to play the customer the agent serves. Later work found problems: simulated users are "overly cooperative" (they can't refuse the goal), and nearly a quarter of successful runs contain a simulated-user rule violation the score doesn't catch.
- **Lost in Simulation** ([arXiv 2601.17087](https://arxiv.org/abs/2601.17087)): agent success rates swing by up to 9 points depending on which LLM plays the user. Simulated users underestimate agents on hard tasks and overestimate them on medium ones.
- **What we take:** an LLM-played kitten would bias the benchmark in ways that are hard to see. If we use one, we must (1) fix the model and version, (2) report how results change when the cat model changes, and (3) check the cat against real behavior data.

---

## 3. Simulated environments used as agent benchmarks

### AgentClinic (Schmidgall et al., 2024) — [arXiv 2405.07960](https://arxiv.org/abs/2405.07960), [site](https://agentclinic.github.io/)
- **What:** the closest structural analog. A doctor agent (the model being tested) must diagnose a patient through conversation and by ordering tests. Separate LLM agents play the patient and the test lab, and a moderator grades the final diagnosis. Cases come from MedQA, MIMIC-IV records and NEJM case challenges.
- **Findings:** turning static exam questions into this step-by-step format cut diagnostic accuracy to as little as a tenth of the static score. Which LLM plays the patient changes the results substantially.
- **What we take:** the ground truth (the diagnosis) is fixed data, and only the *conversation* is LLM-generated. That's the split we want: a hidden health state in code, with LLMs at most generating how it shows.

### Vending-Bench (Andon Labs, 2025) — [arXiv 2502.15840](https://arxiv.org/abs/2502.15840)
- **What:** an agent runs a simulated vending-machine business (inventory, orders, pricing, daily fees) over runs of 20M+ tokens.
- **Findings:** every model had runs that derailed: misreading delivery schedules, forgetting orders, or getting stuck in "meltdown" loops. Failures did *not* line up with the context window filling up. Variance across runs was high.
- **What we take:** long-horizon coherence is the interesting failure, and it shows up only in long runs with many seeds. Our 14-day episodes may be too short to see it, and we need many runs per model.

### TamaBench (2026) — [github.com/Cheva1234/TamaBench](https://github.com/Cheva1234/TamaBench)
- **What:** an agent keeps a Tamagotchi-style pet alive for 7 simulated days. The engine runs minute by minute, with an accelerated mode that jumps over intervals where nothing changes. Welfare is measured every simulated minute "so more frequent decisions do not inflate average health or happiness." It includes a rule-based reference agent, a versioned environment spec, and determinism tests over 1,000 random trajectories.
- **What we take:** it is very close to our current build, and it independently made the same choices (per-minute engine, a rule-based reference agent, welfare measured per minute). Our current build adds less than I implied. What would set KittenBench apart is realism and validation: real veterinary facts, cat behavior checked against data, and exploration of a physical space.

### Household simulators: VirtualHome, Habitat 3.0, BEHAVIOR-1K
- [VirtualHome](https://www.researchgate.net/publication/329750393_VirtualHome_Simulating_Household_Activities_Via_Programs) represents household activities as programs of atomic actions on objects with states. [Habitat 3.0](https://arxiv.org/abs/2310.13724) adds humanoid avatars alongside robots. [BEHAVIOR-1K](https://arxiv.org/abs/2403.09227) defines 1,000 household activities chosen by a human survey, in a physics simulator.
- **What we take:** BEHAVIOR-1K picked its tasks by surveying what people actually want help with. We should pick our events the same way, from what actually goes wrong for new kitten owners (vet visit reasons, poison-control call statistics), not from what I think is interesting. 3D physics simulators are far beyond what we need.

---

## 4. Rule-based simulations of bodies and creatures

### Pulse Physiology Engine (Kitware) — [overview](https://www.kitware.com/introducing-the-pulse-physiology-engine-open-source-computational-models-for-human-medical-simulation/), [methodology](https://pulse.kitware.com/_system_methodology.html)
- **What:** an open-source human physiology simulator used in medical training. It models organ systems as circuits and transport models, behind a common data model and an API.
- **Validation:** every system, action and condition goes through a verification-and-validation suite. Outputs are compared with published values, and combined scenarios are checked against the literature and with subject-matter experts.
- **What we take:** the standard for a believable simulated body is a V&V suite: a test for every rule, comparing the sim's output to the published number. Our current tests check a few lily timelines; they should cover every sourced rule. I found no veterinary or feline physiology engine.

### Veterinary virtual patients
- Vet schools use virtual patients for emergency training, e.g. Bristol's "Tilly" ([news](https://www.bristol.ac.uk/news/2021/september/vetpatient-simulator.html)), where students triage, stabilize and treat a simulated emergency case on their own. The [Virtual Veterinary Clinic](https://doi.org/10.3138/jvme.32.1.93) runs online small-animal case simulations.
- **What we take:** existing vet case simulations are a source of realistic emergency scenarios and correct responses, and their authors are the experts who could review ours.

### Creatures (Grand & Cliff, 1997) — [paper](https://www.researchgate.net/publication/226997131_Creatures_Entertainment_Software_Agents_with_Artificial_Life)
- **What:** the classic virtual-pet artificial-life game. Each creature has an artificial biochemistry (energy metabolism, hormones) coupled to a neural network that chooses behavior and learns during its lifetime, and both are encoded in genes.
- **What we take:** the precedent for a pet whose behavior comes out of a simulated body rather than a script. It also shows the cost: emergent behavior is hard to calibrate against real animals, and Creatures never tried.

### The Sims: needs-based AI and smart objects — [Zubek, "Needs-based AI"](https://robert.zubek.net/publications/Needs-based-AI-draft.pdf), [Bourse 2012](https://yo252yo.com/old/ens/sims-rapport.pdf)
- **What:** characters have needs; *objects* advertise which needs they satisfy and by how much ("smart objects"). A character scores every advertised action against its current needs and picks the best. The behavior logic lives in the objects, so adding a new object doesn't require changing the character.
- **What we take:** our kitten already uses needs-based selection, but with the logic hard-coded in the cat. Moving it into objects (a lily vase that advertises "chew" to a bored, impulsive kitten) is the scalable version. This is a known game-AI technique, not a research contribution.

---

## 5. How simulations are validated

### ODD protocol and pattern-oriented modeling (Grimm et al.) — [ODD 2020](https://www.jasss.org/23/2/7.html)
- **ODD** (Overview, Design concepts, Details) is the standard way to document agent-based models so they can be replicated. It lists entities, state variables, processes and scheduling, then the design concepts (sensing, adaptation, stochasticity, ...), then every detail.
- **Pattern-oriented modeling** says to choose a set of patterns observed in the real system and require the model to reproduce all of them at once. The patterns decide which state variables and processes the model needs, are used to compare alternative behavior rules, and rule out parameter values that don't reproduce them.
- **What we take:** our design doc should follow ODD, and our cat should be held to a list of real patterns rather than to my intuitions.

### Real cat behavior data (candidate patterns)
- Accelerometer studies of shelter cats: inactive about 85% of the time and active about 15%, with activity peaks around caretaker visits ([arXiv 2410.06594](https://arxiv.org/abs/2410.06594); [review](https://www.sciencedirect.com/science/article/pii/S0168159124001357)).
- Kitten development: play is most intense from 7–14 weeks, and social play peaks at 12–14 weeks ([Frontiers in Veterinary Science, 2021](https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2021.712310/full); not yet read in full).
- Growth curves, water intake, and the lily and dehydration timelines already cited in `rules.toml`.
- These are adult or shelter data; kitten-specific time budgets at 8–10 weeks are still a gap to fill.

---

## 6. What this means for KittenBench

1. **Keep the body in code.** LLMs mis-simulate exactly the kind of state a kitten's body is (Wang et al. 2024). Every sim that needs a reliable ground truth keeps it as code or data (AgentClinic, Pulse, Vending-Bench, TamaBench).
2. **Behavior is where realism is weakest, and it's unvalidated.** Our utility AI is a reasonable technique (The Sims, Humanoid Agents), but its numbers are guesses. Either calibrate it against real patterns (pattern-oriented modeling) or have an LLM choose behavior from code-computed needs (Humanoid Agents). Either way, validate with data and human raters (Generative Agents, Simile).
3. **An LLM-played cat brings measurable bias** (τ-bench critics, Lost in Simulation, AgentClinic, Pet-Bench). If used, fix it and report results under more than one cat model.
4. **Choose events from real data** on what goes wrong for kitten owners (BEHAVIOR-1K's survey approach), with responses taken from vet case simulations.
5. **Validate like Pulse:** a test per sourced rule, plus vet review of the assumptions.
6. **Long runs and many seeds** are where agent failures show up (Vending-Bench).
7. **What would be new:** TamaBench already covers "LLM keeps a virtual pet alive". The new part would be a *validated* simulated kitten (body grounded in vet sources, behavior checked against cat data and owner ratings) in an apartment the agent has to explore.
