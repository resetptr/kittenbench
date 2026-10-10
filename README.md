# KittenBench

An LLM raises kittens in a simulated apartment. It plays the owner through tool calls (look around, fill a bowl,
scoop the litter box, take a kitten to the vet...); the simulator advances time, the kittens live their lives, and
the run is scored on how well every kitten was cared for.

The kittens are not LLMs: they're a small game engine (needs that change over time, Sims-style activity choice, an
event queue, a tile map), so runs are cheap and repeatable. Every number about cat health comes from a cited vet
source or is flagged as an assumption (`kittenbench/data/rules.toml`). See `DESIGN.md` for the design and
`RELATED_WORK.md` for prior work.

## Results

One scenario (seed 0): **6 kittens for 7 days**, one of them a newcomer a friend drops off on day 2. The owner
works weekdays 08:30-18:00. The scenario includes a cat flu outbreak and kittens that don't get along. Score is 0-1
(1 = perfect care); "worst-off" is the lowest-scoring kitten.

| Model | Score | Worst-off | Nutrition | Hygiene | Welfare | Flu care | Accidents | Fights | API cost |
|---|---|---|---|---|---|---|---|---|---|
| qwen/qwen3.8-2.4t-a95b | **0.780** | 0.644 | 0.97 | 0.70 | 0.66 | 0.52 | 13 | 12 | $3.72 |
| openai/gpt-6-sol | 0.768 | **0.707** | 0.96 | 0.97 | 0.59 | 0.32 | 0 | 8 | $0.77 |
| anthropic/claude-sonnet-5.5 | 0.717 | 0.617 | 0.97 | 0.64 | 0.70 | 0.15 | 14 | 2 | $1.82 |
| *Checklist script (vet guidelines)* | *0.905* | *0.863* | | | | | | | |
| *Feeds once a day, nothing else* | *0.074* | *0.0* | | | | | | | *5 of 6 kittens die* |

One run per model, so the scores are close enough to call a tie; the care styles differ much more (see the logs).
Each run's files are in `results/<model>/`:

- `*.agent.jsonl`: everything the model saw and did, turn by turn (open in `viewer/log.html`)
- `*.result.json`: the score breakdown per kitten, tokens and cost
- `*.trace.jsonl`: everything that happened in the apartment, minute by minute

Reproduce a run (needs `OPENROUTER_API_KEY`):

```sh
uv run --no-project --with openai python run_agent.py --model openai/gpt-6-sol --seed 0 --days 7 --cats 6 --newcomer-day 2 --max-cost 5
```

## Layout

```
kittenbench/          the simulator (a Python package)
  data/               rules.toml (vet-sourced numbers), kitten.toml (behavior), apartment map
  sim.py              time, needs, event queue, apartment and pathfinding
  cat.py              illnesses, activities, the kitten itself
  env.py              the game the agent plays: text, actions, vet visits, the engine loop
  scoring.py          what's measured and the score
  players.py          the tool list given to LLMs, and scripted reference players
run_agent.py          run an LLM through the benchmark (OpenRouter)
run_reference.py      run the scripted players (free sanity check)
play.py               play it yourself in the terminal
viewer/log.html       turn-by-turn log viewer (open in a browser)
results/              the runs above
tests/                tests
```

## Run an LLM through the benchmark

Needs `OPENROUTER_API_KEY` set in your shell. Python 3.11+.

```sh
uv run --no-project --with openai python run_agent.py --model qwen/qwen3.5-9b --seed 0 --days 2
```

- `--model`: any OpenRouter model id with tool support
- `--seed`: which scenario (kitten personalities and which events happen when); same seed = same setup
- `--days`: how many simulated days (default 2)
- `--cats`: how many kittens share the apartment (default 1, up to 6). With more than one, kitten actions take
  `cat="<name>"`, and the score reports the mean kitten (`kws`) and the worst-off kitten (`kws_worst`).
  The kittens share bowls and litter boxes (more can be bought), get along or don't (play, groom, sleep together,
  or hiss, fight and guard the food bowl), and can pass cat flu to each other
- `--newcomer-day`: with `--cats 2` or more, the last kitten isn't there at the start; a friend brings it on this
  day, and it starts out wary of the others
- `--max-cost`: stop once the run has cost this many dollars (default 2.00); results are still saved
- `--max-steps`: cap on actions (default 3000)
- `--out`: output folder (default `outputs/pilot`)

Prompt caching is on (Anthropic automatically, Qwen/DeepSeek with cache markers, OpenAI by itself); every 10 turns
the progress line shows what share of prompt tokens were read from the cache. Raise `--max-cost` for long runs.

Each model call is one **turn**: at least 15 minutes of game time, in which the model can call up to 8 tools
(run in order). If its actions finish early, the rest of the 15 minutes passes on its own.

It prints one line per turn, and where it saved everything at the end:
`<model>_seed<N>_d<days>[_c<cats>][_n<newcomer day>].agent.jsonl`, `.result.json` and `.trace.jsonl`.
If the `.result.json` already exists the run is skipped; delete it to rerun.

## Look at a run

- **Text log:** open `viewer/log.html` in a browser, click **Open run folder**, pick e.g. `results/gpt-6-sol`.
  It shows every turn: the exact context sent, the model's reasoning, each action and what it returned.

## Play it yourself

```sh
python3 play.py --seed 3 --days 2            # add --cats 2 for two kittens, --newcomer-day 2 for a newcomer
```

Type actions at the `>` prompt (`go_to kitchen`, `use fill food_bowl item=kibble amount=30`, `wait 60`, `help`).
Ctrl-C quits and shows your score.

## Sanity-check the simulator (no LLM, free)

```sh
python3 run_reference.py            # scripted players on 20 seeds x 14 days, prints a score table (takes --cats, --newcomer-day)
uvx --python 3.13 pytest -q         # tests
```
