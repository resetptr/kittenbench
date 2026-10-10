"""Run the scripted reference policies on many seeds and print a summary table.

This is the simulator's sanity check: a sensible simulator ranks oracle >= guideline > minimal > neglect.

    python run_reference.py                      # all policies, 20 seeds, 14 days
    python run_reference.py --policies guideline,minimal --seeds 5

Writes outputs/reference/results.jsonl (one row per episode) and full traces for seeds 0-2.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from kittenbench.env import KittenEnv
from kittenbench.players import POLICIES
from kittenbench.scoring import score

TABLE_COLUMNS = ["kws", "kws_worst", "survived", "nutrition", "hydration", "hygiene", "welfare", "lily", "uri",
                 "unnecessary_vet_visits", "spent", "steps"]
MULTI_CAT_COLUMNS = ["fights", "uri_caught"]   # added to the table with several kittens
TRACED_SEEDS = 3  # save full traces for this many seeds per policy


def run_episode(policy_name, seed, days, trace_dir=None, n_cats=1, newcomer_day=None):
    env = KittenEnv(seed=seed, days=days, n_cats=n_cats, trace=trace_dir is not None, newcomer_day=newcomer_day)
    obs = env.reset()
    policy = POLICIES[policy_name](env, seed)
    while not env.done:
        obs = env.step(policy.act(obs.text))
    if trace_dir:
        suffix = (f"_c{n_cats}" if n_cats > 1 else "") + (f"_n{newcomer_day}" if newcomer_day is not None else "")
        env.save_trace(Path(trace_dir) / f"{policy_name}_seed{seed}{suffix}.jsonl")
    return {"policy": policy_name, **score(env)}


def print_table(rows, policy_names, columns=TABLE_COLUMNS):
    """Mean of each column per policy. Components that don't apply to a seed (e.g. no lily event) are skipped."""
    print(f"\n{'policy':<11}" + "".join(f"{c[:10]:>11}" for c in columns))
    for name in policy_names:
        policy_rows = [r for r in rows if r["policy"] == name]
        cells = []
        for col in columns:
            values = [float(r[col]) for r in policy_rows if r.get(col) is not None]
            cells.append(f"{statistics.mean(values):>11.2f}" if values else f"{'-':>11}")
        print(f"{name:<11}" + "".join(cells))


def print_deaths(rows):
    causes = {}
    for r in rows:
        if not r["survived"]:
            causes.setdefault(r["policy"], []).append(r["cause_of_death"])
    for policy, cs in causes.items():
        print(f"deaths[{policy}]: " + ", ".join(sorted(f"{c} x{cs.count(c)}" for c in set(cs))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--cats", type=int, default=1, help="number of kittens")
    ap.add_argument("--newcomer-day", type=int, default=None, help="with --cats 2+, the last kitten is a newcomer a friend brings on this day")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--out", default="outputs/reference")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy_names = args.policies.split(",")
    rows = []
    for name in policy_names:
        start = time.time()
        for seed in range(args.seeds):
            trace_dir = out if seed < TRACED_SEEDS else None
            rows.append(run_episode(name, seed, args.days, trace_dir, args.cats, args.newcomer_day))
        print(f"{name}: {time.time() - start:.1f}s", flush=True)

    with open(out / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print_table(rows, policy_names, TABLE_COLUMNS + (MULTI_CAT_COLUMNS if args.cats > 1 else []))
    print_deaths(rows)


if __name__ == "__main__":
    main()
