"""Play KittenBench yourself in the terminal, as the owner.

    python play.py --seed 3
    > go_to kitchen
    > use fill food_bowl item=kibble amount=30
    > examine litter_box
    > help

Commands are a tool name followed by arguments, either positional (in the order the tool lists them) or
key=value. Your score is printed when the episode ends.
"""

import argparse
import ast
import re
import shlex

from kittenbench.env import KittenEnv
from kittenbench.scoring import score
from kittenbench.players import TOOLS

SPEC = {t["name"]: t for t in TOOLS}


def parse(line):
    """Turn a typed command into a tool call. Both styles work:

        use fill food_bowl item=kibble amount=30
        use(verb="fill", object="food_bowl", item="kibble", amount=30)
    """
    name = re.match(r"\s*(\w+)", line).group(1)
    schema = SPEC[name]["input_schema"]
    # Positional arguments fill the required parameters first, then the optional ones.
    order = list(schema["required"]) + [k for k in schema["properties"] if k not in schema["required"]]
    if "(" in line:
        call = ast.parse(line.strip(), mode="eval").body
        args = {order[i]: ast.literal_eval(a) for i, a in enumerate(call.args)}
        args.update({kw.arg: ast.literal_eval(kw.value) for kw in call.keywords})
        return {"name": name, "arguments": args}
    args = {}
    for i, part in enumerate(shlex.split(line)[1:]):
        key, value = part.split("=", 1) if "=" in part else (order[i], part)
        args[key] = int(value) if value.isdigit() else value
    return {"name": name, "arguments": args}


def print_help():
    for t in TOOLS:
        params = " ".join(t["input_schema"]["properties"])
        print(f"  {t['name']} {params}  — {t['description']}")
    print("""
Examples:
  go_to kitchen
  examine food_bowl                              (shows what you can do with it)
  use fill food_bowl item=kibble amount=30
  use fill water_bowl
  use scoop litter_box
  inspect_cat eyes
  wait 60
  use(verb="take", object="toy_mouse")           (function-call style works too)
Object names are the ones in [brackets] in what you see. Ctrl-C or Ctrl-D quits and shows your score.""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--cats", type=int, default=1, help="number of kittens")
    ap.add_argument("--newcomer-day", type=int, default=None, help="with --cats 2+, the last kitten is a newcomer a friend brings on this day")
    args = ap.parse_args()

    env = KittenEnv(seed=args.seed, days=args.days, n_cats=args.cats, newcomer_day=args.newcomer_day)
    print(env.reset().text)
    while not env.done:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):   # Ctrl-D / Ctrl-C: stop and show the score so far
            print("\nStopped.")
            break
        if not line:
            continue
        if line in ("help", "?"):
            print_help()
            continue
        if re.match(r"\s*(\w+)", line).group(1) not in SPEC:
            print("Unknown action. Type 'help'.")
            continue
        try:
            call = parse(line)
        except (IndexError, ValueError, SyntaxError) as e:
            print(f"Couldn't parse: {e}")
            continue
        print(env.step(call).text)
    print("\n", score(env))


if __name__ == "__main__":
    main()
