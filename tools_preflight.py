#!/usr/bin/env python3
"""
Pre-flight: assert the agent-editable modules are in a KNOWN state, by SCORE.

Twice today a run launched from the wrong code because a reset silently
no-opped, and twice a string-matching check gave the wrong answer -- once by
missing the change (`str.replace` returns the input unchanged when the anchor is
absent) and once by false-alarming on the word "listwise" appearing in a
docstring.

Source text is the wrong thing to check. What matters is what the code DOES.
This runs the pipeline and compares the score against the expected value.

    python3 tools_preflight.py --expect reference   # 0.6014
    python3 tools_preflight.py --expect winning     # 0.6034

Exits non-zero on mismatch, so it can gate a launch with &&.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

EXPECTED = {
    "reference": 0.6014,   # pointwise logloss, row batching -- the organisers' FM
    "winning":   0.6042,   # listwise + group_size=5 + lr 2e-4, seed 0 (human)
    "agent":     0.6024,   # weighted_logloss, found unaided (5-seed mean 0.6031)
}
TOL = 0.0005               # tighter than one seed std; same seed must reproduce


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", required=True, choices=sorted(EXPECTED))
    ap.add_argument("--data_dir", default=str(ROOT / "KuaiRand/KuaiRand-Pure/data"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    lock = ROOT / ".loop_running.lock"
    if lock.exists():
        print("PRE-FLIGHT REFUSED — a research loop is live. Running a pipeline\n"
              "now would swap modules underneath it and corrupt its experiment.",
              file=sys.stderr)
        return 1

    from solution import runner
    from solution.leakcheck import find_leaks

    for m in ("features.py", "model.py"):
        leaks = find_leaks((ROOT / "solution" / m).read_text(), m)
        if leaks:
            print(f"PRE-FLIGHT FAILED — {m} leaks:\n  " + "\n  ".join(leaks),
                  file=sys.stderr)
            return 1

    res = runner.run(a.data_dir, config={"seed": a.seed}, seed=a.seed)
    if res["status"] != "ok":
        print(f"PRE-FLIGHT FAILED — pipeline does not run:\n"
              f"{res['error'].splitlines()[0]}", file=sys.stderr)
        return 1

    got, want = res["metrics"]["primary"], EXPECTED[a.expect]
    if abs(got - want) > TOL:
        print(f"PRE-FLIGHT FAILED — expected the '{a.expect}' configuration "
              f"({want:.4f}) but the code on disk scores {got:.4f}.\n"
              f"The modules are not in the state you think they are.",
              file=sys.stderr)
        return 1

    print(f"PRE-FLIGHT OK — '{a.expect}' confirmed by score: {got:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
