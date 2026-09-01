#!/usr/bin/env python3
"""
Score the submitted CSV on the hidden test split. Run ONCE, after the
configuration is final.

    python3 tools_score_test.py                     # scores ./submission.csv

This is the organisers' own procedure, reproduced exactly: the alignment-checking
reader from official/submit.py, then official/evaluate.py unmodified. Nothing of
ours takes part in the arithmetic.

The test split is deliberately unreachable everywhere else in this repository:
solution/runner.py is never called with evaluate_test during the loop, so every
keep/revert decision, every exploit sweep and the 20-seed confirmation ran on
validation alone. Running this script does not feed back into anything -- it
reads a CSV that has already been written and reports a number.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official"))

# Published hidden-test scores for the official FM baseline (Starter Kit).
BASELINE = {"GAUC": 0.6610, "nDCG@5": 0.5282, "primary": 0.5946}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(ROOT / "submission.csv"))
    ap.add_argument("--data_dir", default=str(ROOT / "KuaiRand/KuaiRand-Pure/data"))
    a = ap.parse_args()

    from data import load                       # noqa: E402
    from evaluate import evaluate               # noqa: E402
    from submit import read_submission          # noqa: E402

    rows = load(a.data_dir)["test"]             # (date, user, video, author, tab, dur, label)
    scores = read_submission(a.path, rows)      # raises on any misalignment
    m = evaluate([r[1] for r in rows], [r[6] for r in rows], scores)

    print(f"{Path(a.path).name}: {len(rows):,} test rows\n")
    print(f"{'metric':10s} {'ours':>8s} {'baseline':>9s} {'delta':>9s}")
    for k in ("GAUC", "nDCG@5", "primary"):
        print(f"{k:10s} {m[k]:8.4f} {BASELINE[k]:9.4f} {m[k] - BASELINE[k]:+9.4f}")

    delta = ((m["GAUC"] - BASELINE["GAUC"])
             + (m["nDCG@5"] - BASELINE["nDCG@5"])) / 2
    print(f"\nscore_dataset = mean(delta_GAUC, delta_nDCG@5) = {delta:+.4f}")
    print(f"beats the official baseline: {m['primary'] > BASELINE['primary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
