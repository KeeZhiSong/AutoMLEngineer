"""
Produce the final submission CSV for the hidden-test split.

THIS IS THE ONLY SCRIPT THAT MAY TOUCH THE TEST SPLIT. The research loop runs on
train + validation alone; scoring test during the search would turn the held-out
split into another validation set. Run this once, at the end, on the checkpoint
the loop designated as best.

    python3 make_submission.py --data_dir <KuaiRand-Pure/data> --out submission.csv

Writes the schema pinned by the starter kit:

    row_id,user_id,video_id,score

`row_id` is a 0-based index into the evaluation split in official row order --
required because (user_id, video_id) is NOT unique (3.06% of test rows are
repeated pairs, up to 12 times). solution/dataset.py asserts our row order
matches official.data.load(), and this script re-asserts it before writing.

Validate the result with the organisers' own checker:

    cd official && python3 submit.py --check --split test ../submission.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from solution import dataset, runner                      # noqa: E402
from solution.scoring import BASELINE, delta_vs_baseline  # noqa: E402

HEADER = ["row_id", "user_id", "video_id", "score"]


def write_submission(path: Path, split, scores: np.ndarray) -> int:
    """Write the CSV, refusing anything the official checker would reject."""
    if len(scores) != split.n:
        raise ValueError(f"{len(scores)} scores for {split.n} rows")
    if not np.all(np.isfinite(scores)):
        bad = int((~np.isfinite(scores)).sum())
        raise ValueError(f"{bad} non-finite scores; NaN/Inf are rejected")

    users, videos = split.user_id, split.video_id
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i in range(split.n):
            w.writerow([i, users[i], videos[i], f"{float(scores[i]):.6g}"])
    return split.n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold-test-score", action="store_true",
                    help="Write the CSV but do NOT compute or print the test "
                         "metric. Use for an interim/insurance submission: the "
                         "file needs test PREDICTIONS, but seeing the test "
                         "SCORE before the config is final turns the held-out "
                         "split into another validation set.")
    ap.add_argument("--config", default="{}",
                    help="JSON training config for the final checkpoint, e.g. "
                         '\'{"batch_mode":"user","lr":0.005}\'')
    a = ap.parse_args()

    cfg = {**json.loads(a.config), "seed": a.seed}
    print(f"training final checkpoint (config={cfg}) ...")

    # evaluate_test=True is set HERE and nowhere else in the project.
    res = runner.run(a.data_dir, config=cfg, seed=a.seed,
                     evaluate_test=True, verbose=True)
    if res["status"] != "ok":
        print("FAILED:\n" + res["error"], file=sys.stderr)
        return 1

    ds = runner._load_cached(a.data_dir)
    counts = dataset.verify_alignment(ds, a.data_dir)   # guard row_id validity
    print(f"row order verified against official loader: {counts}")

    out = Path(a.out)
    n = write_submission(out, ds.test, res["test_scores"])

    v, dv = res["metrics"], res["delta"]
    print(f"\nwrote {out} ({n:,} rows)")
    if a.hold_test_score:
        print(f"{'':14s} {'GAUC':>8s} {'nDCG@5':>8s} {'primary':>8s}")
        print(f"{'valid':14s} {v['GAUC']:8.4f} {v['nDCG@5']:8.4f} {v['primary']:8.4f}")
        print(f"{'  baseline':14s} {BASELINE['valid']['GAUC']:8.4f} "
              f"{BASELINE['valid']['nDCG@5']:8.4f} {BASELINE['valid']['primary']:8.4f}")
        print(f"{'  delta':14s} {dv['GAUC']:+8.4f} {dv['nDCG@5']:+8.4f} {dv['primary']:+8.4f}")
        print("\ntest score WITHHELD (--hold-test-score). The CSV contains test")
        print("predictions; the test METRIC has not been computed or shown, so the")
        print("held-out split is still clean for a single final scoring.")
        print(f"\nvalidate with:\n  cd official && python3 submit.py --check "
              f"--split test {out.resolve()}")
        return 0

    t, dt = res["test_metrics"], res["test_delta"]
    print(f"{'':14s} {'GAUC':>8s} {'nDCG@5':>8s} {'primary':>8s}")
    print(f"{'valid':14s} {v['GAUC']:8.4f} {v['nDCG@5']:8.4f} {v['primary']:8.4f}")
    print(f"{'  baseline':14s} {BASELINE['valid']['GAUC']:8.4f} "
          f"{BASELINE['valid']['nDCG@5']:8.4f} {BASELINE['valid']['primary']:8.4f}")
    print(f"{'  delta':14s} {dv['GAUC']:+8.4f} {dv['nDCG@5']:+8.4f} {dv['primary']:+8.4f}")
    print(f"{'test':14s} {t['GAUC']:8.4f} {t['nDCG@5']:8.4f} {t['primary']:8.4f}")
    print(f"{'  baseline':14s} {BASELINE['test']['GAUC']:8.4f} "
          f"{BASELINE['test']['nDCG@5']:8.4f} {BASELINE['test']['primary']:8.4f}")
    print(f"{'  delta':14s} {dt['GAUC']:+8.4f} {dt['nDCG@5']:+8.4f} {dt['primary']:+8.4f}")
    print(f"\nscore_dataset (mean of GAUC and nDCG@5 deltas on test): "
          f"{dt['score_dataset']:+.4f}")
    print(f"\nvalidate with:\n  cd official && python3 submit.py --check "
          f"--split test {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
