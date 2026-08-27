#!/usr/bin/env python3
"""
Entry point for the Autonomous ML Research Agent on KuaiRand-Pure.

    # 1. reproduce the official baseline (Task Requirement #1)
    python3 run.py baseline

    # 2. run the autonomous research loop (train + validation only)
    python3 run.py loop --max-iterations 20

    # 3. produce the final submission (the ONLY step that touches test)
    python3 run.py submit --out submission.csv

    # verify the contracts at any time
    python3 run.py test

Data defaults to ./KuaiRand/KuaiRand-Pure/data; override with --data_dir.
Requires OPENAI_API_KEY for `loop`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "KuaiRand" / "KuaiRand-Pure" / "data"


def _check_data(data_dir: Path) -> None:
    required = [
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "video_features_basic_pure.csv",
    ]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        sys.exit(
            f"Missing KuaiRand-Pure files in {data_dir}:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nDownload (no registration required):\n"
            "  wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz\n"
            "  tar xzf KuaiRand-Pure.tar.gz"
        )


def cmd_baseline(a) -> int:
    """Reproduce the organisers' reference pipeline, unmodified."""
    return subprocess.call(
        [sys.executable, "baseline.py", "--data_dir", str(a.data_dir),
         "--model", a.model, "--seed", str(a.seed)],
        cwd=str(ROOT / "official"),
    )


def cmd_loop(a) -> int:
    """Run the autonomous research loop. Never touches the test split."""
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")
    from agent.controller import run_loop

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    summary = run_loop(
        data_dir=str(a.data_dir),
        max_iterations=a.max_iterations,
        wall_clock_seconds=a.wall_clock,
        token_budget=a.token_budget,
        llm_model=a.model,
        seed=a.seed,
        workspace=a.workspace,
        pipeline=a.pipeline,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_submit(a) -> int:
    """Train the final checkpoint and write the submission CSV."""
    return subprocess.call(
        [sys.executable, "make_submission.py", "--data_dir", str(a.data_dir),
         "--out", a.out, "--seed", str(a.seed), "--config", a.config]
        + (["--hold-test-score"] if a.hold_test_score else []),
        cwd=str(ROOT),
    )


def cmd_test(a) -> int:
    """Run the contract tests."""
    return subprocess.call(
        [sys.executable, "tests/test_contract.py", "--data_dir", str(a.data_dir)],
        cwd=str(ROOT),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("baseline", help="reproduce the official baseline")
    p.add_argument("--model", default="fm", choices=["fm", "pop", "random"])
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("loop", help="run the autonomous research loop")
    p.add_argument("--max-iterations", type=int, default=20)
    p.add_argument("--wall-clock", type=int, default=5400, help="seconds")
    p.add_argument("--token-budget", type=int, default=300_000)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workspace", default="workspace")
    p.add_argument("--pipeline", default="classify",
                   choices=["classify", "library"])
    p.set_defaults(fn=cmd_loop)

    p = sub.add_parser("submit", help="write the final submission (touches test)")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="{}")
    p.add_argument("--hold-test-score", action="store_true",
                   help="write the CSV without computing the test metric")
    p.set_defaults(fn=cmd_submit)

    p = sub.add_parser("test", help="run the contract tests")
    p.set_defaults(fn=cmd_test)

    a = ap.parse_args()
    _check_data(a.data_dir)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
