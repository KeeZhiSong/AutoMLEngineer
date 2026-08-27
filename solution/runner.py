"""
runner.py -- FROZEN. The agent may not edit this file.

The seam the Executor invokes. Wires the three swappable modules together in a
fixed order and returns a machine-readable result:

    dataset.load  ->  features  ->  model  ->  train  ->  scoring
      (frozen)       (agent)     (agent)   (agent)     (frozen)

Keeping the wiring frozen while the modules inside it are rewritten means a bad
edit breaks ONE module instead of the whole run.

THE TEST SPLIT IS NOT AVAILABLE TO THE LOOP. run() scores validation only unless
evaluate_test=True is passed explicitly. The rules require developing on
train + validation alone; a test score leaking into the agent's decisions would
turn the held-out split into just another validation set and make its estimate of
the hidden-test result meaningless. Only the final checkpoint sets that flag.

run() never raises. A crash is a RESULT (status='bug' plus the traceback), because
the loop must be able to log a failure and carry on rather than dying with it.
"""

from __future__ import annotations

import importlib
import signal
import time
import traceback
from pathlib import Path

import numpy as np

from . import dataset
from .scoring import delta_vs_baseline, score

_CACHE: dict = {}


def _load_cached(data_dir: str):
    """Parsing 1.4M CSV rows takes ~20s; the loop runs this many times."""
    key = str(Path(data_dir).resolve())
    if key not in _CACHE:
        _CACHE[key] = dataset.load(key)
    return _CACHE[key]


def _attach_aux(enc, split):
    """Attach auxiliary TARGETS and raw ids to an Encoded, here in the frozen
    layer rather than in features.py.

    Multi-task, funnel and watch-time objectives all need the other feedback
    signals, but fit() only ever receives Encoded objects -- so without this
    those directions are impossible to express. Doing it here keeps features.py
    unable to read a feedback column at all, so the leak guard stays intact:
    the signals are reachable as targets in train.py, never as inputs.
    """
    from .dataset import FEEDBACK_COLUMNS
    enc.aux = {c: split.feedback(c) for c in FEEDBACK_COLUMNS}
    enc.aux["duration_ms"] = split.log["duration_ms"].to_numpy()
    enc.aux["date"] = split.log["date"].to_numpy()
    enc.aux["hourmin"] = split.log["hourmin"].to_numpy()
    enc.video_id = split.video_id
    return enc


class _CycleTimeout(Exception):
    pass


def _alarm(signum, frame):                      # noqa: ARG001
    raise _CycleTimeout()


def run(data_dir: str,
        config: dict | None = None,
        seed: int = 0,
        evaluate_test: bool = False,
        verbose: bool = False,
        timeout_seconds: int = 900) -> dict:
    """Execute one full pipeline. Returns a dict shaped for ledger.record().

    timeout_seconds caps ONE experiment. Enforced HERE, in the frozen layer, via
    SIGALRM -- the agent rewrites train.py every cycle and could remove a check
    placed there.

    Why it exists: one cycle used eval_sized_groups() (240,024 groups of ~5 rows)
    directly as optimiser BATCHES instead of packing them, giving 240k Adam steps
    per epoch over the full embedding matrix instead of 140. It ran 5,948s --
    98% of the entire run's compute -- and starved the remaining 18 cycles, which
    exhausted the wall-clock budget with 86% of the token budget unspent.
    A slow experiment is not a wrong answer, but it is a wrong PRICE.
    """
    started = time.time()
    out: dict = {"status": "ok", "error": "", "metrics": {}, "test_metrics": {},
                 "wall_seconds": 0.0, "audit": {}}
    _prev = None
    try:
        _prev = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(timeout_seconds))
        cfg = {**(config or {}), "seed": seed}

        # Reimported every run so the Coder's edits take effect in-process.
        features = importlib.reload(importlib.import_module("solution.features"))
        model_mod = importlib.reload(importlib.import_module("solution.model"))
        train_mod = importlib.reload(importlib.import_module("solution.train"))

        ds = _load_cached(data_dir)
        out["rows"] = {k: ds[k].n for k in ("train", "valid", "test")}

        state = features.fit(ds.train)
        train_enc = _attach_aux(features.transform(ds.train, state), ds.train)
        valid_enc = _attach_aux(features.transform(ds.valid, state), ds.valid)

        mdl = model_mod.build(state, cfg)
        log = print if verbose else (lambda *a, **k: None)
        mdl, history = train_mod.fit(mdl, train_enc, valid_enc, cfg, log=log)

        preds = mdl.predict(valid_enc.X)

        # ROW-COUNT INVARIANT. The scored set must be EXACTLY the evaluation
        # split -- same rows, same count, same order. An agent that pads,
        # augments, filters or reweights rows inside transform() is scoring a
        # different and usually easier problem, and the number it reports is
        # meaningless.
        #
        # Observed: a "synthetic augmentation" patch padded every user's list to
        # a fixed length with UNK rows carrying label 0, turning 124,909 scored
        # rows into 451,647. GAUC went 0.669 -> 0.890 because ranking real
        # impressions above synthetic filler is trivial. It was KEPT.
        # The leak guard cannot see this: no feedback column is touched.
        if len(preds) != ds.valid.n or len(valid_enc.y) != ds.valid.n:
            raise ValueError(
                f"row-count violation: the validation split has {ds.valid.n:,} "
                f"rows but features.transform() produced {len(valid_enc.y):,} "
                f"labels and {len(preds):,} scores. transform() must return one "
                f"row per input row, in the same order. Padding, augmenting, "
                f"filtering or de-duplicating rows changes what is being scored."
            )
        if not np.all(np.array_equal(valid_enc.user_id, ds.valid.user_id)):
            raise ValueError(
                "row-order violation: valid_enc.user_id does not match the "
                "split's user_id sequence. transform() must preserve row order."
            )

        if not np.all(np.isfinite(preds)):
            raise ValueError(
                f"{(~np.isfinite(preds)).sum()} non-finite validation scores; "
                f"a submission with NaN/Inf is rejected."
            )

        out["metrics"] = score(valid_enc.user_id, valid_enc.y, preds)
        out["delta"] = delta_vs_baseline(out["metrics"], "valid")
        out["epochs_run"] = len(history)
        out["history"] = history
        out["audit"] = {"features": features.audit(state),
                        "model": model_mod.audit(mdl)}

        if evaluate_test:
            test_enc = _attach_aux(features.transform(ds.test, state), ds.test)
            test_preds = mdl.predict(test_enc.X)
            if len(test_preds) != ds.test.n:
                raise ValueError(
                    f"row-count violation on test: expected {ds.test.n:,} rows, "
                    f"got {len(test_preds):,}. The submission would be rejected.")
            out["test_metrics"] = score(test_enc.user_id, test_enc.y, test_preds)
            out["test_delta"] = delta_vs_baseline(out["test_metrics"], "test")
            out["test_scores"] = test_preds

    except _CycleTimeout:
        out["status"] = "timeout"
        out["error"] = (
            f"cycle exceeded {timeout_seconds}s. The reference pipeline trains in "
            f"~15s, so this is ~{timeout_seconds // 15}x the baseline cost. The "
            f"usual cause is using loss GROUPS as optimiser BATCHES: "
            f"eval_sized_groups() returns ~240,000 groups of ~5 rows, and "
            f"stepping the optimiser once per group is ~1,700x more updates than "
            f"the 140 batches user_batches() produces. Group for the LOSS; batch "
            f"for the OPTIMISER.")
    except Exception as exc:                    # noqa: BLE001 -- a crash is a result
        out["status"] = "bug"
        out["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
    finally:
        signal.alarm(0)
        if _prev is not None:
            signal.signal(signal.SIGALRM, _prev)
        out["wall_seconds"] = round(time.time() - started, 2)
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Run one end-to-end pipeline.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--evaluate-test", action="store_true",
                    help="Score the held-out split. FINAL CHECKPOINT ONLY.")
    a = ap.parse_args()

    res = run(a.data_dir, config={"epochs": a.epochs}, seed=a.seed,
              evaluate_test=a.evaluate_test, verbose=True)
    res.pop("history", None)
    res.pop("test_scores", None)
    print(json.dumps(res, indent=2, default=float))
