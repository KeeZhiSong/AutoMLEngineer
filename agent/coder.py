"""
Coder: turn a hypothesis into a code patch on exactly ONE module.

Two guards, both learned from a previous run in which every cycle died on
`KeyError: 'engagement'` -- a column the model invented:

  1. The prompt carries the real interface (dataclass fields, array shapes,
     column names) rather than a prose description of it.
  2. The result is syntax-checked AND import-checked before being accepted. A
     module that parses but cannot import is still broken, and catching it here
     costs a second instead of a full training run.

On any failure the previous file is restored, so a bad generation cannot leave
the tree in a broken state.
"""

from __future__ import annotations

import ast
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from lib.llm import complete

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solution.leakcheck import assert_clean  # noqa: E402

logger = logging.getLogger("amra.coder")

ROOT = Path(__file__).resolve().parent.parent
SOLUTION = ROOT / "solution"
EDITABLE_MODULES = ("features.py", "model.py", "train.py")

INTERFACE = '''
## The interface your code must honour

`solution/dataset.py` (FROZEN, do not edit) gives you a `Split`:
    split.n            -> int, number of rows
    split.log          -> pandas DataFrame, ALL raw log columns, official row order
    split.video        -> pandas DataFrame of video-side features
    split.user_id      -> np.ndarray (N,)  int64
    split.video_id     -> np.ndarray (N,)  int64
    split.label        -> np.ndarray (N,)  float32, the scored `long_view` label
    split.feedback(c)  -> np.ndarray (N,), c in: is_click, is_like, is_follow,
                          is_comment, is_forward, is_hate, is_profile_enter,
                          play_time_ms, profile_stay_time, comment_stay_time

`split.log` columns, exactly these and no others:
    user_id, video_id, date, hourmin, time_ms, is_click, is_like, is_follow,
    is_comment, is_forward, is_hate, long_view, play_time_ms, duration_ms,
    profile_stay_time, comment_stay_time, is_profile_enter, is_rand, tab,
    author_id

`solution/features.py` produces:
    Encoded(X: np.ndarray (N, F) int32,   # field indices, offsets applied
            y: np.ndarray (N,) float32,   # the long_view label
            user_id: np.ndarray (N,),     # for within-user grouping
            aux: dict[str, np.ndarray],   # auxiliary TARGETS -- see below
            video_id: np.ndarray (N,))    # raw ids, for item-side priors

    `aux` is attached by the frozen runner, NOT by features.py, and holds:
      is_click, is_like, is_follow, is_comment, is_forward, is_hate,
      is_profile_enter, play_time_ms, profile_stay_time, comment_stay_time,
      duration_ms, date, hourmin
    These are TARGETS for multi-task / funnel / watch-time objectives in
    train.py. Putting one into X leaks the label.

    NOTE what is NOT on Encoded: there is no `.feedback()` method and no
    `.log` frame. Split has those; Encoded does not. train.py receives Encoded
    objects only -- use `enc.aux[...]` and `enc.video_id`.
    FeatureState(vocabs, unk, offsets, dim, duration_edges, fields)
    fit(train: Split) -> FeatureState
    transform(split: Split, state: FeatureState) -> Encoded
    audit(state) -> dict

`solution/model.py` produces:
    FM(V: (dim,k) float32, W: (dim,) float32, b: float32, k: int)
    FM.logits(X)      -> (scores (B,), E (B,F,k), S (B,k))
    FM.predict(X)     -> (N,) float scores
    FM.grad(X, dz)    -> (gV, gW, gb)   # dz is dLoss/dlogit, one per row
    FM.snapshot() / FM.restore(snap)
    build(state: FeatureState, config: dict) -> FM
    audit(model) -> dict

`solution/train.py` produces:
    fit(model, train_enc: Encoded, valid_enc: Encoded, config: dict,
        loss_fn=..., log=print) -> (model, history: list[dict])
    group_by_user(user_id)          -> list[np.ndarray]  # per-user row indices
    group_segments(user_id)         -> (order, starts, lengths)   VECTORISED
    segment_softmax(scores, starts, lengths) -> per-group softmax  VECTORISED
    segment_sum(values, starts)     -> per-group sums              VECTORISED
    user_batches(user_id, bs, rng, group_size=None) -> list of BATCHES
    eval_sized_groups(user_id, rng, size) -> list of LOSS GROUPS

  GROUPS are not BATCHES. A loss group is the set of rows a within-user
  objective compares against each other (~5 rows). A batch is the set of rows
  between optimiser steps (~8192 rows). `user_batches` PACKS groups into
  batches. Passing eval_sized_groups() straight to the batch loop gives ~240,000
  optimiser steps per epoch instead of ~140 -- an experiment that did this ran
  5,948s instead of 15s and was killed by the cycle timeout.

  Prefer the VECTORISED segment helpers over `for g in group_by_user(...)`:
  a Python loop runs once per user (~26,000 iterations) inside every batch.

    fit() receives Encoded objects ONLY -- never a Split. Auxiliary signals are
    reachable as `train_enc.aux["is_click"]` etc., aligned row-for-row with X.

    A loss function has signature (model, X, y) -> (dz, loss) where
    `dz` is dLoss/dlogit as a (B,) float32 array, ONE VALUE PER ROW in batch
    order, and `loss` is a float. model.grad(X, dz) consumes exactly that.

    For a WITHIN-USER ranking loss, use group_by_user and scatter back by index:

        def my_ranking_loss(model, X, y, user_id):
            z, _, _ = model.logits(X)
            dz = np.zeros(len(y), dtype=np.float32)
            total = 0.0
            for g in group_by_user(user_id):
                s, yg = z[g], y[g]
                if yg.sum() == 0 or yg.sum() == len(yg):
                    continue                 # no ordering signal in this group
                ...                          # compute this group's gradient
                dz[g] = <per-row gradient for these rows>
            return dz / len(y), total / len(y)

    NOTE: a grouped loss needs `user_id` per batch. fit() slices train_enc.X and
    train_enc.y by `idx`; slice train_enc.user_id by the same `idx` and pass it
    through. Changing fit()'s internals to do that is expected and allowed.

`solution/scoring.py` (FROZEN):
    score(user_ids, labels, scores) -> {"GAUC":…, "nDCG@5":…, "primary":…}
    Never compute a metric yourself. Import this.

## Correctness notes (general, not dataset-specific)
- A listwise softmax target must be a PROPER DISTRIBUTION over the group: with
  multiple positives use `t = y_g / y_g.sum()`, never raw 0/1. Raw targets ask
  the softmax to sum to the positive count, which is unreachable.
- Config keys `fit()` reads: lr, l2, batch, epochs, patience, seed, batch_mode
  ("row"|"user"), group_size (int|None). `batch_mode="user"` keeps each user's
  impressions in one batch; `group_size` additionally caps list length.
- A grouped loss has a different gradient scale from pointwise logloss, so set
  `lr` deliberately rather than inheriting a default tuned for another objective.

## Hard rules
1. Reference ONLY names listed above. Inventing a column or attribute is the
   single most common way these patches fail.
2. Keep every public signature identical -- the frozen runner calls them.
3. Learn statistics on TRAIN only, apply them in transform(). A statistic
   computed over valid or test is leakage.
4. Feedback columns are TARGETS, never inputs, and this is ENFORCED: features.py
   and model.py are rejected outright if they reference is_click, is_like,
   is_follow, is_comment, is_forward, is_hate, is_profile_enter, play_time_ms,
   profile_stay_time, comment_stay_time or long_view -- in any form, including
   a derived column under a different name. They are outcomes of the same
   impression being ranked, so you cannot know them at ranking time.
   For user history: derive it from PRIOR impressions only, shifted so the
   current row is excluded, fitted on train alone. Auxiliary feedback objectives
   belong in train.py, where they are allowed.
5. numpy and pandas only. No torch, no sklearn.
6. Output the COMPLETE updated file, not a diff.
'''


def _backup(path: Path) -> Path:
    bak = path.with_suffix(".py.coder_bak")
    shutil.copy2(path, bak)
    return bak


def _validate(path: Path, module: str) -> tuple[bool, str]:
    """Parse, then import in a subprocess. Import catches NameError,
    bad signatures at module scope, and the leakage guard in features.py."""
    try:
        ast.parse(path.read_text())
    except SyntaxError as exc:
        return False, f"SyntaxError line {exc.lineno}: {exc.msg}"

    name = f"solution.{module[:-3]}"
    proc = subprocess.run(
        [sys.executable, "-c", f"import {name}"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return False, "ImportError: " + (tail[-1] if tail else "unknown")
    return True, ""


def apply_code_patch(idea: dict, model: str = "gpt-4o-mini",
                     prior_code: str | None = None) -> tuple[str, int, str]:
    """Generate and install one patch. Returns (code, tokens_used, module).

    `prior_code` puts the coder in REVISION mode: it edits the near-miss version
    rather than the accepted baseline, so a tuning fix builds on the attempt it
    is fixing instead of rewriting it from scratch.
    """
    module = idea.get("module", "train.py")
    if module not in EDITABLE_MODULES:
        raise ValueError(f"{module!r} is not agent-editable")

    path = SOLUTION / module
    revising = bool(prior_code)
    current = prior_code if revising else path.read_text()

    if revising:
        framing = (
            "You are REVISING a change that came close but did not beat the "
            "incumbent. The code below is that attempt. Make the ONE specific "
            "fix described, and change nothing else -- an unrelated edit makes "
            "the result unattributable."
        )
    else:
        framing = (
            "Rewrite the module to implement the hypothesis. Make the smallest "
            "change that tests it -- do not refactor unrelated code."
        )

    prompt = f"""You are an ML engineer improving a KuaiRand-Pure ranking pipeline.

## The change to make
Hypothesis: {idea['hypothesis']}
Rationale: {idea.get('rationale', '')}
Technique: {idea.get('source_technique', '')}
Predicted effect: {idea.get('predicted_effect', '')}
{INTERFACE}
## {'The attempt you are revising' if revising else f'Current solution/{module}'}
```python
{current}
```

{framing} Comment each change with a short note naming the technique.
Output ONLY the complete new file as Python."""

    code, tokens = complete("coder", prompt, fallback_model=model,
                            max_tokens=4000)

    if "```" in code:
        start = code.find("```")
        start = code.find("\n", start) + 1
        end = code.find("```", start)
        code = code[start:end if end != -1 else len(code)].strip()

    if not code.strip():
        raise ValueError("Coder returned empty output")

    bak = _backup(path)
    path.write_text(code)
    ok, err = _validate(path, module)
    if not ok:
        shutil.copy2(bak, path)          # never leave the tree broken
        bak.unlink(missing_ok=True)
        raise SyntaxError(f"Generated {module} rejected -- {err}")

    # Leakage check BEFORE the run, not after. A leaking feature scores well, so
    # the Reflector would keep it -- the only safe place to catch it is here.
    try:
        assert_clean(code, module)
    except ValueError as exc:
        shutil.copy2(bak, path)
        bak.unlink(missing_ok=True)
        raise ValueError(str(exc)) from None
    bak.unlink(missing_ok=True)

    logger.info(f"applied patch to {module} ({len(code.splitlines())} lines)")
    return code, tokens, module
