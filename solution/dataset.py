"""
dataset.py -- FROZEN. The agent may not edit this file.

A richer loader than official/data.py, which keeps only 5 fields and drops the
label columns the organisers themselves point at (is_click, is_like, play_time_ms,
hourmin, ...). Those are needed for the multi-task and sequence directions, so we
read every column.

THE ROW ORDER HERE IS LOAD-BEARING. submit.py indexes the evaluation split by
row_id, defined as the order produced by official.data.load(): read
log_standard_4_08_to_4_21 first, then log_standard_4_22_to_5_08, filter by date,
keep original file order. verify_alignment() asserts we match it exactly, and the
test suite runs that against the real files. If this drifts, every submission
silently misaligns.

The label is long_view and is not configurable. Scoring authority is
official/evaluate.py, which is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Pinned by the starter kit. Do not change.
LABEL = "long_view"
SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)

# Files in KuaiRand-Pure that we deliberately DO NOT load, and why. Read this
# before "helpfully" adding one.
#
#   video_features_statistic_pure.csv  -- LEAKAGE HAZARD. 52 columns of per-video
#       aggregate counters (show_cnt, play_cnt, complete_play_cnt,
#       long_time_play_cnt, like_cnt, ...). They carry no date column and the
#       values are non-integer averages, i.e. they are aggregated over the whole
#       collection window -- which INCLUDES the 04-29..05-08 test period.
#       `long_time_play_cnt / show_cnt` is therefore very close to a per-video
#       long_view rate computed partly on test labels. Using it would inflate
#       validation and is exactly the kind of test-label contamination the rules
#       forbid. If you want an item-side popularity signal, compute it from the
#       TRAIN split only -- see the `popularity-prior-blend` technique card.
#
#   user_features_pure.csv -- available and legal, but the organisers measured
#       that adding the coarse user-side buckets gives no gain, and a term that
#       is constant within a user cannot reorder that user's list at all. Only
#       worth loading for a user x item CROSS, never as a first-order feature.
#
#   log_random_4_22_to_5_08_pure.csv -- randomised-exposure log (1.18M rows).
#       Legal and useful, but ONLY as an unbiased cross-check. It must never be
#       trained on, and selection must still be made on the official validation
#       split. See the `unbiased-validation` technique card.

# Columns carrying feedback other than the scored label. Available for auxiliary
# multi-task objectives; never as a direct input feature (they leak the target).
FEEDBACK_COLUMNS = (
    "is_click", "is_like", "is_follow", "is_comment", "is_forward",
    "is_hate", "is_profile_enter", "play_time_ms", "profile_stay_time",
    "comment_stay_time",
)


@dataclass
class Split:
    """One date-range slice, in official row order."""

    name: str
    log: pd.DataFrame          # every raw log column, original order preserved
    video: pd.DataFrame        # video-side features, indexed by video_id

    @property
    def n(self) -> int:
        return len(self.log)

    @property
    def user_id(self) -> np.ndarray:
        return self.log["user_id"].to_numpy()

    @property
    def video_id(self) -> np.ndarray:
        return self.log["video_id"].to_numpy()

    @property
    def label(self) -> np.ndarray:
        """The scored relevance label: long_view, binarised as the official
        loader does it (`1 if value != '0' else 0`)."""
        return (self.log[LABEL].to_numpy() != 0).astype(np.float32)

    def feedback(self, name: str) -> np.ndarray:
        """An auxiliary feedback signal, for multi-task objectives."""
        if name not in FEEDBACK_COLUMNS:
            raise KeyError(
                f"{name!r} is not an auxiliary feedback column. "
                f"Available: {FEEDBACK_COLUMNS}"
            )
        return self.log[name].to_numpy()


@dataclass
class Dataset:
    train: Split
    valid: Split
    test: Split
    video: pd.DataFrame = field(repr=False)

    def __getitem__(self, name: str) -> Split:
        return {"train": self.train, "valid": self.valid, "test": self.test}[name]


def load(data_dir: str | Path) -> Dataset:
    """Read the two standard logs plus video features, split by date.

    Order matches official.data.load() exactly -- see module docstring.
    """
    data_dir = Path(data_dir)

    video = pd.read_csv(data_dir / "video_features_basic_pure.csv")

    frames = [pd.read_csv(data_dir / f) for f in LOG_FILES]
    log = pd.concat(frames, ignore_index=True)   # file order, then row order

    # author_id is the one video-side field the official baseline uses. Merging
    # here (rather than in features) keeps the join out of the agent's way; the
    # rest of the video table stays available via Split.video.
    log = log.merge(video[["video_id", "author_id"]], on="video_id", how="left")
    log["author_id"] = log["author_id"].fillna(-1).astype(np.int64)

    splits = {}
    for name, (lo, hi) in SPLITS.items():
        mask = (log["date"] >= lo) & (log["date"] <= hi)
        # .copy() so downstream edits cannot write through to the parent frame.
        sub = log.loc[mask].reset_index(drop=True).copy()
        splits[name] = Split(name=name, log=sub, video=video)

    return Dataset(train=splits["train"], valid=splits["valid"],
                   test=splits["test"], video=video)


def verify_alignment(ds: Dataset, data_dir: str | Path) -> dict[str, int]:
    """Assert our row order matches official.data.load() on every split.

    Returns per-split row counts. Raises AssertionError on any mismatch. This is
    the guard that keeps submission row_ids valid -- run it in CI, not just once.
    """
    import sys

    official_dir = Path(__file__).resolve().parent.parent / "official"
    sys.path.insert(0, str(official_dir))
    try:
        import data as official_data
        ref = official_data.load(str(data_dir))
    finally:
        sys.path.remove(str(official_dir))

    counts = {}
    for name in ("train", "valid", "test"):
        ours, theirs = ds[name], ref[name]
        assert ours.n == len(theirs), (
            f"{name}: row count {ours.n} != official {len(theirs)}"
        )
        # official rows are (date, user_id, video_id, author, tab, dur, label)
        # with ids as strings; compare as strings to sidestep dtype differences.
        ours_u = ours.user_id.astype(str)
        ours_v = ours.video_id.astype(str)
        ours_y = ours.label
        for i in (0, ours.n // 2, ours.n - 1):
            assert ours_u[i] == theirs[i][1], f"{name} row {i}: user_id mismatch"
            assert ours_v[i] == theirs[i][2], f"{name} row {i}: video_id mismatch"

        theirs_u = np.array([r[1] for r in theirs])
        theirs_v = np.array([r[2] for r in theirs])
        theirs_y = np.array([r[6] for r in theirs], dtype=np.float32)
        assert np.array_equal(ours_u, theirs_u), f"{name}: user_id sequence differs"
        assert np.array_equal(ours_v, theirs_v), f"{name}: video_id sequence differs"
        assert np.array_equal(ours_y, theirs_y), f"{name}: label sequence differs"
        counts[name] = ours.n

    return counts


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Load and verify the KuaiRand splits.")
    ap.add_argument("--data_dir", required=True)
    a = ap.parse_args()

    ds = load(a.data_dir)
    print({k: ds[k].n for k in ("train", "valid", "test")})
    print("verifying against official.data.load() ...")
    print("aligned:", verify_alignment(ds, a.data_dir))
