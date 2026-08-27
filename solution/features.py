"""
features.py -- AGENT-EDITABLE.

Turns raw Split rows into the (N, F) integer field matrix that model.py consumes.

    fit(train)              -> FeatureState   (learned from TRAIN ONLY)
    transform(split, state) -> Encoded

Anything learned from data -- vocabularies, bucket edges, target statistics --
must be fitted in fit() on train and merely applied in transform(). A statistic
computed over valid or test is leakage, and it is the easiest way to produce a
great validation number that collapses on the hidden test set.

The starting point reproduces official/data.py encode() exactly: 5 fields,
duration bucketed into deciles on train, one UNK slot per field. That is the
floor, not the ceiling.

TWO THINGS THE ORGANISERS ALREADY MEASURED -- do not spend iterations rediscovering:

  * Adding the full 13-field CWM static feature set scores 0.5940 against 0.5950
    for these 5 fields. No gain.
  * Pure user-side first-order terms contribute EXACTLY ZERO. Ranking is computed
    within a user, so any term constant across that user's impressions cannot
    change the intra-group order. User-side features can only act through crosses
    with item-side features.

Feedback columns (is_click, is_like, play_time_ms, ...) are targets, not inputs.
Reading them here leaks the label. Use them via train.py's auxiliary objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dataset import FEEDBACK_COLUMNS, Split

# Updated field list to include time-based features for context-awareness
FIELDS = [
    "user_id", "video_id", "author_id", "tab", "dur_bucket", 
    "hour_bucket", "session_length"
]

N_DURATION_BUCKETS = 10
N_HOUR_BUCKETS = 24  # For each hour of the day

@dataclass
class Encoded:
    """What model.py consumes. Row order matches the Split it came from."""

    X: np.ndarray            # (N, F) int32, offsets already applied
    y: np.ndarray            # (N,) float32, the scored long_view label
    user_id: np.ndarray      # (N,) for within-user grouping and evaluation

    def __len__(self) -> int:
        return len(self.y)

@dataclass
class FeatureState:
    """Everything fitted on train and reused verbatim on valid/test."""

    vocabs: list[dict] = field(default_factory=list)
    unk: list[int] = field(default_factory=list)
    offsets: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    dim: int = 0
    duration_edges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fields: list[str] = field(default_factory=lambda: list(FIELDS))

def _raw_columns(split: Split, edges: np.ndarray) -> list[np.ndarray]:
    """Field values for every row, before vocabulary mapping."""
    log = split.log
    # Calculate session length: difference between max and min time_ms per user-session
    session_length = (log.groupby('user_id')['time_ms'].transform('max') -
                      log.groupby('user_id')['time_ms'].transform('min')) / 1000  # in seconds
    return [
        log["user_id"].to_numpy(),
        log["video_id"].to_numpy(),
        log["author_id"].to_numpy(),
        log["tab"].to_numpy(),
        np.searchsorted(edges, log["duration_ms"].to_numpy().astype(np.float64)),
        log["hourmin"].to_numpy() // 100,  # Hour bucket
        session_length.to_numpy()  # Session length
    ]

def fit(train: Split) -> FeatureState:
    """Learn vocabularies and bucket edges from the training split only."""
    durations = train.log["duration_ms"].to_numpy().astype(np.float64)
    edges = np.quantile(durations, np.linspace(0, 1, N_DURATION_BUCKETS + 1)[1:-1])

    columns = _raw_columns(train, edges)
    vocabs, unk, dims = [], [], []
    for col in columns:
        uniq = np.unique(col)
        vocabs.append({v: i for i, v in enumerate(uniq.tolist())})
        unk.append(len(uniq))
        dims.append(len(uniq) + 1)

    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)
    return FeatureState(vocabs=vocabs, unk=unk, offsets=offsets,
                        dim=int(sum(dims)), duration_edges=edges,
                        fields=list(FIELDS))

def transform(split: Split, state: FeatureState) -> Encoded:
    """Apply the fitted encoding to any split. Unseen values fall to UNK."""
    columns = _raw_columns(split, state.duration_edges)

    X = np.empty((split.n, len(columns)), dtype=np.int32)
    for i, col in enumerate(columns):
        vocab, unk, off = state.vocabs[i], state.unk[i], state.offsets[i]
        mapped = np.fromiter((vocab.get(v, unk) for v in col.tolist()),
                             dtype=np.int32, count=len(col))
        X[:, i] = mapped + off

    return Encoded(X=X, y=split.label, user_id=split.user_id)

def audit(state: FeatureState) -> dict:
    """Report the encoding for the ledger, so a diff is legible to a judge."""
    return {
        "fields": list(state.fields),
        "dim": int(state.dim),
        "field_dims": [len(v) + 1 for v in state.vocabs],
    }

_leaked = set(FIELDS) & set(FEEDBACK_COLUMNS)
if _leaked:
    raise ValueError(
        f"Feedback columns {_leaked} used as input features -- this leaks the "
        f"target. Use them as auxiliary objectives in train.py instead."
    )