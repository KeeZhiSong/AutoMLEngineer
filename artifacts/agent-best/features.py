"""
features.py -- AGENT-EDITABLE.

Turns raw Split rows into the (N, F) integer field matrix that model.py consumes.

    fit(train)              -> FeatureState   (learned from TRAIN ONLY)
    transform(split, state) -> Encoded

Anything learned from data -- vocabularies, bucket edges, target statistics --
must be fitted in fit() on train and merely applied in transform(). A statistic
computed over valid or test is leakage.

The starting point reproduces official/data.py encode() exactly: 5 fields,
duration bucketed into deciles on train, one UNK slot per field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dataset import Split

# The official baseline's five fields.
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]

N_DURATION_BUCKETS = 10

# length-normalization: synthetic padding field used to extend feature lists.
PAD_FIELD_PREFIX = "__length_norm_pad__"


@dataclass
class Encoded:
    """What model.py consumes. Row order matches the Split it came from."""

    X: np.ndarray            # (N, F) int32, offsets already applied
    y: np.ndarray            # (N,) float32, the scored label
    user_id: np.ndarray      # (N,) for within-user grouping and evaluation
    aux: dict[str, np.ndarray] = field(default_factory=dict)
    video_id: np.ndarray | None = None

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
    return [
        log["user_id"].to_numpy(),
        log["video_id"].to_numpy(),
        log["author_id"].to_numpy(),
        log["tab"].to_numpy(),
        np.searchsorted(edges, log["duration_ms"].to_numpy().astype(np.float64)),
    ]


def fit(train: Split) -> FeatureState:
    """Learn vocabularies and bucket edges from the training split only."""
    durations = train.log["duration_ms"].to_numpy().astype(np.float64)
    edges = np.quantile(durations, np.linspace(0, 1, N_DURATION_BUCKETS + 1)[1:-1])

    columns = _raw_columns(train, edges)

    # length-normalization: fit target padding length on TRAIN only.
    # The hypothesis names the validation median, but using validation statistics
    # here would leak; the train median is the non-leaky analogue.
    user_list_lengths = train.log["user_id"].value_counts().to_numpy()
    if len(user_list_lengths) == 0:
        pad_to_length = len(FIELDS)
    else:
        pad_to_length = max(len(FIELDS), int(np.ceil(np.median(user_list_lengths))))

    pad_fields = [f"{PAD_FIELD_PREFIX}_{i}" for i in range(pad_to_length - len(FIELDS))]
    fields = list(FIELDS) + pad_fields

    vocabs, unk, dims = [], [], []
    for col in columns:
        uniq = np.unique(col)
        vocabs.append({v: i for i, v in enumerate(uniq.tolist())})
        unk.append(len(uniq))
        dims.append(len(uniq) + 1)

    # length-normalization: each padding position is a one-value dummy field.
    for _ in pad_fields:
        vocabs.append({})
        unk.append(0)
        dims.append(1)

    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)
    return FeatureState(vocabs=vocabs, unk=unk, offsets=offsets,
                        dim=int(sum(dims)), duration_edges=edges,
                        fields=fields)


def transform(split: Split, state: FeatureState) -> Encoded:
    """Apply the fitted encoding to any split. Unseen values fall to UNK."""
    columns = _raw_columns(split, state.duration_edges)

    X = np.empty((split.n, len(state.fields)), dtype=np.int32)
    for i, col in enumerate(columns):
        vocab, unk, off = state.vocabs[i], state.unk[i], state.offsets[i]
        mapped = np.fromiter((vocab.get(v, unk) for v in col.tolist()),
                             dtype=np.int32, count=len(col))
        X[:, i] = mapped + off

    # length-normalization: append dummy entries so every encoded list reaches
    # the train-fitted median list length.
    for i in range(len(columns), len(state.fields)):
        X[:, i] = state.offsets[i]

    return Encoded(X=X, y=split.label, user_id=split.user_id,
                   video_id=split.video_id)


def audit(state: FeatureState) -> dict:
    """Report the encoding for the ledger, so a diff is legible to a judge."""
    return {
        "fields": list(state.fields),
        "dim": int(state.dim),
        "field_dims": [len(v) + 1 for v in state.vocabs],
        "pad_to_length": int(len(state.fields)),  # length-normalization
    }