"""
model.py -- AGENT-EDITABLE.

The scoring function. Consumes features.Encoded, emits one real score per row;
only relative order within a user matters.

    build(state, config) -> Model
    Model.predict(X)     -> (N,) float scores

The starting point is the official baseline's Factorization Machine (k=16), which
is what we must beat. It is a real, trained model -- not a placeholder.

MEASURED BY THE ORGANISERS: embedding size k = 8 / 16 / 32 scores
0.5895 / 0.5902 / 0.5887. Capacity is NOT the bottleneck; 1.14M rows will not
support a bigger model. Swapping FM for DeepFM/DCN/xDeepFM is explicitly ranked
BELOW changing the loss (train.py) and adding sequence features (features.py).

Any replacement must keep gradients available to train.py -- see FM.grad().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FeatureState


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class FM:
    """Factorization Machine. Second-order interactions via the standard
    O(nk) identity: 0.5 * ((sum e)^2 - sum(e^2))."""

    V: np.ndarray      # (dim, k) interaction embeddings
    W: np.ndarray      # (dim,)   first-order weights
    b: np.float32      # global bias
    k: int

    def logits(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (scores, per-field embeddings, summed embedding)."""
        E = self.V[X]                                   # (B, F, k)
        S = E.sum(1)                                    # (B, k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        """Score rows in batches. Raw logits -- evaluation only ranks them."""
        return np.concatenate(
            [self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)]
        )

    def grad(self, X: np.ndarray, dz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Backprop an upstream gradient dz (dLoss/dlogit, one per row).

        Separating this from the loss is what lets train.py swap pointwise
        logloss for a pairwise or listwise objective without touching the model.
        """
        _, E, S = self.logits(X)
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, dz[:, None])
        np.add.at(gV, X, dz[:, None, None] * (S[:, None, :] - E))
        return gV, gW, float(dz.sum())

    def params(self) -> tuple[np.ndarray, np.ndarray]:
        return self.V, self.W

    def snapshot(self) -> tuple:
        return self.V.copy(), self.W.copy(), np.float32(self.b)

    def restore(self, snap: tuple) -> None:
        self.V, self.W, self.b = snap[0], snap[1], np.float32(snap[2])


def build(state: FeatureState, config: dict | None = None) -> FM:
    """Construct the scoring model from the fitted feature state."""
    cfg = config or {}
    k = int(cfg.get("k", 16))
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    return FM(
        V=rng.normal(0, 0.01, (state.dim, k)).astype(np.float32),
        W=np.zeros(state.dim, dtype=np.float32),
        b=np.float32(0.0),
        k=k,
    )


def audit(model: FM) -> dict:
    """Describe the model for the ledger."""
    return {
        "type": type(model).__name__,
        "k": getattr(model, "k", None),
        "n_params": int(model.V.size + model.W.size + 1),
    }
