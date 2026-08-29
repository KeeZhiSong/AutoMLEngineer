"""
train.py -- AGENT-EDITABLE.

The optimiser and the objective.

    fit(model, train_enc, valid_enc, config) -> (model, history)

The starting point matches the official baseline: pointwise logistic loss, Adam
(lr 1e-3), batch 8192, up to 40 epochs, early stopping on validation primary with
patience 4.

THIS IS THE ORGANISERS' #1 RANKED DIRECTION. Their words: the loss is pointwise
logloss while the metrics (GAUC, nDCG@5) are *ranking* metrics. Aligning the
objective with the scoring -- pairwise BPR, or a listwise softmax over each
user's impressions -- is what they consider most likely to work. Encoded carries
`user_id` precisely so a grouped objective can be written here.

Their #3 direction also lives here: the auxiliary feedback signals
(is_click, is_like, is_follow, is_comment, is_forward, play_time_ms) reachable via
Split.feedback(), used as auxiliary tasks for the long_view main task. Those are
targets -- they must never become input features.

Early stopping reads the VALIDATION split only. The test split is not available
to this function and must not be.
"""

from __future__ import annotations

import inspect
import time

import numpy as np

from .features import Encoded
from .model import FM, sigmoid
from .scoring import score


# Random scoring gets 0.4834 primary on valid. Anything below that is broken,
# not merely undertrained.
RANDOM_FLOOR = 0.4834


def group_segments(user_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort rows into contiguous per-user segments, for VECTORISED grouped losses.

    Returns (order, starts, lengths). Prefer this to looping over
    group_by_user(): a Python loop runs once per user (~26,000 iterations)
    inside every batch.
    """
    order = np.argsort(user_id, kind="stable")
    su = user_id[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    lengths = np.diff(np.r_[starts, len(su)])
    return order, starts, lengths


def segment_softmax(scores: np.ndarray, starts: np.ndarray,
                    lengths: np.ndarray) -> np.ndarray:
    """Softmax within each segment. `scores` must be sorted by group.
    Numerically stable: subtracts each segment's max first."""
    seg_max = np.maximum.reduceat(scores, starts)
    e = np.exp(scores - np.repeat(seg_max, lengths))
    seg_sum = np.add.reduceat(e, starts)
    return e / np.repeat(seg_sum, lengths)


def segment_sum(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Sum within each segment. `values` must be sorted by group."""
    return np.add.reduceat(values, starts)


def group_by_user(user_id: np.ndarray) -> list[np.ndarray]:
    """Split row indices into per-user groups.

    Provided because within-user grouping is the fiddly part of every ranking
    loss and getting it wrong is silent -- the run still produces a number, just
    a meaningless one. Returns a list of index arrays, one per distinct user.

        for g in group_by_user(user_ids):
            s = scores[g]           # this user's impression scores
            y = labels[g]           # this user's labels

    Note the gradient contract: `loss_fn` returns `dz`, one value per ROW of the
    batch, in batch order. To scatter per-group gradients back, write into a
    zeros array at the group indices:

        dz = np.zeros(len(y), dtype=np.float32)
        for g in group_by_user(user_ids):
            dz[g] = <this group's dLoss/dlogit>
    """
    order = np.argsort(user_id, kind="stable")
    sorted_u = user_id[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_u[1:] != sorted_u[:-1]])
    return np.split(order, boundaries[1:])


def eval_sized_groups(user_id: np.ndarray, rng, size: int = 5,
                      jitter: int = 2) -> list[np.ndarray]:
    """Split each user's rows into SHORT lists resembling the evaluation split.

    MEASURED MOTIVATION. Impressions per user:
        valid  median  4   mean  5.6
        test   median  5   mean  7.1
    63.7% of validation users have their ENTIRE list inside the nDCG@5 cutoff.

    So a ranking loss trained on a user's full 31-item train list optimises a
    materially different distribution from the 4-5 item lists it is scored on.
    This chops each user's rows into groups of roughly `size` (+/- jitter) so the
    training objective sees list lengths matching evaluation.

    Returns a list of index arrays, one per synthetic short list.
    """
    out = []
    for g in group_by_user(user_id):
        g = g.copy()
        rng.shuffle(g)
        i = 0
        while i < len(g):
            n = max(1, size + int(rng.integers(-jitter, jitter + 1)))
            out.append(g[i:i + n])
            i += n
    return out


def user_batches(user_id: np.ndarray, batch_size: int, rng,
                 group_size: int | None = None) -> list[np.ndarray]:
    """Batch by USER, keeping each user's impression list intact.

    Keeps each user's impression list intact within a batch, instead of the
    default random permutation which splits a user's rows across many batches.

    Use this whenever the objective is a within-user ranking loss. For a
    pointwise loss it makes no difference and row batching is fine.
    """
    # group_size chops users into eval-sized lists first -- see
    # eval_sized_groups() for the measured reason this matters.
    groups = (eval_sized_groups(user_id, rng, group_size) if group_size
              else group_by_user(user_id))
    rng.shuffle(groups)

    batches, current, count = [], [], 0
    for g in groups:
        # A single user larger than the batch still ships whole -- splitting it
        # would reintroduce exactly the fragmentation this function prevents.
        if count and count + len(g) > batch_size:
            batches.append(np.concatenate(current))
            current, count = [], 0
        current.append(g)
        count += len(g)
    if current:
        batches.append(np.concatenate(current))
    return batches


class Adam:
    """Adam, matching the official baseline's hyperparameters."""

    def __init__(self, params: list[np.ndarray], lr: float = 1e-3,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] *= self.b1
            self.m[i] += (1 - self.b1) * g
            self.v[i] *= self.b2
            self.v[i] += (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def focal_logloss(model: FM, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Binary focal loss. Returns (dLoss/dlogit per row, mean loss)."""
    # focal-loss-weighting: downweight easy examples and upweight rare positives.
    gamma = 2.0
    alpha = 0.75

    z, _, _ = model.logits(X)
    p = np.clip(sigmoid(z), 1e-7, 1.0 - 1e-7).astype(np.float32)

    pos = y
    neg = 1.0 - y
    q = 1.0 - p

    # focal-loss-weighting: loss contribution depends on prediction correctness.
    loss_pos = -alpha * pos * (q ** gamma) * np.log(p)
    loss_neg = -(1.0 - alpha) * neg * (p ** gamma) * np.log(q)
    loss = float(np.mean(loss_pos + loss_neg))

    # focal-loss-weighting: exact d focal-loss / d logit for FM.grad().
    grad_pos = alpha * pos * (
        gamma * p * (q ** gamma) * np.log(p) - (q ** (gamma + 1.0))
    )
    grad_neg = (1.0 - alpha) * neg * (
        (p ** (gamma + 1.0)) - gamma * (p ** gamma) * q * np.log(q)
    )
    dz = ((grad_pos + grad_neg) / len(y)).astype(np.float32)
    return dz, loss


def pointwise_logloss(model: FM, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Baseline objective retained for optional experiments."""
    z, _, _ = model.logits(X)
    p = sigmoid(z)
    dz = ((p - y) / len(y)).astype(np.float32)
    loss = float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
    return dz, loss


def _takes_user_id(fn) -> bool:
    """Whether a loss function wants per-row user ids (i.e. is a grouped loss)."""
    try:
        return "user_id" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def fit(model: FM,
        train_enc: Encoded,
        valid_enc: Encoded,
        config: dict | None = None,
        loss_fn=focal_logloss,  # focal-loss-weighting: make focal loss the default objective.
        log=print) -> tuple[FM, list[dict]]:
    """Train with early stopping on validation primary. Never sees test."""
    cfg = config or {}
    lr = float(cfg.get("lr", 1e-3))
    l2 = float(cfg.get("l2", 1e-6))
    bs = int(cfg.get("batch", 8192))
    max_epochs = int(cfg.get("epochs", 40))
    patience = int(cfg.get("patience", 4))
    seed = int(cfg.get("seed", 0))

    # "user" keeps each user's impressions in one batch -- required for any
    # within-user ranking loss. "row" is classic random-permutation batching,
    # correct for a pointwise loss.
    batch_mode = str(cfg.get("batch_mode", "row"))

    rng = np.random.default_rng(seed)
    opt = Adam(list(model.params()), lr=lr)

    best, best_snap, bad = -1.0, model.snapshot(), 0
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        if batch_mode == "user":
            batches = user_batches(train_enc.user_id, bs, rng,
                                   group_size=cfg.get("group_size"))
        else:
            order = rng.permutation(len(train_enc))
            batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        losses = []

        for idx in batches:
            Xb, yb = train_enc.X[idx], train_enc.y[idx]

            # A grouped loss needs the batch's user ids; a pointwise one does
            # not. Pass them only when the loss accepts them.
            if _takes_user_id(loss_fn):
                dz, loss = loss_fn(model, Xb, yb, train_enc.user_id[idx])
            else:
                dz, loss = loss_fn(model, Xb, yb)
            gV, gW, gb = model.grad(Xb, dz)
            gV += l2 * model.V
            gW += l2 * model.W

            opt.step([model.V, model.W], [gV, gW])
            model.b -= np.float32(lr * gb)
            losses.append(loss)

        metrics = score(valid_enc.user_id, valid_enc.y, model.predict(valid_enc.X))
        row = {"epoch": epoch, "loss": float(np.mean(losses)),
               "seconds": round(time.time() - t0, 2), **metrics}
        history.append(row)
        log(f"  epoch {epoch:2d} | loss {row['loss']:.4f} | valid "
            f"GAUC {metrics['GAUC']:.4f} nDCG@5 {metrics['nDCG@5']:.4f} "
            f"primary {metrics['primary']:.4f} | {row['seconds']}s")

        # Divergence guard. A run scoring below the random baseline (0.4834 on
        # valid) is not a slow starter, it is broken -- an unstable learning
        # rate or a sign error. Observed: a bad pairwise loss burned all 40
        # epochs to finish at 0.3974. Abort so the loop spends its budget on
        # the next idea instead.
        if epoch >= 3 and metrics["primary"] < RANDOM_FLOOR:
            log(f"  diverged: primary {metrics['primary']:.4f} is below the "
                f"random floor {RANDOM_FLOOR} at epoch {epoch}; aborting")
            row["diverged"] = True
            break

        if metrics["primary"] > best + 1e-5:
            best, bad = metrics["primary"], 0
            best_snap = model.snapshot()
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stop at epoch {epoch}")
                break

    model.restore(best_snap)
    return model, history
