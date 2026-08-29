"""
instrument.py -- FROZEN. Measure what a patch ACTUALLY DID, before training it.

WHY. The other guards prove a patch parses, imports, and does not read a label.
None prove it does WHAT IT CLAIMED. That gap is the measured bottleneck: in one
25-cycle run the agent named the correct problem in 11 of 16 scored cycles and
not one implementation cleared the accept margin -- and at least one patch was
confirmed to change nothing at all.

Worse, a broken implementation of a GOOD hypothesis writes evidence AGAINST that
hypothesis into beliefs.jsonl, so the loop accumulates false negatives about
directions that may well work.

CHEAP BY CONSTRUCTION, and this is not a slogan -- the first version of this
file cost 241s against a 15s full training run, i.e. 16x the thing it was meant
to save, and hung a run for 28 minutes. The rules now:

  * the encoded data is computed ONCE and cached (feature fitting on 1.14M rows
    was being redone on every call)
  * NO training. optimiser_steps_per_epoch is len(batches), not a counted epoch.
    loss_fn_name is introspection, not two fits.
  * group statistics are vectorised over ONE batch, not looped over eight

Everything here is descriptive: no thresholds, no judgements. The contract says
what should hold; this says what does.
"""

from __future__ import annotations

import importlib
import inspect
import signal
import time

import numpy as np


class _MeasureTimeout(BaseException):
    """Derives from BaseException DELIBERATELY.

    As an Exception it was swallowed by the `except Exception` guarding each
    probe below: the alarm fired inside the first model build, the handler
    recorded it as a probe error, and because signal.alarm is one-shot the
    NEXT build ran with no cap at all. One cycle spent 35 minutes past a 120s
    limit that way. A guard's own exception must not be catchable by the code
    it guards.
    """


def _alarm(signum, frame):                          # noqa: ARG001
    raise _MeasureTimeout()


def _rearm(deadline: float) -> None:
    """Re-arm the one-shot alarm for whatever time is left.

    Belt and braces alongside the BaseException change: even if some future
    handler swallows the timeout, the next probe still starts with a live cap.
    """
    left = deadline - time.monotonic()
    if left <= 0:
        raise _MeasureTimeout()
    signal.alarm(max(1, int(left)))


_ENCODED: dict = {}


def _reload(name):
    return importlib.reload(importlib.import_module(name))


def invalidate_cache() -> None:
    """Call when features.py changes -- the encoding is no longer valid."""
    _ENCODED.clear()


def _encoded(ds, features):
    """Encode once. Feature fitting on 1.14M rows dominated the old cost."""
    key = id(ds)
    if key not in _ENCODED:
        state = features.fit(ds.train)
        _ENCODED[key] = (state,
                         features.transform(ds.train, state),
                         features.transform(ds.valid, state))
    return _ENCODED[key]


def _loss_fn_of(train_mod):
    """The default loss fit() would use, by introspection. No training."""
    fn = getattr(train_mod, "fit", None)
    if fn is None:
        return "unknown"
    try:
        sig = inspect.signature(fn)
        d = sig.parameters.get("loss_fn")
        if d is not None and d.default is not inspect.Parameter.empty:
            return getattr(d.default, "__name__", str(d.default))
    except (TypeError, ValueError):
        pass
    return "unknown"


def measure(ds, config: dict | None = None, timeout_seconds: int = 120) -> dict:
    """Realised state of the training setup. Never raises."""
    cfg = dict(config or {})
    out: dict = {}
    _prev = None
    try:
        _prev = signal.signal(signal.SIGALRM, _alarm)
        deadline = time.monotonic() + timeout_seconds
        signal.alarm(int(timeout_seconds))

        features = _reload("solution.features")
        train_mod = _reload("solution.train")

        state, tr, va = _encoded(ds, features)

        # ---- shape (free) ------------------------------------------------
        out["train_rows"] = int(len(tr.y))
        out["valid_rows"] = int(len(va.y))
        out["n_feature_fields"] = int(tr.X.shape[1])
        out["embedding_dim_total"] = int(state.dim)
        out["train_positive_rate"] = round(float(tr.y.mean()), 5)
        _, vc = np.unique(va.user_id, return_counts=True)
        out["eval_group_size_median"] = float(np.median(vc))
        out["eval_group_size_mean"] = round(float(vc.mean()), 2)

        # ---- batching: ask the module how it will batch -------------------
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        bs = int(cfg.get("batch", 8192))
        mode = str(cfg.get("batch_mode", "row"))
        gsz = cfg.get("group_size")
        if mode == "user" and hasattr(train_mod, "user_batches"):
            try:
                batches = train_mod.user_batches(tr.user_id, bs, rng, group_size=gsz)
            except TypeError:
                batches = train_mod.user_batches(tr.user_id, bs, rng)
        else:
            order = rng.permutation(len(tr.y))
            batches = [order[i:i + bs] for i in range(0, len(order), bs)]

        out["n_batches_per_epoch"] = int(len(batches))
        out["rows_covered"] = int(sum(len(b) for b in batches))
        out["rows_covered_exactly_once"] = bool(out["rows_covered"] == len(tr.y))
        # One optimiser step per batch. This is the number that caught a cycle
        # doing 240,024 steps instead of 140 -- and it needs no training to get.
        out["optimiser_steps_per_epoch"] = int(len(batches))

        # ---- group structure, vectorised over ONE batch -------------------
        if batches:
            b = batches[0]
            u, y = tr.user_id[b], tr.y[b]
            o = np.argsort(u, kind="stable")
            us, ys = u[o], y[o]
            starts = np.flatnonzero(np.r_[True, us[1:] != us[:-1]])
            lengths = np.diff(np.r_[starts, len(us)])
            npos = np.add.reduceat(ys, starts)
            mixed = (npos > 0) & (npos < lengths)
            out["train_group_size_median"] = float(np.median(lengths))
            out["train_group_size_mean"] = round(float(lengths.mean()), 2)
            out["train_group_size_max"] = int(lengths.max())
            out["pct_groups_mixed_label"] = round(100 * float(mixed.mean()), 2)
            out["pct_rows_in_mixed_label_group"] = round(
                100 * float(lengths[mixed].sum()) / max(len(b), 1), 2)
            out["unique_users_per_batch"] = int(len(lengths))

        # ---- model, by construction only (no training) --------------------
        _rearm(deadline)
        try:
            model_mod = _reload("solution.model")
            _m = model_mod.build(state, {**cfg, "seed": cfg.get("seed", 0)})
            out["model_type"] = type(_m).__name__
            n = 0
            for attr in vars(_m).values():
                if isinstance(attr, np.ndarray):
                    n += int(attr.size)
                elif isinstance(attr, (int, float, np.floating)):
                    n += 1
            out["model_n_params"] = int(n)
            out["model_k"] = int(getattr(_m, "k", 0) or 0)
        except Exception as exc:                          # noqa: BLE001
            out["model_probe_error"] = f"{type(exc).__name__}: {exc}"[:120]

        # ---- objective, by introspection then ONE forward pass ------------
        out["loss_fn_name"] = _loss_fn_of(train_mod)
        _rearm(deadline)
        try:
            model_mod = _reload("solution.model")
            lf = inspect.signature(train_mod.fit).parameters["loss_fn"].default
            if callable(lf) and batches:
                m = model_mod.build(state, {**cfg, "seed": cfg.get("seed", 0)})
                idx = batches[0]
                args = (m, tr.X[idx], tr.y[idx])
                if "user_id" in inspect.signature(lf).parameters:
                    args = args + (tr.user_id[idx],)
                dz, loss0 = lf(*args)                      # one forward pass
                out["initial_loss"] = round(float(loss0), 6)
                out["grad_dz_absmean"] = float(f"{np.abs(dz).mean():.3g}")
                out["grad_dz_nonzero_pct"] = round(100 * float((dz != 0).mean()), 2)
        except Exception as exc:                          # noqa: BLE001
            out["loss_probe_error"] = f"{type(exc).__name__}: {exc}"[:120]

    except _MeasureTimeout:
        out["error"] = (f"instrumentation exceeded {timeout_seconds}s -- the "
                        f"patch made even a single forward pass pathological")
        out["timed_out"] = True
    except Exception as exc:                              # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        if _prev is not None:
            signal.signal(signal.SIGALRM, _prev)
    return out


def diff(before: dict, after: dict) -> dict:
    """What actually changed between two measurements."""
    out = {}
    for k in sorted(set(before) | set(after)):
        b, a = before.get(k), after.get(k)
        if b == a:
            continue
        e = {"before": b, "after": a}
        if all(isinstance(x, (int, float)) and not isinstance(x, bool)
               for x in (b, a)):
            e["delta"] = round(a - b, 5)
            if b:
                e["ratio"] = round(a / b, 3)
        out[k] = e
    return out
