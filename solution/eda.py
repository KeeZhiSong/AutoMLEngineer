"""
eda.py -- FROZEN. Dataset inspection tools for the Ideator.

WHY THIS EXISTS. Every real insight in this project so far came from a human
measuring the data and writing the conclusion into a technique card. The agent
never looked at the dataset; it selected from a shelf someone else stocked. That
makes its "why" always "a card told me", which is not research.

These tools let the agent measure for itself.

DESIGN RULE, AND IT IS THE WHOLE POINT: every function returns NEUTRAL
MEASUREMENTS and NO INTERPRETATION. No function flags a "mismatch", ranks a
finding as important, or suggests an action. The moment a tool says
"train median 31 vs eval median 4 -- MISMATCH", the human has leaked the answer
through a different channel and the agent is executing, not discovering.
The numbers are the tool's job. Noticing is the agent's.

Cheap by construction: everything runs on already-loaded frames, no training.
"""

from __future__ import annotations

import numpy as np

from .dataset import FEEDBACK_COLUMNS, LABEL, Dataset, Split


def _pct(x) -> float:
    return round(float(x) * 100, 3)


def list_length_distribution(ds: Dataset) -> dict:
    """Impressions per user, per split."""
    out = {}
    for name in ("train", "valid", "test"):
        _, c = np.unique(ds[name].user_id, return_counts=True)
        out[name] = {
            "users": int(len(c)),
            "rows": int(ds[name].n),
            "min": int(c.min()), "p10": int(np.percentile(c, 10)),
            "median": float(np.median(c)), "mean": round(float(c.mean()), 2),
            "p90": int(np.percentile(c, 90)), "max": int(c.max()),
            "pct_users_with_at_most_5": _pct((c <= 5).mean()),
            "pct_users_with_at_most_10": _pct((c <= 10).mean()),
        }
    return out


def label_rates(ds: Dataset) -> dict:
    """Base rate of the scored label, per split."""
    return {n: {"label": LABEL, "positive_rate": round(float(ds[n].label.mean()), 4)}
            for n in ("train", "valid", "test")}


def feedback_signal_rates(ds: Dataset) -> dict:
    """Positive rate of each auxiliary feedback column on train."""
    tr = ds.train
    out = {}
    for c in FEEDBACK_COLUMNS:
        v = tr.feedback(c)
        if v.dtype.kind in "iub" or set(np.unique(v[:1000]).tolist()) <= {0, 1}:
            out[c] = {"positive_rate": round(float((v != 0).mean()), 5)}
        else:
            out[c] = {"mean": round(float(v.mean()), 2),
                      "median": float(np.median(v))}
    return out


def feedback_label_relationship(ds: Dataset, column: str) -> dict:
    """Joint behaviour of an auxiliary feedback column and the scored label.

    These are TRAINING TARGETS. This tool exists so their relationship to the
    label can be reasoned about; it does not make them usable as inputs.
    """
    tr = ds.train
    if column not in FEEDBACK_COLUMNS:
        return {"error": f"{column!r} is not a feedback column",
                "available": list(FEEDBACK_COLUMNS)}
    v = (tr.feedback(column) != 0).astype(float)
    y = tr.label
    return {
        "column": column,
        "corr_with_label": round(float(np.corrcoef(v, y)[0, 1]), 4),
        "P(label=1 | signal=1)": round(float(y[v == 1].mean()), 4) if v.sum() else None,
        "P(label=1 | signal=0)": round(float(y[v == 0].mean()), 4) if (1 - v).sum() else None,
    }


def cold_start_rates(ds: Dataset) -> dict:
    """Share of evaluation impressions whose ids were never seen in train."""
    tr = ds.train
    sv, sa, su = (set(tr.video_id.tolist()),
                  set(tr.log["author_id"].tolist()),
                  set(tr.user_id.tolist()))
    out = {}
    for n in ("valid", "test"):
        sp = ds[n]
        out[n] = {
            "pct_rows_unseen_video": _pct((~np.isin(sp.video_id, list(sv))).mean()),
            "pct_rows_unseen_author": _pct(
                (~np.isin(sp.log["author_id"].to_numpy(), list(sa))).mean()),
            "pct_rows_unseen_user": _pct((~np.isin(sp.user_id, list(su))).mean()),
        }
    return out


def evaluation_population(ds: Dataset, split: str = "valid") -> dict:
    """Per-user positive structure of an evaluation split.

    GAUC counts only users with 0 < positives < impressions; nDCG counts every
    user, scoring all-negative users as 0.
    """
    sp = ds[split]
    u, y = sp.user_id, sp.label
    order = np.argsort(u, kind="stable")
    us, ys = u[order], y[order]
    bounds = np.flatnonzero(np.r_[True, us[1:] != us[:-1]])
    groups = np.split(np.arange(len(us)), bounds[1:])
    npos = np.array([ys[g].sum() for g in groups])
    n = np.array([len(g) for g in groups])
    return {
        "split": split, "users": int(len(n)),
        "pct_users_all_negative": _pct((npos == 0).mean()),
        "pct_users_all_positive": _pct((npos == n).mean()),
        "pct_users_mixed": _pct(((npos > 0) & (npos < n)).mean()),
        "mean_positives_per_user": round(float(npos.mean()), 2),
    }


def temporal_span(ds: Dataset) -> dict:
    """Date coverage of each split, and the gaps between them."""
    out = {}
    for n in ("train", "valid", "test"):
        d = ds[n].log["date"].to_numpy()
        out[n] = {"first_date": int(d.min()), "last_date": int(d.max()),
                  "distinct_days": int(len(np.unique(d)))}
    return out


def usable_group_fraction(ds: Dataset, group_sizes=(None, 3, 5, 7, 10)) -> dict:
    """Share of TRAIN rows landing in a mixed-label group at various groupings.

    A grouped ranking loss can only learn from groups containing both classes;
    homogeneous groups carry no ordering signal.
    """
    from .train import eval_sized_groups, group_by_user
    tr = ds.train
    y, u = tr.label, tr.user_id
    out = {}
    for gs in group_sizes:
        groups = (eval_sized_groups(u, np.random.default_rng(0), gs) if gs
                  else group_by_user(u))
        sizes = np.array([len(g) for g in groups])
        rows = sum(len(g) for g in groups if 0 < y[g].sum() < len(g))
        out["full_user_lists" if gs is None else f"group_size_{gs}"] = {
            "n_groups": int(len(groups)),
            "median_group_size": float(np.median(sizes)),
            "pct_rows_in_mixed_label_group": _pct(rows / len(y)),
        }
    return out


def column_cardinalities(ds: Dataset) -> dict:
    """Distinct values per log column on train."""
    tr = ds.train.log
    return {c: int(tr[c].nunique()) for c in tr.columns}


TOOLS = {
    "list_length_distribution": list_length_distribution,
    "label_rates": label_rates,
    "feedback_signal_rates": feedback_signal_rates,
    "feedback_label_relationship": feedback_label_relationship,
    "cold_start_rates": cold_start_rates,
    "evaluation_population": evaluation_population,
    "temporal_span": temporal_span,
    "usable_group_fraction": usable_group_fraction,
    "column_cardinalities": column_cardinalities,
}


def describe_tools() -> str:
    """One-line signature per tool, for the Ideator's prompt."""
    return "\n".join(
        f"- {name}({'ds, column' if 'relationship' in name else 'ds'})"
        f"  -- {fn.__doc__.strip().splitlines()[0]}"
        for name, fn in TOOLS.items())
