"""
Specifier: state what the intervention should CAUSE, before writing any code.

A semantic contract, fixed BEFORE the patch exists and immutable afterwards.

WHY THE ORDER MATTERS. If the agent writes the code first and the contract
second, it will write a contract its code happens to satisfy -- the same failure
shape as the metric gaming incident, where the model optimised the measurement
instead of the thing. So the contract is derived from the NAMED PROBLEM (which
the classifier produced independently) and from the stated hypothesis, and the
coder never sees it while writing.

The contract has three parts:

  postconditions  what must CHANGE, as a measurable comparison
  invariants      what must NOT change
  rationale       why satisfying it would mean the hypothesis was implemented

Checked against solution/instrument.py BEFORE any training compute is spent.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.llm import complete  # noqa: E402

logger = logging.getLogger("amra.specifier")

# Quantities a patch can legitimately CONTROL. Postconditions only over these.
CONTROLLABLE = [
    # data / batching -- what a features.py or batching change moves
    "n_feature_fields", "embedding_dim_total",
    "n_batches_per_epoch", "optimiser_steps_per_epoch",
    "train_group_size_median", "train_group_size_mean", "train_group_size_max",
    "pct_groups_mixed_label", "pct_rows_in_mixed_label_group",
    "unique_users_per_batch",
    # objective -- what a train.py LOSS change moves. Without these, a
    # hypothesis about the loss had nothing legitimate to contract over, so the
    # specifier picked an unrelated batching metric and the gate blocked a
    # CORRECT focal-loss patch for failing a condition it never controlled.
    "loss_fn_name", "initial_loss", "grad_dz_absmean", "grad_dz_nonzero_pct",
    # model -- what a model.py change moves. Without these, a model.py
    # intervention had no legitimate target at all.
    "model_type", "model_n_params", "model_k",
]

# Properties of the DATA, not of the training setup. A patch cannot change these
# without resampling rows, which the mandatory invariants forbid -- so a
# postcondition over one is self-contradictory and blocks the cycle for nothing.
#
# Observed: a contract demanded train_positive_rate become 0.3134 (the
# VALIDATION rate) when it is 0.3366. Unreachable, and the cycle was blocked for
# failing a condition it could never have met.
FIXED_BY_DATA = [
    "train_rows", "valid_rows", "rows_covered", "rows_covered_exactly_once",
    "train_positive_rate", "eval_group_size_median", "eval_group_size_mean",
]

MEASURABLE = CONTROLLABLE + FIXED_BY_DATA

# An over-constrained contract blocks everything. One arrived with 17
# invariants; the mandatory three plus one or two deliberate ones is the point.
MAX_INVARIANTS = 6
MAX_POSTCONDITIONS = 3

OPS = ("<", "<=", ">", ">=", "==", "!=", "approx", "changed", "unchanged")

# Always enforced, whatever the agent writes. These are the invariants whose
# violation previously produced a false win.
MANDATORY_INVARIANTS = [
    {"metric": "train_rows", "op": "unchanged"},
    {"metric": "valid_rows", "op": "unchanged"},
    {"metric": "rows_covered_exactly_once", "op": "==", "value": True},
]


def specify(idea: dict, problem: dict, before: dict,
            model: str = "gpt-4o") -> tuple[dict, int]:
    """Produce the semantic contract for an intervention. Returns (contract, tokens)."""
    prompt = f"""You are about to implement an intervention. BEFORE writing any
code, state what it must measurably CAUSE.

THE PROBLEM (named independently from measurements)
  {problem.get('statement','')}
  dimension: {problem.get('dimension','')}
  magnitude: {problem.get('magnitude','')}

THE INTERVENTION
  {idea.get('hypothesis','')}
  module: {idea.get('module','')}
  mechanism: {idea.get('rationale','')}

CURRENT MEASURED STATE of the training setup:
{json.dumps({k: v for k, v in before.items() if k in MEASURABLE}, indent=2)}

Postconditions may ONLY be written over quantities a patch can CONTROL:
{chr(10).join('  ' + m for m in CONTROLLABLE)}

MATCH THE METRIC TO THE INTERVENTION. This is by far the most common way to
write a contract that blocks correct code:

  If your change is to the LOSS or how rows are WEIGHTED
  -- focal loss, pairwise, reweighting, temperature, hard-negative mining,
     instance weights, auxiliary terms --
  then use: loss_fn_name, grad_dz_absmean, grad_dz_nonzero_pct, initial_loss
  and NOTHING ELSE. Re-weighting rows does not change which rows exist or how
  they are grouped. `pct_rows_in_mixed_label_group` is GROUP COMPOSITION, not
  gradient magnitude -- it will not move, and your correct patch will be
  rejected.

  If your change is to BATCHING or GROUPING
  -- batch_mode, group_size, how users are packed into batches --
  then use: train_group_size_median/mean/max, pct_groups_mixed_label,
  pct_rows_in_mixed_label_group, unique_users_per_batch,
  optimiser_steps_per_epoch.

  If your change is to FEATURES
  -- adding or removing fields, changing the encoding --
  then use: n_feature_fields, embedding_dim_total.

  If your change is to the MODEL
  -- architecture, extra embeddings, dropout, a different scoring function --
  then use: model_type, model_n_params, model_k, and possibly initial_loss.
  NOT embedding_dim_total: that is the feature vocabulary size, set by
  features.py, and a model change will not move it.

Ask yourself: "if my code is perfect, will this number actually move?" If you
are not sure, pick a different metric.

These are properties of the DATA and CANNOT be changed by your patch. They are
available as invariants only -- a postcondition over one is impossible to
satisfy and would block the cycle for no reason:
{chr(10).join('  ' + m for m in FIXED_BY_DATA)}

Write the contract. Rules:
  · A postcondition is a claim that would be FALSE if your intervention had not
    been implemented. If a postcondition would hold even when nothing changed,
    it is worthless -- delete it.
  · PREFER `changed` OVER A DIRECTION. Your job here is to prove the
    intervention HAPPENED, not to predict which way it moves the metric. Two
    real rejections of CORRECT code:
      - a loss redesign required `initial_loss < 0.693`; the loss moved to
        0.866, which PROVES the patch worked, and it was rejected for moving
        the wrong way.
      - a Transformer scoring function required `model_type == "Transformer"`
        and produced "TransformerModel" -- rejected on a substring.
    `initial_loss changed` and `model_type changed` would have passed both, and
    a no-op still fails them. Use `<`/`>`/`==` only when the direction or exact
    value is genuinely part of what you are claiming to implement.
    Your directional prediction belongs in `predicted_effect`, which is recorded
    and does not gate anything.
  · Use the CURRENT values above to pick thresholds that actually discriminate.
    "train_group_size_median < 31" is not discriminating if it is already 5.
  · 1-3 postconditions. Prefer the ONE quantity your hypothesis is really
    about. More conditions is not a stronger contract, only a more brittle one.
  · At most 6 invariants, and three are added automatically (row counts and
    coverage). Add one or two more only if your change could plausibly break
    something else. An over-constrained contract blocks every cycle.
  · Do NOT restate the metric you hope to improve. This contract is about
    whether the CODE DID THE THING, not whether the thing helped.

ops: {", ".join(OPS)}   ("changed"/"unchanged" take no value)

Return ONLY JSON:
{{"target_quantity": "<the one quantity this is really about>",
  "postconditions": [{{"metric": "<measurable>", "op": "<op>", "value": <number|true|false>,
                      "why": "<why this proves implementation>"}}],
  "invariants": [{{"metric": "<measurable>", "op": "unchanged"}}],
  "rationale": "<one sentence: satisfying this means the hypothesis was implemented>"}}"""

    txt, tokens = complete("classifier", prompt, fallback_model=model,
                           max_tokens=900, json_mode=True)
    c = json.loads(txt)

    # Drop postconditions the patch cannot possibly satisfy.
    kept, dropped = [], []
    for pc in c.get("postconditions", []):
        if pc.get("metric") in CONTROLLABLE and pc.get("op") in OPS:
            kept.append(pc)
        elif pc.get("metric") in FIXED_BY_DATA:
            dropped.append(pc["metric"])
    if dropped:
        logger.info(f"    dropped unsatisfiable postconditions on {dropped} "
                    f"(fixed by the data, not by the patch)")
    c["postconditions"] = kept[:MAX_POSTCONDITIONS]
    c["invariants"] = [i for i in c.get("invariants", [])
                       if i.get("metric") in MEASURABLE and i.get("op") in OPS
                       ][:MAX_INVARIANTS]
    # The mandatory ones are not the agent's to negotiate.
    have = {(i["metric"], i["op"]) for i in c["invariants"]}
    for m in MANDATORY_INVARIANTS:
        if (m["metric"], m["op"]) not in have:
            c["invariants"].append(dict(m))

    if not c["postconditions"]:
        logger.warning("    contract has no satisfiable postcondition; "
                       "semantic gate disabled for this cycle")
        return None, tokens

    logger.info(f"  contract on '{c.get('target_quantity','?')}': "
                f"{len(c['postconditions'])} postconditions, "
                f"{len(c['invariants'])} invariants")
    for p in c["postconditions"]:
        v = p.get("value")
        logger.info(f"    require {p['metric']} {p['op']}"
                    f"{'' if v is None else ' ' + str(v)}")
    return c, tokens


def _cmp(op: str, got, want) -> bool:
    if op == "unchanged":
        return got is None or got == want          # want carries the 'before'
    if op == "changed":
        return got != want
    if got is None:
        return False
    try:
        if op == "<":   return got < want
        if op == "<=":  return got <= want
        if op == ">":   return got > want
        if op == ">=":  return got >= want
        if op == "==":  return got == want
        if op == "!=":  return got != want
        if op == "approx":
            return abs(float(got) - float(want)) <= max(0.05 * abs(float(want)), 1e-9)
    except TypeError:
        return False
    return False


def check(contract: dict, before: dict, after: dict) -> dict:
    """Was the contract satisfied? Returns a verdict with per-clause detail."""
    results, failures = [], []

    for p in contract.get("postconditions", []):
        metric, op = p["metric"], p["op"]
        want = before.get(metric) if op in ("changed", "unchanged") else p.get("value")
        ok = _cmp(op, after.get(metric), want)
        row = {"kind": "postcondition", "metric": metric, "op": op,
               "want": want, "got": after.get(metric), "ok": ok}
        results.append(row)
        if not ok:
            failures.append(f"{metric} {op} {want} -> got {after.get(metric)}")

    for i in contract.get("invariants", []):
        metric, op = i["metric"], i["op"]
        want = before.get(metric) if op in ("changed", "unchanged") else i.get("value")
        ok = _cmp(op, after.get(metric), want)
        row = {"kind": "invariant", "metric": metric, "op": op,
               "want": want, "got": after.get(metric), "ok": ok}
        results.append(row)
        if not ok:
            failures.append(f"INVARIANT {metric} {op} {want} -> got {after.get(metric)}")

    return {"satisfied": not failures, "failures": failures, "clauses": results}
