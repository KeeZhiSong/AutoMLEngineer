"""
Planner: one committed hypothesis -> several IMPLEMENTATIONS, scored, best chosen.

WHY THIS EXISTS. Run 11 diagnosed the decisive problem four times and shipped a
working patch zero times. The candidates were not the problem -- in every one of
those cycles a direct `train.py` change that could have expressed the fix was
GENERATED and then lost the selection step to something more elaborate:
curriculum learning, adaptive embedding sizes, a novelty feature. All four
changed nothing measurable.

So the loop already searched hypotheses. It never searched IMPLEMENTATIONS. This
stage sits between INVENT and CODE and asks: given that we are committed to this
mechanism, what are the ways to realise it, and which one most directly tests it
while disturbing least else?

Scoring is deliberately biased AGAINST elaboration. On this task the elaborate
option has lost every time it has been tried, and the measured ceiling for any
single change is +0.0022.
"""
from __future__ import annotations

import json
import logging

from lib.llm import complete

logger = logging.getLogger("amra.planner")

MAX_PLANS = 4

# Weights. `directness` and `fidelity` dominate because the observed failure is
# always the same: a patch that reads as the intervention and changes nothing.
# Reweighted after run 12. `capability` is new and heavily weighted because the
# measured failure was a COST MISJUDGEMENT, not a reasoning failure: the planner
# ranked the direct fix last on isolation 2 / cheapness 2, when the machinery it
# needed already existed. Cost drops to 0.08 because "how much typing" is the
# least informative axis once you know what the codebase already does.
# `fidelity` is held level with `directness` on purpose: "will this measurably
# move something" is the criterion that catches a patch reading as the
# intervention and doing nothing, which is the dominant failure in this project
# (contract satisfaction has run 45-88%). A plan that cannot name the quantity it
# moves must not win on sounding direct.
WEIGHTS = {"directness": 0.28, "fidelity": 0.28, "capability": 0.22,
           "isolation": 0.14, "cheapness": 0.08}


def score_plan(p: dict) -> float:
    """Weighted score from the criteria, each 1-5 as judged by the agent.

    Fidelity 1 is DISQUALIFYING, not merely penalised. If the plan cannot name a
    measurable quantity it moves, no semantic contract can be written for it, so
    it cannot be verified before training and a no-op would reach a full run
    undetected. That is the failure the contract mechanism exists to prevent, and
    no weighting on the other axes should be able to buy past it.
    """
    if float(p.get("fidelity", 0) or 0) <= 1:
        return 0.0
    return round(sum(WEIGHTS[k] * float(p.get(k, 0) or 0) for k in WEIGHTS), 3)


GROUPING_KEYS = {"group_size", "batch_mode"}
_OBJECTIVE_WORDS = ("listwise", "softmax", "loss_fn", "objective", "ranking loss",
                    "grouped loss", "pairwise", "bpr", "loss function")


def _inert_grouping(plan: dict, capabilities: str) -> bool:
    """Does this plan move grouping while leaving a non-grouped objective?

    That combination is a measured no-op, so it cannot test the hypothesis it
    appears to test. Detected structurally: the capability map reports whether
    the objective in force consumes the grouping.
    """
    if "is NOT a grouped objective" not in (capabilities or ""):
        return False                                # objective already grouped
    cfg = plan.get("config") or {}
    if not (GROUPING_KEYS & set(cfg)):
        return False                                # not a grouping plan
    text = f"{plan.get('how','')} {plan.get('control_point','')}".lower()
    return not any(w in text for w in _OBJECTIVE_WORDS)


def plan_implementations(idea: dict, problem: dict, instrument_keys: list[str],
                         model: str = "gpt-4o",
                         capabilities: str = "") -> tuple[dict, list[dict], int]:
    """Return (chosen_plan, all_plans_scored, tokens)."""
    prompt = f"""You have COMMITTED to this intervention. Do not reconsider it.

  HYPOTHESIS: {idea.get('hypothesis','')}
  MECHANISM:  {idea.get('why_it_should_work', idea.get('rationale',''))}
  PROBLEM:    {problem.get('statement','')}
  MODULE:     {idea.get('module','train.py')}

Your job is NOT to pick a better idea. It is to enumerate the different ways to
IMPLEMENT this one, and choose the way that most directly tests the mechanism.

These quantities can be measured on the patched code in 0.4s without training:
{', '.join(instrument_keys)}

{capabilities}

A MECHANISM MAY HAVE SEVERAL PARTS. If the map says a code path is inert
without a config key, or that one key is only read when another is set, or that
the objective currently in force cannot consume what you are changing, then a
plan touching that mechanism MUST include EVERY part in one plan -- the code
edit AND the config. A plan carrying only part of a mechanism changes nothing
measurable, or changes something the objective ignores; either way it tests
nothing, and it will be scored 0. Assemble the whole mechanism or choose a
different one.

A change that only needs to CALL a function that already exists, or to set a
config key that already selects an existing branch, is CHEAP -- however
architectural it sounds. Judge cost from the map above, not from how the change
sounds. Re-batching, re-grouping and swapping an objective are often config-level
here; a "small" bespoke reweighting can be a bigger edit than either.

Propose {MAX_PLANS} distinct implementations. For each, score 1-5:

  directness  does it change the DIAGNOSED VARIABLE itself, or something
              correlated with it? If the diagnosis is "training list length is
              6-8x evaluation", an intervention that alters list construction
              scores 5 and one that reweights the existing lists scores 2 --
              reweighting leaves the diagnosed quantity untouched.
  capability  is it supported by machinery already in the codebase (see the map
              above)? Calling an existing function or setting an existing config
              key scores 5. Requiring a new subsystem scores 1.
  fidelity    will it MEASURABLY move one of the quantities above? If you cannot
              name the quantity it moves, score 1. This is the criterion that
              catches a patch that reads correct and does nothing -- the single
              most common failure in this project.
  isolation   how few UNRELATED quantities does it disturb? Changing one thing
              scores 5; a rewrite that also alters capacity and batching scores 1.
  cheapness   1 = a large rewrite with many ways to be subtly wrong,
              5 = a few lines.

Be honest rather than generous: on this task the elaborate option has lost every
time, and no single change has ever gained more than +0.0022.

ACTIVATION CONFIG IS PART OF THE SAME INTERVENTION, NOT A SECOND CHANGE.
Some code paths are inert unless a config key selects them. Changing the code and
setting the key that switches it on is ONE mechanism and stays attributable --
the contract verifies the realised effect either way. Splitting them produces two
cycles that each measure nothing. If your plan needs a config key set to have any
effect, put it in "config".

Return ONLY JSON:
{{"plans": [{{"name": "<short>", "how": "<2-3 sentences, concrete>",
  "moves_quantity": "<which measurable quantity above it moves, and which way>",
  "control_point": "<module.py::function you would modify or call>",
  "config": {{}},
  "directness": 1-5, "capability": 1-5, "fidelity": 1-5,
  "isolation": 1-5, "cheapness": 1-5,
  "risk": "<the most likely way this silently does nothing>"}}]}}"""

    txt, tokens = complete("inventor", prompt, fallback_model=model,
                           max_tokens=1400, json_mode=True)
    plans = (json.loads(txt) or {}).get("plans", [])[:MAX_PLANS]
    if not plans:
        return {}, [], tokens

    # Enforce, do not merely state. The capability map already says grouping has
    # no effect on an objective that does not consume it -- and in 3/3 offline
    # trials the planner read that and still chose a grouping-only plan at 5.00,
    # because `capability` rewards using what exists and writing a new objective
    # scores low on it. So the scoring criterion introduced to fix run 12 steers
    # directly into a MEASURED no-op (handoff 04, 5b). A plan that moves grouping
    # while leaving a non-grouped objective in place would satisfy its contract,
    # score baseline, and be recorded as evidence AGAINST the correct hypothesis.
    for p in plans:
        if _inert_grouping(p, capabilities):
            p["fidelity"] = 1                      # disqualifying, see score_plan
            p["risk"] = ("grouping is inert under the current non-grouped "
                         "objective; this would test nothing. " + str(p.get("risk", "")))[:300]
            logger.info(f"    disqualified '{p.get('name','?')}': changes grouping "
                        f"but leaves a non-grouped objective in place")
        p["score"] = score_plan(p)
    plans.sort(key=lambda p: -p["score"])
    chosen = plans[0]

    logger.info(f"  {len(plans)} implementation plans considered:")
    for p in plans:
        mark = "  <- chosen" if p is chosen else ""
        cfg = p.get("config") or {}
        logger.info(f"    [{p['score']:.2f}] {str(p.get('name',''))[:44]:44s} "
                    f"d{p.get('directness')} k{p.get('capability')} "
                    f"f{p.get('fidelity')} i{p.get('isolation')} "
                    f"c{p.get('cheapness')}"
                    f"{' cfg=' + str(cfg) if cfg else ''}{mark}")
    return chosen, plans, tokens
