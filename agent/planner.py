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
# Weights. `directness` and `fidelity` dominate because the observed failure is
# always the same: a patch that reads as the intervention and changes nothing.
WEIGHTS = {"directness": 0.40, "fidelity": 0.35, "isolation": 0.15, "cheapness": 0.10}


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


def plan_implementations(idea: dict, problem: dict, instrument_keys: list[str],
                         model: str = "gpt-4o") -> tuple[dict, list[dict], int]:
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

Propose {MAX_PLANS} distinct implementations. For each, score 1-5:

  directness  does it change the mechanism ITSELF, or something correlated with
              it? The smallest edit that makes the problem go away scores 5. A
              scheme that approaches the mechanism indirectly scores 2.
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

Return ONLY JSON:
{{"plans": [{{"name": "<short>", "how": "<2-3 sentences, concrete>",
  "moves_quantity": "<which measurable quantity above it moves, and which way>",
  "directness": 1-5, "fidelity": 1-5, "isolation": 1-5, "cheapness": 1-5,
  "risk": "<the most likely way this silently does nothing>"}}]}}"""

    txt, tokens = complete("inventor", prompt, fallback_model=model,
                           max_tokens=1400, json_mode=True)
    plans = (json.loads(txt) or {}).get("plans", [])[:MAX_PLANS]
    if not plans:
        return {}, [], tokens

    for p in plans:
        p["score"] = score_plan(p)
    plans.sort(key=lambda p: -p["score"])
    chosen = plans[0]

    logger.info(f"  {len(plans)} implementation plans considered:")
    for p in plans:
        mark = "  <- chosen" if p is chosen else ""
        logger.info(f"    [{p['score']:.2f}] {str(p.get('name',''))[:52]:52s} "
                    f"d{p.get('directness')} f{p.get('fidelity')} "
                    f"i{p.get('isolation')} c{p.get('cheapness')}{mark}")
    return chosen, plans, tokens
