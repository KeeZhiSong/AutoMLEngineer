"""
Classifier: turn measurements into a NAMED PROBLEM.

THE MISSING RUNG. In an unattended run the Analyst measured the decisive fact
three separate times and flagged it `[high]` confidence:

    "training set has a much higher mean list length (43.54) compared to the
     validation (5.58) and test (7.15) sets"

...and the loop never acted on it. It proposed a named technique card instead,
every time. The failure was not perception. It was that a FACT SUGGESTS NOTHING
while a CARD IS A READY-MADE ACTION, so the card won.

This step creates the abstraction an intervention can attach to. "train mean 43.5
vs eval mean 5.6" is a number. "train/eval distribution mismatch, dimension =
list length, ratio 6x" is a PROBLEM -- and a problem implies a class of fixes,
can be searched for in the literature, and can be argued about.

It deliberately does NOT propose solutions. Naming and solving are separate steps
precisely so that naming cannot be short-circuited by reaching for a known fix.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from lib.llm import complete

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("amra.classifier")

# Vocabulary of problem CLASSES. Deliberately about symptoms, not remedies --
# naming a class must not smuggle in its fix. A model may return "other" with
# its own wording; an unlisted class is a finding, not an error.
PROBLEM_CLASSES = [
    "train/eval distribution mismatch",
    "objective/metric mismatch",
    "label sparsity or class imbalance",
    "insufficient signal in the current inputs",
    "estimation variance / instability",
    "temporal drift between fit and evaluation",
    "population heterogeneity (the metric averages unlike users)",
    "optimisation failure (step size, convergence, divergence)",
    "capacity or expressiveness limit",
    "evaluation artefact (the number does not mean what it appears to)",
]


def classify(observations: dict,
             beliefs=None,
             led=None,
             model: str = "gpt-4o") -> tuple[dict, int]:
    """Name the problem the measurements point at. Returns (record, tokens)."""
    obs = (observations or {}).get("observations", [])
    if not obs:
        return {"problems": []}, 0

    obs_block = "\n".join(
        f"- [{o.get('confidence','?')}] {o.get('what','')}\n"
        f"    the analyst's guess at why: {o.get('why_it_might_matter','')}"
        for o in obs)

    belief_block = beliefs.context() if beliefs is not None else "(none)"

    tried = ""
    if led is not None and led.all():
        recent = [f"- {e.get('source_technique','?')}: "
                  f"{(e.get('conclusion') or '')[:120]}" for e in led.all()[-4:]]
        tried = "\n".join(recent)

    prompt = f"""You are the diagnostician on an ML research loop for KuaiRand-Pure.

TASK CONTRACT
- Within-user ranking over logged impressions. A quantity CONSTANT across one
  user's impressions cannot change that user's ordering at all.
- Label `long_view`. Metrics GAUC and nDCG@5; primary = their mean.
- Training uses the same users' impressions from an EARLIER date range.
- Current best is only +0.0018 over the reference. Seed noise is 0.0008. The
  total distance from the reference to a perfect oracle is 0.247.

WHAT WAS MEASURED THIS CYCLE
{obs_block}

WHAT THIS RUN ALREADY BELIEVES
{belief_block}

RECENT ATTEMPTS
{tried or "(none yet)"}

YOUR JOB
Name the PROBLEM these measurements point at. Not a solution -- a problem.

A good problem statement:
  · names a mechanism, not a symptom
  · says WHICH DIMENSION is affected and BY HOW MUCH, using the measured numbers
  · would still make sense to someone who had never seen this dataset
  · could be searched for in the literature

Bad:  "the model could be better on short lists"
Good: "train/eval distribution mismatch: the training objective operates on
       lists of median length 31 while the metric scores lists of median
       length 4, a 6x mismatch in the exact quantity the objective consumes"

Rules:
  · Ground every claim in a number that was actually measured. No speculation.
  · If the measurements support NO clear problem, say so and return an empty
    list. A cycle that honestly finds nothing is more useful than an invented
    problem.
  · Do NOT name a fix. Naming and solving are separate steps here.
  · Rank by how much of the remaining 0.247 you think the problem accounts for.

Problem classes to choose from (or "other" with your own wording):
{chr(10).join('  · ' + c for c in PROBLEM_CLASSES)}

Return ONLY JSON:
{{"problems": [
   {{"statement": "<the problem, with the numbers in it>",
     "problem_class": "<one of the classes above, or other>",
     "dimension": "<what specifically differs or is lacking>",
     "magnitude": "<the measured size of it>",
     "evidence": "<which measurement supports this>",
     "confidence": "high|medium|low",
     "why_it_could_matter_here": "<mechanism, tied to the task contract>"}}
 ]}}"""

    txt, tokens = complete("classifier", prompt, fallback_model=model,
                           max_tokens=1100, json_mode=True)
    parsed = json.loads(txt)
    problems = parsed.get("problems", [])

    for p in problems:
        logger.info(f"  problem [{p.get('confidence','?')}] "
                    f"({p.get('problem_class','?')}) "
                    f"{str(p.get('statement',''))[:120]}")
    if not problems:
        logger.info("  classifier: measurements support no clear problem")

    return {"problems": problems}, tokens
