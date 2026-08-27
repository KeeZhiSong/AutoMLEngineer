"""
Inventor: named problem -> candidate interventions, with literature on PULL.

Replaces the menu-driven Ideator. Three stages, and the ORDER is the whole point:

  1. INVENT    open-ended interventions for the named problem. No technique
               library is shown. The agent has read the recommender-systems
               literature in pretraining; it does not need a list of method
               names, and being handed one makes it pick rather than think.

  2. ASK       the agent MAY request literature relevant to its problem class.
               Pull, not push. Retrieval placed AFTER invention can only inform
               a hypothesis the agent already owns; placed before, it substitutes
               for having one.

  3. COMMIT    pick one, make it concrete and atomic, attach a kill criterion.

The previous design put a menu in front of the agent before it had formed a view,
and it measured the decisive fact three times without ever acting on it. This
inverts that.
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

logger = logging.getLogger("amra.inventor")

MODULES = ("features.py", "model.py", "train.py")

# Facts the agent cannot derive from the data itself, because they are the
# organisers' own published ablations. This is prior knowledge, NOT a menu of
# actions -- there is nothing here to pick.
GIVEN_FACTS = """
- Adding the full 13-field static feature set scores 0.5940 against 0.5950 for
  the current 5 fields. No gain. Measured by the organisers.
- Embedding dimension k = 8 / 16 / 32 scores 0.5895 / 0.5902 / 0.5887. Flat.
  Capacity is NOT the bottleneck.
- Pure user-side first-order terms contribute EXACTLY ZERO: ranking is computed
  within a user, so a term constant across that user's impressions cannot
  reorder them. Verified: item_pop x user_bias scores identically to item_pop.
- Seed standard deviation is 0.0008. Differences below that are not evidence.
- The oracle ceiling is 0.8484, not 1.0: 27% of evaluation users have no
  positive label, so their nDCG is 0 for any model.
- video_features_statistic_pure.csv is BANNED: its counters span the test
  window, so using it leaks test-period labels.
- Feedback columns (is_click, is_like, play_time_ms, ...) are TARGETS available
  to train.py via `enc.aux[...]`. Using one as a model INPUT is leakage and is
  rejected before the run.
"""


def _ask_literature(question: str, problem: dict, model: str) -> tuple[str, int]:
    """Answer the agent's own literature question from published work."""
    prompt = f"""An ML researcher working on within-user ranking (KuaiRand-Pure,
label long_view, metrics GAUC and nDCG@5) has diagnosed this problem:

  {problem.get('statement','')}
  class: {problem.get('problem_class','')}

They ask: "{question}"

Answer from the published recommender-systems and learning-to-rank literature.
Name real papers or methods and say WHAT MECHANISM each one uses to address this
class of problem. Be concrete about the mechanism, not just the name.

If the literature does not really address this, say so plainly -- a made-up
citation is worse than "this looks under-explored".

Keep to 200 words."""
    return complete("inventor", prompt, fallback_model=model, max_tokens=500)


def invent(problems: list[dict],
           beliefs=None,
           led=None,
           model: str = "gpt-4o") -> tuple[dict, int]:
    """Named problem -> one committed, atomic intervention. Returns (idea, tokens)."""
    if not problems:
        raise ValueError("no named problem to work from")

    problem = problems[0]
    tokens = 0

    belief_block = beliefs.context() if beliefs is not None else "(none)"
    tried = ""
    if led is not None and led.all():
        tried = "\n".join(
            f"- {e.get('source_technique','?')} -> "
            f"{(e.get('metrics') or {}).get('primary', 'failed')}"
            for e in led.all()[-6:])

    # ---- stage 1: invent, with NO library in front of the agent -------------
    invent_prompt = f"""You are an ML researcher. You have diagnosed this problem:

  PROBLEM:   {problem.get('statement','')}
  CLASS:     {problem.get('problem_class','')}
  DIMENSION: {problem.get('dimension','')}
  MAGNITUDE: {problem.get('magnitude','')}
  MECHANISM: {problem.get('why_it_could_matter_here','')}

FACTS YOU CANNOT DERIVE FROM THE DATA (the organisers measured these):
{GIVEN_FACTS}

WHAT THIS RUN BELIEVES SO FAR
{belief_block}

ALREADY ATTEMPTED
{tried or "(nothing yet)"}

You may edit exactly one of: features.py (what the model sees), model.py (the
scoring function), train.py (objective, optimiser, batching).

Propose 3 DIFFERENT interventions that would address this problem. Think from
the mechanism, not from a catalogue of method names. Two of them should be
things you could implement today in numpy; one may be more ambitious.

For each, state the mechanism by which it would change the metric. Remember the
metric is computed WITHIN a user, so anything constant across a user's
impressions does nothing.

Then say whether consulting the literature would change your ranking, and if so
what exactly you would want to look up.

Return ONLY JSON:
{{"interventions": [
   {{"what": "<the change, concretely>",
     "module": "<features.py|model.py|train.py>",
     "mechanism": "<why this moves the metric, given within-user ranking>",
     "expected_effect": "<signed number, calibrated to a 0.247 total headroom>",
     "risk": "<what could make it fail>"}}],
 "want_literature": true,
 "literature_question": "<what you would look up, or empty>"}}"""

    txt, t = complete("inventor", invent_prompt, fallback_model=model,
                      max_tokens=1400, json_mode=True)
    tokens += t
    stage1 = json.loads(txt)
    candidates = stage1.get("interventions", [])
    if not candidates:
        raise ValueError("inventor produced no candidates")

    for c in candidates:
        logger.info(f"  candidate: [{c.get('module','?')}] {str(c.get('what',''))[:100]}")

    # ---- stage 2: literature, only if the agent asked for it ---------------
    lit = ""
    question = (stage1.get("literature_question") or "").strip()
    if stage1.get("want_literature") and question:
        logger.info(f"  PULL literature: {question[:110]}")
        lit, lit_tokens = _ask_literature(question, problem, model)
        tokens += lit_tokens
    else:
        logger.info("  agent did not request literature this cycle")

    # ---- stage 3: commit to one -------------------------------------------
    commit_prompt = f"""Your three candidate interventions for the problem:

  {problem.get('statement','')}

{json.dumps(candidates, indent=2)}

{"LITERATURE YOU ASKED FOR:" + chr(10) + lit if lit else "You chose not to consult the literature."}

Pick ONE and make it precise enough to implement. Rules:
  · Change ONE thing. A bundled change cannot be attributed.
  · It must plausibly move a WITHIN-USER ordering.
  · Give a kill criterion: the result that would tell us this is wrong, so we
    do not burn revisions on it.
  · Calibrate the prediction. Seed noise is 0.0008 and the organisers' own
    ablations move this metric by less than 0.002. A prediction of +0.05 means
    you have found a leak, not a method.

Return ONLY JSON:
{{"hypothesis": "<one sentence: the change and what it should do>",
  "rationale": "<why this one, given the problem and any literature>",
  "source_technique": "<a short slug naming the idea, e.g. eval-length-groups>",
  "grounded_in": "<the measurement or the problem statement this rests on>",
  "module": "<features.py|model.py|train.py>",
  "predicted_effect": "+0.00X primary",
  "kill_criterion": "<the result that would end this line of attack>",
  "used_literature": {str(bool(lit)).lower()},
  "config": {{"lr": 0.001}}}}"""

    txt2, t2 = complete("inventor", commit_prompt, fallback_model=model,
                        max_tokens=800, json_mode=True)
    tokens += t2
    idea = json.loads(txt2)

    if idea.get("module") not in MODULES:
        idea["module"] = candidates[0].get("module", "train.py")
    idea["problem"] = problem
    idea["candidates_considered"] = candidates
    idea["literature"] = lit[:2000]
    return idea, tokens
