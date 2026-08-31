"""
Ideator: propose the next hypothesis, grounded in the technique library and the
run history.

Retrieval is deliberately biased by the organisers' own published ranking
(`organizer_ranked` in techniques.jsonl): they measured which directions are dead
and which are unexplored, so an agent that ignores that is burning iterations to
rediscover known negatives. Cards flagged `dead_end` are never offered as
proposals -- they are injected as explicit prohibitions instead.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

from lib.llm import complete

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import ledger as ledger_module          # noqa: E402
from solution.scoring import BASELINE            # noqa: E402

logger = logging.getLogger("amra.ideator")

TECHNIQUES_PATH = ROOT / "lib" / "techniques.jsonl"
MODULES = ("features.py", "model.py", "train.py")

# How many times the loop may revise one idea before it must move on.
#
# MEASURED, across 20 runs: 93 revision cycles were attempted and **none was ever
# accepted**. The best a revision ever scored was 0.6019, below the 0.6031
# incumbent it was trying to beat. They consumed 675,815 tokens, 16% of the
# project's entire spend, for zero accepted results.
#
# Set to 0 on that evidence. The anomaly board now does the job revisions were
# meant to do: a near miss stays on the board with its attempt history, so the
# idea remains available to a later cycle without re-running the whole loop
# against a patch that already failed.
#
# Raise it if you want the mechanism back; nothing else depends on it being 0.
MAX_REVISIONS = 0

# A result within this much of the incumbent is worth revising rather than
# discarding. Wider than the noise margin -- a near miss is a tuning problem.
PROMISING_GAP = 0.010


def load_techniques() -> tuple[list[dict], list[dict]]:
    """Returns (candidate techniques, dead-end prohibitions)."""
    live, dead = [], []
    with open(TECHNIQUES_PATH) as fh:
        for line in fh:
            if not line.strip():
                continue
            card = json.loads(line)
            (dead if card.get("dead_end") else live).append(card)
    return live, dead


def _select(live: list[dict], tried: set[str], k: int = 5) -> list[dict]:
    """Prefer untried, organiser-highly-ranked techniques.

    Sorts by (already tried, priority) so unexplored high-priority work surfaces
    first, then keeps a little randomness so the loop can escape a rut.
    """
    ranked = sorted(live, key=lambda c: (c["id"] in tried, c.get("priority", 9)))
    head = ranked[:k]
    tail = [c for c in ranked[k:] if c["id"] not in tried]
    if tail and random.random() < 0.25:
        head[-1] = random.choice(tail)
    return head


def propose_idea(led, jour, model: str = "gpt-4o-mini",
                 observations: dict | None = None,
                 beliefs=None) -> tuple[dict, int]:
    """Produce one atomic, testable idea. Returns (idea, tokens_used)."""
    live, dead = load_techniques()
    entries = led.all()
    tried = set(led.tried_techniques()) if entries else set()

    best = ledger_module.best_metrics(entries) if entries else {}
    best_primary = best.get("primary")

    candidates = _select(live, tried)

    history_lines = []
    for e in entries[-6:]:
        m = e.get("metrics") or {}
        outcome = (f"primary {m['primary']:.4f}" if m.get("primary")
                   else f"FAILED ({(e.get('error') or '')[:70]})")
        history_lines.append(
            f"- cycle {e.get('cycle')}: {e.get('source_technique', '?')} "
            f"on {e.get('module_changed', '?')} -> {outcome}")
    history = "\n".join(history_lines) or "(no cycles yet)"

    forbidden = "\n".join(
        f"- {c['name']}: {c['key_insight']} (measured: {c['when']})" for c in dead)

    catalogue = "\n\n".join(
        f"### {c['id']} -- {c['name']}  [edit {c['module']}]\n"
        f"Problem: {c['problem']}\n"
        f"Approach: {c['approach']}\n"
        f"When: {c['when']}\n"
        f"Insight: {c['key_insight']}\n"
        f"Source: {c['paper']}"
        + (f"\nCAUTION: {c['caution']}" if c.get("caution") else "")
        for c in candidates)

    # Observations the analyst measured THIS cycle. An idea grounded in one of
    # these is worth more than one grounded only in a library card: the card is
    # prior knowledge someone else wrote down, the observation is evidence this
    # run produced.
    obs_block = "(no measurements taken this cycle)"
    if observations and observations.get("observations"):
        obs_block = "\n".join(
            f"- [{o.get('confidence','?')}] {o.get('what','')}\n"
            f"    possible mechanism: {o.get('why_it_might_matter','')}"
            for o in observations["observations"])

    belief_block = beliefs.context() if beliefs is not None else "(none)"

    prompt = f"""You are directing an autonomous ML research loop on KuaiRand-Pure.

## Task contract (fixed, not negotiable)
- Within-user ranking over logged impressions. No full-catalogue retrieval.
- Relevance label: `long_view` (0/1).
- Metrics: GAUC and nDCG@5; **primary = mean of the two**.
- Official FM baseline to beat: valid primary {BASELINE['valid']['primary']}.
- Oracle ceiling is 0.8484 on valid, NOT 1.0 -- 27% of users are all-negative.

## Current state
Best validation primary so far: {best_primary if best_primary else 'none yet (baseline is the starting point)'}

Recent cycles:
{history}

## Measured dead ends -- proposing any of these is a wasted iteration
{forbidden}

## What the analyst MEASURED this cycle
{obs_block}

## What this run currently believes (with evidence weight)
{belief_block}

## Candidate techniques from the library (prior knowledge, not findings)
{catalogue}

## Your job
Propose ONE atomic, testable change. You may ground it EITHER in a measurement
above OR in a library technique -- and an idea grounded in something you measured
is preferred, because the library is prior knowledge someone else wrote down
while the measurement is evidence from this dataset. If a measurement contradicts
a library card, trust the measurement and say so.

Rules:
- Change one thing. A hypothesis that bundles three changes cannot be attributed.
- Say which module to edit: exactly one of features.py, model.py, train.py.
- Predict the effect on validation primary as a signed number, and be honest --
  a wrong prediction is informative, a vague one is not.
- Do not propose anything on the dead-end list.
- Prefer a technique not already tried, unless you have a specific reason to
  revisit one (say what changed).

Return ONLY a JSON object:
{{"hypothesis": "one sentence: what change, and what it should do to primary",
  "rationale": "why this is the right next step given the state above",
  "source_technique": "<technique id from the catalogue, or observation:<short-slug> if grounded in a measurement>",
  "grounded_in": "<the specific measurement or card this rests on>",
  "module": "<features.py|model.py|train.py>",
  "predicted_effect": "+0.008 primary",
  "config": {{"lr": 0.001, "batch_mode": "row"}}}}

`config` overrides training hyperparameters for this experiment. Set
`batch_mode` to "user" for any within-user ranking loss (row batching fragments
the lists), and set `lr` deliberately -- 1e-3 was tuned for pointwise logloss and
a grouped loss has a different gradient scale."""

    txt, tokens = complete("inventor", prompt, fallback_model=model,
                           max_tokens=700, json_mode=True)
    idea = json.loads(txt)

    # Trust the technique card's module over the model's free-text choice.
    by_id = {c["id"]: c for c in live}
    card = by_id.get(idea.get("source_technique", ""))
    if card:
        # A library card knows which module it belongs in; trust it over the
        # model's free-text choice. An observation-grounded idea has no card,
        # so its own module choice stands.
        idea["module"] = card["module"]
    if idea.get("module") not in MODULES:
        idea["module"] = "train.py"

    dead_ids = {c["id"] for c in dead}
    if idea.get("source_technique") in dead_ids:
        raise ValueError(
            f"Ideator proposed a known dead end: {idea['source_technique']}")

    return idea, tokens


def propose_revision(candidate: dict, model: str = "gpt-4o-mini") -> tuple[dict, int]:
    """Propose a targeted fix to a promising-but-rejected change.

    The loop's job is not to sample 20 techniques once each -- the organisers'
    Figure 1 puts "reflect + revise" in the cycle, and a human MLE tunes a near
    miss rather than discarding it. This is that step: same technique, same
    module, one specific change aimed at the diagnosed weakness.
    """
    m = candidate["metrics"]
    idea = candidate["idea"]
    gap = candidate["incumbent"] - m["primary"]

    prompt = f"""An experiment in an autonomous ML research loop came CLOSE but did
not beat the incumbent. Your job is to revise it, not to replace it.

## What was tried
Technique: {idea.get('source_technique')}
Module: {candidate['module']}
Hypothesis: {idea.get('hypothesis')}
Config used: {candidate.get('config')}

## Result
GAUC {m['GAUC']:.4f} | nDCG@5 {m['nDCG@5']:.4f} | primary {m['primary']:.4f}
Incumbent primary: {candidate['incumbent']:.4f}  (short by {gap:.4f})
Revision attempt {candidate['attempts'] + 1} of {MAX_REVISIONS}.

## Diagnosis from the last cycle
{candidate.get('lesson', '')}

## Known traps already accounted for
- A listwise softmax target must be normalised over the user's positives.
- `batch_mode="user"` keeps each user's impressions in one batch; row batching
  leaves only ~19% of rows in a mixed-label group.
- lr=1e-3 was tuned for pointwise logloss, not for a grouped objective.

## Your job
Name ONE specific change to close the gap. Prefer, in order:
1. A hyperparameter that the diagnosis implicates (learning rate, batch mode,
   temperature, regularisation, number of sampled pairs).
2. A correctness fix in how the objective is computed.
3. A small structural refinement of the same technique.

Do NOT switch to a different technique -- that is what the propose path is for.

Return ONLY JSON:
{{"hypothesis": "one sentence: the specific change and why it closes the gap",
  "rationale": "what in the diagnosis points at this",
  "source_technique": "{idea.get('source_technique')}",
  "module": "{candidate['module']}",
  "predicted_effect": "+0.004 primary",
  "config": {{"lr": 0.001, "batch_mode": "user"}}}}"""

    txt, tokens = complete("inventor", prompt, fallback_model=model,
                           max_tokens=600, json_mode=True)
    revision = json.loads(txt)

    # A revision must stay on the same technique and module, whatever it says.
    revision["source_technique"] = idea.get("source_technique", "")
    revision["module"] = candidate["module"]
    revision["is_revision"] = True
    return revision, tokens
