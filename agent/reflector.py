"""
Reflector: read the run result, decide keep or revert, and write the lesson.

The keep rule is arithmetic, not a judgement call: a change is kept only if the
run succeeded AND validation primary strictly beat the best seen so far. The LLM
is used only to explain WHY, never to decide -- letting a language model rule on
whether a number went up is how you end up keeping ten regressions in a row.

MARGIN exists because primary has a 5-seed std of 0.0008 on the official
baseline; a "gain" below that is noise being promoted to a result.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from lib.llm import complete

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution.scoring import BASELINE          # noqa: E402

logger = logging.getLogger("amra.reflector")

# One published std. Below this, a difference is not evidence.
MARGIN = 0.0008

# Below this a run is broken, not disproved.
RANDOM_FLOOR_PRIMARY = 0.4834

# A single change lifting primary by more than this is not plausible on this
# benchmark: the whole gap from baseline to the ORACLE ceiling is 0.247, and the
# organisers' own ablations move the number by <0.002. A jump this large means a
# leak or a broken evaluation, and it is exactly what a label leak looks like
# (observed: 0.6449 from reading is_click). Flag it rather than banking it.
IMPLAUSIBLE_GAIN = 0.02


def decide_keep(cycle: int,
                idea: dict,
                metrics: dict,
                error: str,
                status: str,
                led,
                jour,
                model: str = "gpt-4o-mini",
                best_primary: float = -1.0,
                history: list[dict] | None = None,
                config: dict | None = None,
                beliefs=None,
                observations: dict | None = None,
                failure_class: str = "scientific") -> tuple[bool, str, int]:
    """Returns (keep, lesson, tokens_used). The decision is made before the
    LLM call and is not influenced by it."""
    ok = status == "ok" and bool(metrics)
    primary = metrics.get("primary") if ok else None

    # If nothing has been kept yet, the incumbent is the official baseline.
    incumbent = best_primary if best_primary > 0 else BASELINE["valid"]["primary"]
    keep = bool(ok and primary is not None and primary > incumbent + MARGIN)

    suspicious = bool(keep and primary - incumbent > IMPLAUSIBLE_GAIN)
    if suspicious:
        logger.warning(
            f"IMPLAUSIBLE GAIN: {primary:.4f} is {primary - incumbent:.4f} above "
            f"the incumbent. The organisers' ablations move this metric by "
            f"<0.002 and the entire baseline-to-oracle gap is 0.247. Treat as a "
            f"suspected leak or broken evaluation until verified."
        )

    if ok:
        outcome = (
            f"primary {primary:.4f} (GAUC {metrics['GAUC']:.4f}, "
            f"nDCG@5 {metrics['nDCG@5']:.4f}) vs incumbent {incumbent:.4f} "
            f"-> {'improvement' if keep else 'no improvement beyond noise'}"
        )
        task = ("Explain in 2-3 sentences why this worked, and what it implies "
                "for the next iteration." if keep else
                "Diagnose in 2-3 sentences why this did NOT help. Be concrete "
                "about the mechanism -- 'needs tuning' is not a diagnosis.")
    else:
        outcome = f"run FAILED: {error.splitlines()[0][:200] if error else 'unknown'}"
        task = ("State the root cause in 2-3 sentences and what the next attempt "
                "must do differently to avoid it.")

    # The training curve separates "bad idea" from "bad learning rate". Without
    # it the Reflector guesses, and its lesson misdirects the next cycle.
    curve = ""
    if history:
        first, last = history[0], history[-1]
        best_ep = max(history, key=lambda h: h.get("primary", 0))
        curve = (
            f"\nTraining curve: {len(history)} epochs, "
            f"loss {first['loss']:.4f} -> {last['loss']:.4f}, "
            f"best primary {best_ep['primary']:.4f} at epoch {best_ep['epoch']}."
        )
        if len(history) <= 3:
            curve += (" Stopped very early -- suspect a learning rate mismatched "
                      "to this objective's gradient scale, not a bad idea.")
        elif best_ep["epoch"] == len(history):
            curve += " Still improving when it stopped -- suspect undertraining."
    if config:
        curve += f"\nConfig used: {config}"

    prompt = f"""An autonomous ML research loop just ran one experiment on
KuaiRand-Pure (within-user ranking, label long_view, primary = mean(GAUC, nDCG@5)).

Hypothesis: {idea.get('hypothesis', '')}
Technique: {idea.get('source_technique', '')}
Module edited: {idea.get('module', '')}
Predicted: {idea.get('predicted_effect', '')}
Actual: {outcome}{curve}

{task} Write plain prose, no preamble."""

    tokens = 0
    try:
        text, tokens = complete("reflector", prompt, fallback_model=model,
                                max_tokens=250)
        lesson = text.strip()
    except Exception as exc:                       # noqa: BLE001
        # The decision already stands; losing the explanation must not lose it.
        logger.warning(f"lesson generation failed: {exc}")
        lesson = outcome

    if suspicious:
        lesson = ("SUSPECTED LEAK / INVALID EVALUATION -- gain far exceeds what "
                  "this benchmark allows. Verify before trusting. " + lesson)

    # --- update the belief store ---------------------------------------
    # Dataset-specific conclusions accumulate HERE, not in techniques.jsonl.
    # The library holds transferable method knowledge; this holds what this run
    # has earned, with evidence on both sides.
    if beliefs is not None:
        try:
            _update_beliefs(beliefs, cycle, idea, metrics, ok, keep,
                            incumbent, observations, model,
                            failure_class=failure_class)
        except Exception as exc:                       # noqa: BLE001
            logger.warning(f"belief update failed: {exc}")

    label = f"{idea.get('source_technique', '?')} on {idea.get('module', '?')}"
    if keep:
        jour.add_insight(f"[cycle {cycle}] {label}: {lesson}")
    else:
        jour.add_dead_end(f"[cycle {cycle}] {label}: {lesson}")

    return keep, lesson, tokens


def classify_failure(status: str, contract_satisfied: bool | None,
                     metrics: dict, incumbent: float,
                     history: list[dict] | None) -> str:
    """Which KIND of failure was this? The distinction matters for beliefs.

    implementation -- the patch did not do what it claimed. The experiment says
                      NOTHING about the hypothesis. Must not weaken the belief.
    optimisation   -- the intervention exists but training misbehaved (diverged,
                      timed out, stopped in 1-2 epochs). Fix the training config,
                      not the idea.
    scientific     -- the contract held, training converged normally, and the
                      metric still did not improve. THIS is evidence against the
                      hypothesis, and the only kind that should weaken it.

    Before this existed, all three were recorded identically, so a broken
    implementation of a good idea wrote a false negative into beliefs.jsonl.
    Across one 25-cycle run the agent named the right problem 11 times; every
    failed patch argued against the thing it had correctly diagnosed.
    """
    if contract_satisfied is False:
        return "implementation"
    if status in ("timeout", "bug"):
        return "optimisation" if status == "timeout" else "implementation"
    primary = (metrics or {}).get("primary")
    if primary is None:
        return "implementation"
    if primary < RANDOM_FLOOR_PRIMARY:
        return "optimisation"                       # diverged, not disproved
    if history is not None and len(history) <= 2:
        return "optimisation"                       # stopped before it trained
    return "scientific"


def _update_beliefs(beliefs, cycle, idea, metrics, ok, keep, incumbent,
                    observations, model, failure_class="scientific") -> None:
    """Turn this cycle's outcome into evidence for or against a claim.

    Deliberately mechanical rather than another LLM call: the claim is the
    hypothesis that was actually tested, and the evidence is the number that came
    back. An LLM-authored belief would drift from what the experiment showed.
    """
    tech = idea.get("source_technique") or "unnamed"
    bid = f"tech:{tech}"
    primary = metrics.get("primary")

    if not ok:
        beliefs.assert_belief(
            belief_id=bid, claim=idea.get("hypothesis", "")[:400],
            confidence=0.5, status="untested", contradicting=[cycle],
            evidence=f"cycle {cycle}: run failed, no evidence either way")
        return

    delta = primary - incumbent
    cur = beliefs.current().get(bid, {})
    sup = list(cur.get("supporting_experiments", []))
    con = list(cur.get("contradicting_experiments", []))
    if keep:
        sup.append(cycle)
    elif failure_class == "scientific":
        con.append(cycle)
    else:
        # Implementation or optimisation failure: the hypothesis was never
        # actually tested. Record the attempt, argue neither way.
        beliefs.assert_belief(
            belief_id=bid, claim=idea.get("hypothesis", "")[:400],
            confidence=cur.get("confidence", 0.5),
            status=cur.get("status", "untested"),
            supporting=sup, contradicting=con,
            evidence=f"cycle {cycle}: {failure_class} failure — hypothesis not "
                     f"tested, belief unchanged")
        return

    # Confidence tracks the weight of evidence, not the size of one result.
    n = len(sup) + len(con)
    confidence = round(len(sup) / n, 2) if n else 0.5
    status = ("supported" if keep else
              ("contradicted" if len(con) >= 2 and not sup else "untested"))

    beliefs.assert_belief(
        belief_id=bid, claim=idea.get("hypothesis", "")[:400],
        confidence=confidence, status=status,
        supporting=sup, contradicting=con,
        evidence=f"cycle {cycle}: primary {primary:.4f} vs incumbent "
                 f"{incumbent:.4f} (delta {delta:+.4f})")

    # An observation the agent measured itself is worth recording separately --
    # it is the evidence trail for how a hypothesis was arrived at.
    for o in (observations or {}).get("observations", [])[:2]:
        what = str(o.get("what", ""))[:300]
        if what:
            beliefs.assert_belief(
                belief_id=f"obs:{abs(hash(what)) % 10**8}",
                claim=what, confidence={"high": 0.8, "medium": 0.5}.get(
                    o.get("confidence"), 0.3),
                status="untested",
                evidence=f"measured by the analyst at cycle {cycle}")
