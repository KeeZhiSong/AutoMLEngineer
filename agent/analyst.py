"""
Analyst: measure the data, then say what you notice.

The step that was missing. Previously the Ideator picked from a technique library
stocked by a human, so its justification was always "a card told me". Now it can
interrogate the dataset first and ground a hypothesis in something it measured.

Two stages, deliberately:
  1. CHOOSE which tools to call, from a menu, given what is already known.
  2. INTERPRET the returned numbers into observations.

Stage 1 keeps it cheap and makes the choice of what to investigate part of the
agent's job. Stage 2 is where the research actually happens -- the tools return
neutral numbers (solution/eda.py enforces this), so noticing which ones matter is
the agent's contribution, not the tool author's.
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

from solution import eda  # noqa: E402

logger = logging.getLogger("amra.analyst")

MAX_TOOLS_PER_CYCLE = 3


def _run_tools(ds, calls: list[dict]) -> dict:
    """Execute the chosen tools, tolerating a bad name or argument."""
    out = {}
    for call in calls[:MAX_TOOLS_PER_CYCLE]:
        name = call.get("tool") if isinstance(call, dict) else str(call)
        fn = eda.TOOLS.get(name)
        if fn is None:
            out[str(name)] = {"error": f"no such tool; available: {list(eda.TOOLS)}"}
            continue
        try:
            if name == "feedback_label_relationship":
                out[name] = fn(ds, call.get("column", "is_click"))
            else:
                out[name] = fn(ds)
        except Exception as exc:                      # noqa: BLE001
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def observe(ds, led, jour, model: str = "gpt-4o") -> tuple[dict, int]:
    """Measure, then interpret. Returns (observation_record, tokens_used)."""
    tokens = 0

    prior = [e.get("conclusion", "") for e in led.all()[-4:] if e.get("conclusion")]
    prior_text = "\n".join(f"- {p[:160]}" for p in prior) or "(nothing yet)"

    # --- stage 1: choose what to measure -------------------------------
    pick_prompt = f"""You are the analyst on an ML research loop for KuaiRand-Pure
(within-user ranking of logged impressions, label `long_view`, scored by
mean(GAUC, nDCG@5)).

Available measurement tools:
{eda.describe_tools()}

What the loop has concluded so far:
{prior_text}

Choose up to {MAX_TOOLS_PER_CYCLE} tools to run THIS cycle. Pick what could
change your mind about where the model is losing points -- not what merely
confirms what is already known.

Return ONLY JSON: {{"calls": [{{"tool": "<name>", "column": "<only for feedback_label_relationship>"}}],
 "why": "one sentence on what you are looking for"}}"""

    txt, t = complete("analyst", pick_prompt, fallback_model=model,
                      max_tokens=350, json_mode=True)
    tokens += t
    plan = json.loads(txt)
    calls = plan.get("calls", [])
    logger.info(f"analyst measuring: {[c.get('tool') for c in calls if isinstance(c, dict)]}")

    measurements = _run_tools(ds, calls)

    # --- stage 2: interpret --------------------------------------------
    read_prompt = f"""You measured the KuaiRand-Pure dataset. Here are the raw numbers.

{json.dumps(measurements, indent=2)}

Task contract, for context:
- Within-user ranking over logged impressions. A quantity that is CONSTANT across
  one user's impressions cannot change that user's ordering at all.
- Label `long_view`. Metrics GAUC and nDCG@5; primary = their mean.
- Training uses these same users' impressions from an earlier date range.

Write 2-4 OBSERVATIONS. An observation states something you can see in these
numbers and why it might matter for the model. Be specific and quantitative --
cite the numbers. If a number surprises you, say so. If nothing here looks
actionable, say that instead of inventing something.

Do NOT propose a fix. Observation only.

Return ONLY JSON:
{{"observations": [{{"what": "<the measured fact, with numbers>",
                    "why_it_might_matter": "<mechanism>",
                    "confidence": "high|medium|low"}}]}}"""

    txt2, t2 = complete("analyst", read_prompt, fallback_model=model,
                        max_tokens=800, json_mode=True)
    tokens += t2
    parsed = json.loads(txt2)

    record = {
        "tools_run": list(measurements.keys()),
        "why": plan.get("why", ""),
        "measurements": measurements,
        "observations": parsed.get("observations", []),
    }
    for o in record["observations"]:
        logger.info(f"  obs [{o.get('confidence','?')}] {str(o.get('what',''))[:120]}")
    return record, tokens
