"""
beliefs.py -- accumulated dataset-specific conclusions, with evidence.

WHY THIS IS SEPARATE FROM techniques.jsonl. Two kinds of knowledge, two
lifetimes, two stores:

  techniques.jsonl  PRIOR / GENERAL method knowledge. "Listwise losses depend on
                    list structure." Transferable, permanent, human-curated. The
                    agent READS it and never writes to it.

  beliefs.jsonl     DATASET-SPECIFIC conclusions this run has earned.
                    "Training groups of ~5 beat full user lists on KuaiRand-Pure
                    (paired t=11.3, n=5)." The agent WRITES these as evidence
                    accumulates.

Collapsing them breaks three things at once: a clean autonomy replay stops being
clean because the library now carries answers; a wrong inference can be promoted
to permanent fact; and there is nowhere to record evidence that CONTRADICTS a
claim, so beliefs can only ever strengthen.

A belief is never deleted. It is contradicted, and keeps both sides of its
evidence, because "we believed X until experiment 14" is itself a finding.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("amra.beliefs")

STATUSES = ("untested", "supported", "contradicted", "superseded")


class BeliefStore:
    """Append-only belief log. Current state is the last entry per id."""

    def __init__(self, workspace: Path | str, filename: str = "beliefs.jsonl"):
        self.path = Path(workspace) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def assert_belief(self, *, belief_id: str, claim: str,
                      confidence: float = 0.5,
                      status: str = "untested",
                      supporting: Optional[list] = None,
                      contradicting: Optional[list] = None,
                      evidence: str = "") -> Optional[dict]:
        """Record or update a claim. Never raises -- instrumentation must not be
        able to kill the loop it instruments."""
        entry = {
            "ts": time.time(),
            "id": str(belief_id)[:80],
            "claim": str(claim)[:600],
            "confidence": max(0.0, min(1.0, float(confidence))),
            "status": status if status in STATUSES else "untested",
            "supporting_experiments": list(supporting or []),
            "contradicting_experiments": list(contradicting or []),
            "evidence": str(evidence)[:800],
        }
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:                       # noqa: BLE001
            logger.warning(f"could not append belief: {exc}")
            return None
        return entry

    def all_entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if isinstance(d, dict):
                    out.append(d)
            except json.JSONDecodeError:
                continue
        return out

    def current(self) -> dict[str, dict]:
        """Latest state of each belief, by id."""
        state: dict[str, dict] = {}
        for e in self.all_entries():
            state[e["id"]] = e
        return state

    def context(self, limit: int = 8) -> str:
        """Compact block for an LLM prompt, most confident first.

        Contradicted beliefs are included on purpose: knowing what was ruled out
        is as useful as knowing what holds, and it stops the loop re-proposing
        something already disproved.
        """
        cur = list(self.current().values())
        if not cur:
            return "(no beliefs recorded yet)"
        cur.sort(key=lambda b: (b["status"] != "supported", -b["confidence"]))
        lines = []
        for b in cur[:limit]:
            n_s, n_c = len(b["supporting_experiments"]), len(b["contradicting_experiments"])
            lines.append(
                f"- [{b['status']}, conf {b['confidence']:.2f}, "
                f"{n_s} for / {n_c} against] {b['claim']}"
            )
        return "\n".join(lines)
