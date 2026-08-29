"""
Anomaly board: named problems that PERSIST across cycles.

WHY THIS EXISTS. In run 11 the agent named the decisive problem -- training
ranking lists ~31 items against evaluation lists ~4 -- in cycles 6, 11, 12 and
16, and each time it was a fresh discovery. CLASSIFY is re-derived every cycle
from whatever OBSERVE happened to sample, so a correct diagnosis had no memory:
noticed, failed, forgotten, re-noticed. Four independent sightings produced four
unrelated interventions and zero follow-through.

The board keeps an unresolved list, raises confidence when an anomaly is seen
again by an independent measurement, and records what has been TRIED against it.
That lets the inventor be asked the question it could never previously answer:
"which high-confidence anomaly has not yet received a faithful intervention?"

An attempt only counts as FAITHFUL if its semantic contract was satisfied -- a
patch that changed nothing is not evidence about the anomaly, and must not
retire it. This is the same distinction the failure classifier draws between
implementation and scientific failure.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _klass(problem: dict) -> str:
    return re.sub(r"[^a-z ]", "", str(problem.get("problem_class", "")).lower()).strip()


def _numbers(problem: dict) -> set[str]:
    return set(re.findall(r"\d+\.?\d*", str(problem.get("statement", ""))))


_STOP = {"the", "a", "an", "of", "in", "to", "and", "is", "are", "for", "with",
         "that", "this", "it", "as", "on", "by", "be", "which", "could", "may",
         "set", "has", "have", "at", "from", "than", "while", "only", "very"}


def _words(problem_or_text) -> set[str]:
    t = (problem_or_text if isinstance(problem_or_text, str)
         else str(problem_or_text.get("statement", "")))
    return {w for w in re.findall(r"[a-z_]{3,}", t.lower()) if w not in _STOP}


def _same_anomaly(problem: dict, entry: dict) -> bool:
    """Is this a re-sighting of `entry`, or a genuinely different anomaly?

    Class alone is too coarse -- run 11 filed a list-length mismatch AND a
    temporal drift under `train/eval distribution mismatch`. Exact number match
    is too strict -- the same anomaly was described as "train 43.54 vs valid
    5.58" in one cycle and "63.7% of users have at most 5 interactions" in
    another, which are the same fact from two angles. Same class plus any shared
    statistic is the rule that merges those two and still separates the drift.
    """
    if _klass(problem) != entry.get("klass"):
        return False
    if _numbers(problem) & set(entry.get("numbers", [])):
        return True
    # Same class, no shared statistic: fall back to wording overlap. Run 12
    # produced 14 entries from 16 cycles -- five of them "population
    # heterogeneity" describing one thing in different words, with no number in
    # common. Jaccard is deliberately strict (0.6) so a genuinely different
    # anomaly in the same class still gets its own entry.
    a, b = _words(problem), _words(entry.get("statement", ""))
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.6


class AnomalyBoard:
    """Persistent per-workspace record of unresolved observations."""

    def __init__(self, workspace: Path | str):
        self.path = Path(workspace) / "anomalies.jsonl"
        self.items: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    a = json.loads(line)
                    self.items[a["id"]] = a

    def _save(self) -> None:
        with self.path.open("w") as fh:
            for a in self.items.values():
                fh.write(json.dumps(a) + "\n")

    def observe(self, problems: list[dict], cycle: int) -> list[str]:
        """Record this cycle's named problems, merging repeats.

        Returns the anomaly id each problem landed on, in order, so the caller
        can attribute an attempt to the anomaly it actually targeted. Without
        this, run 12 attached all 11 of its attempts to whichever entry happened
        to be most active, making "what has been tried" unreliable -- half the
        point of the board.
        """
        ids: list[str] = []
        for p in problems or []:
            if not _klass(p) or not _numbers(p):
                ids.append("")
                continue
            a = next((e for e in self.items.values() if _same_anomaly(p, e)), None)
            if a is None:
                key = f"{_klass(p)}|{len(self.items)}"
                self.items[key] = {
                    "id": key,
                    "klass": _klass(p),
                    "numbers": sorted(_numbers(p)),
                    "statement": str(p.get("statement", ""))[:300],
                    "problem_class": p.get("problem_class", ""),
                    "sightings": 1,
                    "first_seen": cycle,
                    "last_seen": cycle,
                    "attempts": [],
                    "status": "unresolved",
                }
            else:
                # Independent re-sighting raises confidence; it does not create
                # a second entry, which is the bug this class exists to fix.
                a["sightings"] += 1
                a["last_seen"] = cycle
                a["numbers"] = sorted(set(a["numbers"]) | _numbers(p))
                if len(str(p.get("statement", ""))) > len(a["statement"]):
                    a["statement"] = str(p.get("statement", ""))[:300]
            ids.append(a["id"] if a is not None else key)
        self._save()
        return ids

    def record_attempt(self, cycle: int, technique: str, faithful: bool,
                       outcome: str, anomaly_id: str | None = None) -> None:
        """Attach an attempt to the anomaly it targeted.

        `faithful` is the contract verdict. An unfaithful attempt is logged so
        the agent can see the idea was never actually tested, but it does not
        count toward retiring the anomaly.
        """
        target = anomaly_id if anomaly_id in self.items else self._most_active()
        if target is None:
            return
        a = self.items[target]
        a["attempts"].append({"cycle": cycle, "technique": technique,
                              "faithful": bool(faithful), "outcome": outcome[:160]})
        if faithful and outcome == "kept":
            a["status"] = "resolved"
        self._save()

    def _most_active(self) -> str | None:
        live = [a for a in self.items.values() if a["status"] == "unresolved"]
        if not live:
            return None
        return max(live, key=lambda a: (a["sightings"], a["last_seen"]))["id"]

    def confidence(self, a: dict) -> float:
        """Repeated independent sightings raise it; faithful failures lower it."""
        c = min(0.5 + 0.15 * (a["sightings"] - 1), 0.95)
        tried = sum(1 for x in a["attempts"] if x["faithful"])
        return round(max(c - 0.1 * tried, 0.05), 2)

    def unresolved(self, min_sightings: int = 1) -> list[dict]:
        out = [a for a in self.items.values()
               if a["status"] == "unresolved" and a["sightings"] >= min_sightings]
        return sorted(out, key=lambda a: -self.confidence(a))

    def render(self, limit: int = 5) -> str:
        """The board, as the inventor sees it."""
        live = self.unresolved()[:limit]
        if not live:
            return "(no unresolved anomalies on the board yet)"
        lines = []
        for a in live:
            faithful = sum(1 for x in a["attempts"] if x["faithful"])
            unfaithful = len(a["attempts"]) - faithful
            tried = ", ".join(sorted({x["technique"] for x in a["attempts"]})) or "nothing yet"
            lines.append(
                f"- [{self.confidence(a):.2f}] {a['statement'][:190]}\n"
                f"    seen {a['sightings']}x (cycles {a['first_seen']}-{a['last_seen']}); "
                f"faithful attempts {faithful}, no-ops {unfaithful}; tried: {tried}")
        return "\n".join(lines)
