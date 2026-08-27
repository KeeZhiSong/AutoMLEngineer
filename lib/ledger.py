"""
Experiment ledger + research journals for the Autonomous ML Research Agent.

Design (adapted from the append-only pattern in auto-deep-researcher-24x7,
specialised for the KuaiRand-Pure ranking track):

  * ExperimentLedger  -> workspace/experiments.jsonl
        One JSON object per research cycle. Append-only, so it survives a
        crash mid-run, never needs a parse-rewrite, and stays human- and
        tool-readable at zero LLM cost. record() NEVER raises: instrumentation
        must not be able to kill the loop it is instrumenting.

        This file IS the "Run & Iteration Logs" deliverable the track asks for:
        hypothesis, code diff, metrics, error/recovery events, plus the
        resource accounting (tokens + GPU-hours) the Feasibility score needs.

  * Journal (DEAD_ENDS.md / INSIGHTS.md)
        Append-only markdown the agent reads the tail of each cycle, so it
        stops re-proposing failed ideas and keeps durable lessons in view.
        This is the "lessons-learned memory feeding step (1)" loop.

  * Cheap pure-Python readers (best_metrics, detect_stagnation, ...) turn the
    raw trajectory into the signals the controller uses to decide keep/revert,
    when to stop (convergence), and when to force a new idea category.

No third-party dependencies. Standard library only.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("amra.ledger")


def _jsonable(o):
    """Coerce numpy scalars/arrays (and anything else exotic) for json.dumps.

    The official scorer returns numpy floats; json.dumps rejects them outright,
    which previously took down a whole run at the logging step.
    """
    if hasattr(o, "item"):          # numpy scalar
        try:
            return o.item()
        except Exception:           # noqa: BLE001
            pass
    if hasattr(o, "tolist"):        # numpy array
        try:
            return o.tolist()
        except Exception:           # noqa: BLE001
            pass
    return str(o)

# The metrics KuaiRand-Pure is scored on. Higher is better for all three.
# primary = mean(GAUC, nDCG@5) and is the quantity the convergence rule reads.
METRIC_KEYS = ("GAUC", "nDCG@5", "primary")


class ExperimentLedger:
    """Append-only JSONL record of every research cycle."""

    def __init__(self, workspace: Path | str, filename: str = "experiments.jsonl"):
        self.path = Path(workspace) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        cycle: int,
        parent_cycle: Optional[int] = None,   # tree structure: which node this branched from
        hypothesis: str = "",                 # what the agent decided to try, and why
        source_technique: str = "",           # which retrieved method it drew on (RAG evidence)
        predicted_effect: str = "",           # its stated prediction, for the "did it come true?" check
        stage: str = "",                      # draft | improve | debug
        module_changed: str = "",             # e.g. "model.py" — which pipeline module was edited
        code_diff: str = "",                  # the actual change applied
        status: str = "",                     # ok | bug | timeout | diverged | reverted
        metrics: Optional[dict] = None,       # {"GAUC": .., "nDCG@5": .., "primary": ..} on valid
        data_scale: str = "",                 # "subsample" | "full" — inner-loop vs final run
        error: str = "",                      # captured traceback summary if it failed
        recovered: Optional[bool] = None,     # did the agent fix its own error this cycle?
        manual_intervention: bool = False,    # AUTONOMY signal: did a human step in? aim for False
        tokens_in: int = 0,                   # RESOURCE: LLM input tokens this cycle
        tokens_out: int = 0,                  # RESOURCE: LLM output tokens this cycle
        gpu_seconds: float = 0.0,             # RESOURCE: GPU time spent training/evaluating
        conclusion: str = "",                 # short takeaway written after the run
        observations: Optional[dict] = None,   # what the analyst measured and noticed
                                              # -- the evidence for Innovation scoring
        ts: Optional[float] = None,
    ) -> Optional[dict]:
        """Append one cycle's outcome. Never raises — a logging failure must
        not crash the research loop."""
        entry = {
            "ts": time.time() if ts is None else float(ts),
            "cycle": int(cycle),
            "parent_cycle": parent_cycle,
            "stage": str(stage or ""),
            "status": str(status or ""),
            "hypothesis": str(hypothesis or "")[:800],
            "source_technique": str(source_technique or "")[:200],
            "predicted_effect": str(predicted_effect or "")[:300],
            "module_changed": str(module_changed or ""),
            # NOT truncated to a few KB: this file IS the "Run & Iteration
            # Logs" deliverable, and a diff cut mid-expression makes the
            # experiment unreproducible for anyone reading it (it also broke a
            # replay of our own). 200KB is far above any single module.
            "code_diff": str(code_diff or "")[:200_000],
            "metrics": {k: v for k, v in (metrics or {}).items()},
            "data_scale": str(data_scale or ""),
            "error": str(error or "")[:2000],
            "recovered": recovered,
            "manual_intervention": bool(manual_intervention),
            "tokens_in": int(tokens_in),
            "tokens_out": int(tokens_out),
            "gpu_seconds": float(gpu_seconds),
            "conclusion": str(conclusion or "")[:500],
            "observations": observations or {},
        }
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                # default=_jsonable because metrics arrive as numpy scalars from
                # the scorer, and json refuses float32. Instrumentation must not
                # be able to kill the loop it is instrumenting.
                fh.write(json.dumps(entry, ensure_ascii=False,
                                    default=_jsonable) + "\n")
        except Exception as exc:  # noqa: BLE001 — never crash the loop
            logger.warning(f"Failed to append to ledger: {exc}")
            return None
        return entry

    # ---- reading ----

    def all(self) -> list[dict]:
        """Every well-formed entry; malformed lines are skipped."""
        if not self.path.exists():
            return []
        out: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
        return out

    def recent(self, n: int = 5) -> list[dict]:
        n = int(n)
        return self.all()[-n:] if n > 0 else []

    def summary(self, n: int = 6) -> str:
        """Compact context block of the last n cycles, for injection into the
        agent's next 'think of an idea' prompt."""
        entries = self.recent(n)
        if not entries:
            return "(no experiments yet)"
        lines = []
        for e in entries:
            m = e.get("metrics") if isinstance(e.get("metrics"), dict) else {}
            m_str = ", ".join(f"{k}={v}" for k, v in m.items()) or "no metrics"
            hypo = (e.get("hypothesis") or "").strip()
            if len(hypo) > 140:
                hypo = hypo[:137] + "..."
            status = e.get("status") or e.get("stage") or "?"
            line = f"- cycle {e.get('cycle','?')} [{status}] {hypo} ({m_str})"
            concl = (e.get("conclusion") or "").strip()
            if concl:
                line += f" -> {concl[:140]}"
            lines.append(line)
        return "\n".join(lines)

    def tried_techniques(self) -> list[str]:
        """Techniques already attempted — pass to retrieval/ideation so the
        agent does not re-propose what it has already tested."""
        seen = []
        for e in self.all():
            t = (e.get("source_technique") or "").strip()
            if t and t not in seen:
                seen.append(t)
        return seen

    def resource_totals(self) -> dict:
        """Totals for the Feasibility & Practicality deliverable."""
        entries = self.all()
        return {
            "tokens_in": sum(int(e.get("tokens_in", 0)) for e in entries),
            "tokens_out": sum(int(e.get("tokens_out", 0)) for e in entries),
            "gpu_hours": round(sum(float(e.get("gpu_seconds", 0.0)) for e in entries) / 3600.0, 3),
            "cycles": len(entries),
            "manual_interventions": sum(1 for e in entries if e.get("manual_intervention")),
        }


# ---- pure-python signal readers (zero LLM cost) ----

def _series(entries: list[dict], key: str) -> list[float]:
    out = []
    for e in entries:
        m = e.get("metrics")
        if isinstance(m, dict) and key in m:
            try:
                out.append(float(m[key]))
            except (TypeError, ValueError):
                continue
    return out


def best_metrics(entries: list[dict]) -> dict:
    """Best value seen for each scored metric (higher is better)."""
    return {k: (max(_series(entries, k)) if _series(entries, k) else None) for k in METRIC_KEYS}


def delta_over_baseline(entries: list[dict], baseline: dict) -> dict:
    """The thing you're actually scored on: best - baseline, per metric."""
    best = best_metrics(entries)
    return {
        k: (round(best[k] - baseline[k], 5) if best.get(k) is not None and k in baseline else None)
        for k in METRIC_KEYS
    }


def detect_stagnation(entries: list[dict], key: str = "primary",
                      threshold_cycles: int = 3, min_delta: float = 1e-4) -> dict:
    """Has the best value of `key` failed to improve for `threshold_cycles`
    metric-bearing cycles? Advisory — the controller decides what to do
    (stop, or force a different idea category to escape a local optimum)."""
    pts = _series(entries, key)
    verdict = {"metric": key, "stagnating": False, "best": None,
               "cycles_since_improvement": 0, "n_points": len(pts)}
    if len(pts) <= threshold_cycles:
        verdict["reason"] = "not enough points yet"
        verdict["best"] = max(pts) if pts else None
        return verdict
    best = pts[0]
    since = 0
    for v in pts[1:]:
        if v > best + min_delta:
            best, since = v, 0
        else:
            since += 1
    verdict["best"] = best
    verdict["cycles_since_improvement"] = since
    verdict["stagnating"] = since >= threshold_cycles
    return verdict


class Journal:
    """Append-only DEAD_ENDS.md and INSIGHTS.md. The loop injects each file's
    tail into the next 'think of an idea' prompt so the agent avoids known
    dead ends and reuses hard-won lessons."""

    def __init__(self, workspace: Path | str, max_chars: int = 40_000):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.max_chars = max_chars
        self.dead_ends = self.workspace / "DEAD_ENDS.md"
        self.insights = self.workspace / "INSIGHTS.md"
        for p, title in ((self.dead_ends, "Dead Ends"), (self.insights, "Insights")):
            if not p.exists():
                p.write_text(f"# {title}\n\n", encoding="utf-8")

    def _append(self, path: Path, entry: str) -> None:
        entry = (entry or "").strip()
        if not entry:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M")
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"- [{stamp}] {entry}\n")
        except OSError as exc:
            logger.warning(f"Failed to append to {path.name}: {exc}")

    def add_dead_end(self, entry: str) -> None:
        self._append(self.dead_ends, entry)

    def add_insight(self, entry: str) -> None:
        self._append(self.insights, entry)

    def context(self, max_chars: int = 2000) -> str:
        """Tail of both files, for the ideation prompt."""
        def tail(p: Path) -> str:
            try:
                return p.read_text(encoding="utf-8")[-max_chars:] if p.exists() else ""
            except OSError:
                return ""
        return f"## Known dead ends\n{tail(self.dead_ends)}\n\n## Insights\n{tail(self.insights)}"


if __name__ == "__main__":
    # tiny smoke test / usage example
    led = ExperimentLedger("/tmp/amra_demo")
    jour = Journal("/tmp/amra_demo")

    baseline = {"ctr_auc": 0.620, "cvr_auc": 0.630}

    led.record(cycle=1, stage="draft", hypothesis="Baseline shared-bottom MTL model",
               module_changed="model.py", status="ok",
               metrics={"ctr_auc": 0.618, "cvr_auc": 0.629}, data_scale="subsample",
               tokens_in=3200, tokens_out=800, gpu_seconds=280,
               conclusion="Roughly reproduces baseline.")
    led.record(cycle=2, parent_cycle=1, stage="improve",
               hypothesis="Swap shared-bottom for ESMM to fix CVR sample-selection bias",
               source_technique="ESMM (Entire Space Multi-Task Model)",
               predicted_effect="+~0.005 CVR AUC",
               module_changed="model.py", status="ok",
               metrics={"ctr_auc": 0.631, "cvr_auc": 0.641}, data_scale="subsample",
               tokens_in=4100, tokens_out=1200, gpu_seconds=300,
               conclusion="ESMM helped both; prediction held.")
    jour.add_insight("ESMM lifts CVR by modelling over the entire impression space (fixes selection bias).")
    led.record(cycle=3, parent_cycle=2, stage="improve",
               hypothesis="Add user x category feature crossing",
               source_technique="DeepFM-style feature crossing",
               predicted_effect="+CTR AUC", module_changed="features.py",
               status="diverged", metrics={},
               error="loss went to NaN at step 1400", recovered=False,
               data_scale="subsample", tokens_in=3800, tokens_out=1500, gpu_seconds=95,
               conclusion="Unstable; reverted to cycle 2.")
    jour.add_dead_end("Raw user x category crossing without hashing -> NaN loss (too high cardinality).")

    print("Recent summary:\n" + led.summary())
    print("\nBest:", best_metrics(led.all()))
    print("Delta over baseline:", delta_over_baseline(led.all(), baseline))
    print("Stagnation (cvr_auc):", detect_stagnation(led.all(), "cvr_auc"))
    print("Tried techniques:", led.tried_techniques())
    print("Resource totals:", led.resource_totals())
    print("\nJournal context:\n" + jour.context())
