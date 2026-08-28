"""
The autonomous loop: Ideator -> Coder -> Executor -> Reflector.

Runs against KuaiRand-Pure, scored by official/evaluate.py, and stops on the
organisers' convergence rule (validation primary not improved by more than
epsilon = 0.002 over N = 3 consecutive iterations) or on a budget ceiling,
whichever comes first.

Two invariants that the previous version got wrong and that matter for scoring:

  1. REVERT ACTUALLY REVERTS. When the Reflector rejects a change, the module is
     restored from its backup on disk. A cosmetic revert lets rejected code
     contaminate every later cycle.
  2. THE TEST SPLIT IS NEVER TOUCHED. runner.run() is called without
     evaluate_test, so the loop only ever sees validation. The final checkpoint
     is produced by a separate script.

Every cycle is appended to the ledger with hypothesis, code diff, metrics, and
error/recovery state -- that file is the "Run & Iteration Logs" deliverable.
"""

from __future__ import annotations

import argparse
import json
import os
import logging
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import ledger as ledger_module              # noqa: E402
from lib.beliefs import BeliefStore                  # noqa: E402
from lib import llm                                  # noqa: E402
from solution import instrument                      # noqa: E402
from solution import runner                          # noqa: E402
from solution.train import RANDOM_FLOOR              # noqa: E402
from solution.scoring import (                       # noqa: E402
    BASELINE, EPSILON, N_CONSECUTIVE, has_converged,
)

from .analyst import observe                         # noqa: E402
from .classifier import classify                     # noqa: E402
from .specifier import check as check_contract       # noqa: E402
from .specifier import specify                       # noqa: E402
from .inventor import invent                         # noqa: E402
from .coder import apply_code_patch                  # noqa: E402
from .exploiter import run_exploit                    # noqa: E402
from .ideator import (                               # noqa: E402
    MAX_REVISIONS, PROMISING_GAP, propose_idea, propose_revision,
)
from .reflector import MARGIN, classify_failure, decide_keep  # noqa: E402

logger = logging.getLogger("amra.controller")

EDITABLE = ROOT / "solution"
LOCKFILE = ROOT / ".loop_running.lock"
BACKUP_DIR = ROOT / ".module_backups"

# A result this close to the incumbent is inside the noise (2x one published
# std), so it is re-run on a second seed before we believe it either way.
TIEBREAK_BAND = 2 * MARGIN

# How many seeds to average when a result lands in the noise band. Two is NOT
# enough: a candidate scoring 0.6022/0.6025 on two seeds -- beating baseline on
# both -- averaged to exactly the baseline over five, and the untouched control
# itself produced 0.6023 on one seed. With a per-seed std near 0.0005 and the
# effects we are chasing near 0.0007, two samples cannot separate them.
# Only borderline cycles pay this cost.
TIEBREAK_SEEDS = 5


def _restore_all_modules() -> None:
    """Put every agent-editable module back to its last accepted state.

    Killing a run mid-cycle used to leave the agent's patch on disk. Nothing
    restored it, because restore only happened on a normal revert. The next
    thing to touch solution/ then measured the abandoned patch and attributed
    its behaviour to the reference -- which is exactly how a pathological patch
    got misread as slow instrumentation.
    """
    for name in ("features.py", "model.py", "train.py"):
        src = BACKUP_DIR / name
        if src.exists():
            shutil.copy2(src, EDITABLE / name)


def _acquire_lock() -> None:
    """Refuse to start if another loop is live.

    solution/*.py is shared mutable state: the loop rewrites it every cycle and
    the runner reloads it every run. A second process touching those files --
    another loop, or a human running a diagnostic -- silently corrupts whichever
    experiment is in flight, and the corruption is invisible in the logs.
    A whole 25-cycle run had to be discarded for exactly this.
    """
    if LOCKFILE.exists():
        raise SystemExit(
            f"A research loop is already running (lock: {LOCKFILE}).\n"
            f"solution/*.py is shared mutable state -- a second writer corrupts\n"
            f"the in-flight experiment. Wait for it, or if it died uncleanly:\n"
            f"  rm {LOCKFILE}")
    LOCKFILE.write_text(f"pid={os.getpid()} started={time.time()}\n")


def _release_lock() -> None:
    LOCKFILE.unlink(missing_ok=True)


def _snapshot_modules() -> None:
    """Copy the agent-editable modules aside so a revert has something to
    restore. Runs once at loop start."""
    BACKUP_DIR.mkdir(exist_ok=True)
    for name in ("features.py", "model.py", "train.py"):
        shutil.copy2(EDITABLE / name, BACKUP_DIR / name)


def _restore_module(name: str) -> bool:
    """Put a rejected module back. Returns True if a restore happened."""
    src = BACKUP_DIR / name
    if not src.exists():
        logger.error(f"No backup for {name}; cannot revert.")
        return False
    shutil.copy2(src, EDITABLE / name)
    return True


def _accept_module(name: str) -> None:
    """Promote the current file to be the new revert target."""
    shutil.copy2(EDITABLE / name, BACKUP_DIR / name)


def _compose_cfg(seed: int, base_cfg: dict, idea_cfg: dict | None) -> dict:
    """Training config for a cycle: seed, then parameters an EXPLOIT win tuned,
    then whatever the idea sets explicitly.

    base_cfg exists because the next cycle used to rebuild the config from the
    idea alone, dropping a tuned value while `incumbent` kept the score that
    value earned -- so every later experiment was judged against a bar its own
    config could not reach. The idea still wins, so the agent can set a
    parameter deliberately.
    """
    return {"seed": seed, **base_cfg, **(idea_cfg or {})}


def _execute(data_dir: str, cfg: dict, seed: int, incumbent: float) -> dict:
    """Run the pipeline; re-run on a second seed if the result lands in the noise.

    The keep margin is one published std (0.0008). A single-seed result inside
    that band is a coin flip, so a decision made on it is not evidence. Only
    borderline cycles pay the 2x cost.
    """
    # Cap one experiment. A runaway cycle once consumed 98% of a run's compute.
    result = runner.run(data_dir, config=cfg, seed=seed, timeout_seconds=900)
    if result["status"] != "ok":
        return result

    primary = result["metrics"]["primary"]
    if abs(primary - incumbent) >= TIEBREAK_BAND:
        result["seeds_run"] = 1
        return result

    logger.info(f"  primary {primary:.4f} is within {TIEBREAK_BAND:.4f} of "
                f"incumbent {incumbent:.4f} -- confirming over "
                f"{TIEBREAK_SEEDS} seeds")
    runs = [result["metrics"]]
    for extra in range(1, TIEBREAK_SEEDS):
        s = seed + extra
        r = runner.run(data_dir, config={**cfg, "seed": s}, seed=s)
        if r["status"] != "ok":
            logger.warning(f"  seed {s} failed; continuing with {len(runs)} seeds")
            continue
        runs.append(r["metrics"])
        result["wall_seconds"] += r["wall_seconds"]

    merged = {k: sum(m[k] for m in runs) / len(runs)
              for k in ("GAUC", "nDCG@5", "primary")}
    spread = [f"{m['primary']:.4f}" for m in runs]
    logger.info(f"  seeds: {' '.join(spread)} | mean {merged['primary']:.4f} "
                f"(n={len(runs)})")
    result["metrics"].update(merged)
    result["seeds_run"] = len(runs)
    return result


def run_loop(data_dir: str,
             max_iterations: int = 20,
             wall_clock_seconds: int = 7200,
             token_budget: int = 200_000,
             llm_model: str = "gpt-4o-mini",
             seed: int = 0,
             workspace: Path | str = "workspace",
             pipeline: str = "classify") -> dict:
    """Run the loop to convergence or budget. Returns a summary dict."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    _acquire_lock()

    # Ctrl-C / SIGTERM must not leave a half-applied patch behind.
    import atexit
    import signal as _sig

    def _cleanup(*_a):
        _restore_all_modules()
        _release_lock()

    atexit.register(_cleanup)
    for _s in (_sig.SIGINT, _sig.SIGTERM):
        try:
            _sig.signal(_s, lambda *_a: (_cleanup(), sys.exit(130)))
        except (ValueError, OSError):
            pass                                   # not on the main thread

    led = ledger_module.ExperimentLedger(workspace)
    jour = ledger_module.Journal(workspace)
    beliefs = BeliefStore(workspace)

    _snapshot_modules()

    # The analyst needs the frames in-process. runner keeps its own cache; this
    # shares the same underlying load via the same cache key.
    from solution import dataset as _ds_mod
    ds_for_eda = _ds_mod.load(data_dir)

    logger.info("instrumenting the starting configuration ...")
    baseline_instr = instrument.measure(ds_for_eda, {"seed": seed})

    logger.info(f"data_dir={data_dir} seed={seed}")
    logger.info(f"ideation pipeline: {pipeline}   "
                f"({'observe -> classify -> invent -> pull-literature' if pipeline == 'classify' else 'observe -> pick-from-library'})")
    logger.info(f"budget: {max_iterations} iters / {wall_clock_seconds}s / "
                f"{token_budget} tokens")
    logger.info("model routing by role:\n" + llm.routing_table(llm_model))
    logger.info(f"convergence: epsilon={EPSILON} over N={N_CONSECUTIVE} iterations")
    logger.info(f"baseline to beat: valid primary {BASELINE['valid']['primary']}")

    # MEASURE THE STARTING CODE before crediting anything to the agent.
    # The incumbent used to default to the published baseline, so a run begun
    # from an already-good config awarded a phantom "win" on its first cycle
    # for a score that was sitting in the file the agent edited. The incumbent
    # must be what is actually on disk.
    logger.info("scoring the starting configuration ...")
    _start = runner.run(data_dir, config={"seed": seed}, seed=seed)
    if _start["status"] == "ok":
        start_primary = _start["metrics"]["primary"]
        logger.info(f"starting code scores {start_primary:.4f} "
                    f"(published baseline {BASELINE['valid']['primary']}). "
                    f"Improvements are measured from HERE.")
    else:
        start_primary = -1.0
        logger.warning(f"starting code does not run: "
                       f"{_start['error'].splitlines()[0][:120]}")

    started = time.time()
    tokens_used = 0
    cycle = 0
    primary_history: list[float] = []
    best_primary = start_primary
    # Distinct from `best_primary > 0`. Anchoring the incumbent to the starting
    # code (correct) made that test true from cycle 1, which started the
    # convergence clock immediately and killed a 25-cycle run at cycle 4. What
    # the convergence rule needs to know is whether the agent has ACCEPTED an
    # improvement -- until it has, the run is still searching.
    has_accepted_improvement = False
    best_checkpoint: dict | None = None
    base_cfg: dict = {}
    candidate: dict | None = None      # a near miss being revised
    stop_reason = "max_iterations"

    while cycle < max_iterations:
        cycle += 1
        elapsed = time.time() - started

        if elapsed > wall_clock_seconds:
            stop_reason = "wall_clock"
            logger.info(f"wall-clock budget reached ({elapsed:.0f}s). stopping.")
            break
        if tokens_used > token_budget:
            stop_reason = "token_budget"
            logger.info(f"token budget reached ({tokens_used}). stopping.")
            break

        logger.info("=" * 72)
        logger.info(f"cycle {cycle} | elapsed {elapsed:.0f}s | tokens {tokens_used}")

        idea = patch = module = None
        idea_tokens = coder_tokens = refl_tokens = obs_tokens = 0
        observations = None

        # --- 0. Observe: measure the data before theorising --------------
        # Skipped while revising -- a revision is a follow-up on a specific
        # result, not a fresh look at the problem.
        revising = bool(candidate and candidate["attempts"] < MAX_REVISIONS)
        if not revising:
            try:
                obs_record, obs_tokens = observe(ds_for_eda, led, jour, model=llm_model)
                tokens_used += obs_tokens
                observations = obs_record
            except Exception as exc:                   # noqa: BLE001
                logger.warning(f"analyst failed ({exc}); proceeding without observations")

        # --- 1. Ideate: revise a near miss, or propose something new -----
        prior_code = None
        try:
            if revising:
                idea, idea_tokens = propose_revision(candidate, model=llm_model)
                prior_code = candidate["code"]
                logger.info(f"REVISE ({candidate['attempts'] + 1}/{MAX_REVISIONS}) "
                            f"{candidate['idea'].get('source_technique')} "
                            f"@ {candidate['metrics']['primary']:.4f}")
            elif pipeline == "classify":
                # observe -> CLASSIFY -> invent open-ended -> pull literature.
                # The classifier turns measurements into a named problem, which
                # is the abstraction an intervention can attach to. Without it
                # the loop measured the decisive fact and never acted on it.
                probs, cls_tokens = classify(observations, beliefs=beliefs,
                                             led=led, model=llm_model)
                idea_tokens += cls_tokens
                problems = probs.get("problems", [])
                if not problems:
                    logger.info("no problem named this cycle; skipping")
                    led.record(cycle=cycle, stage="observe", status="skipped",
                               observations=observations,
                               tokens_in=obs_tokens + cls_tokens,
                               manual_intervention=False)
                    candidate = None
                    continue
                idea, inv_tokens = invent(problems, beliefs=beliefs, led=led,
                                          model=llm_model)
                idea_tokens += inv_tokens
                candidate = None
            else:
                idea, idea_tokens = propose_idea(led=led, jour=jour,
                                                 model=llm_model,
                                                 observations=observations,
                                                 beliefs=beliefs)
                candidate = None
            tokens_used += idea_tokens
            logger.info(f"idea: {idea['hypothesis'][:110]}")
            logger.info(f"  technique: {idea.get('source_technique')} "
                        f"| predicts: {idea.get('predicted_effect')}")
            if idea.get("grounded_in"):
                logger.info(f"  grounded in: {str(idea['grounded_in'])[:110]}")
            if idea.get("kill_criterion"):
                logger.info(f"  kill if: {str(idea['kill_criterion'])[:110]}")
            if idea.get("used_literature"):
                logger.info("  (consulted literature on its own request)")
        except Exception as exc:
            logger.error(f"ideator failed: {exc}", exc_info=True)
            led.record(cycle=cycle, stage="ideate", status="bug", error=str(exc),
                       recovered=True, manual_intervention=False)
            jour.add_dead_end(f"Ideator crash: {str(exc)[:180]}")
            continue

        # --- 1b. SPECIFY: fix the contract BEFORE the code exists --------
        # Written from the NAMED PROBLEM, not from the implementation. If the
        # coder wrote first, it would write a contract its own code satisfies --
        # the same shape as the metric-gaming failure.
        contract = None
        if pipeline == "classify" and idea.get("problem"):
            try:
                contract, spec_tokens = specify(idea, idea["problem"],
                                                baseline_instr, model=llm_model)
                tokens_used += spec_tokens
                idea_tokens += spec_tokens
                # specify() returns None when it could produce no satisfiable
                # postcondition. Running ungated beats blocking on nothing.
            except Exception as exc:               # noqa: BLE001
                logger.warning(f"specify failed ({exc}); no semantic gate this cycle")

        # --- 2. Code ----------------------------------------------------
        try:
            patch, coder_tokens, module = apply_code_patch(
                idea=idea, model=llm_model, prior_code=prior_code)
            tokens_used += coder_tokens
            logger.info(f"patched {module} ({len(patch.splitlines())} lines)")
        except Exception as exc:
            logger.error(f"coder failed: {exc}")
            led.record(cycle=cycle, hypothesis=idea["hypothesis"],
                       source_technique=idea.get("source_technique", ""),
                       stage="code", status="bug", error=str(exc),
                       recovered=True, manual_intervention=False,
                       tokens_in=idea_tokens + coder_tokens)
            jour.add_dead_end(
                f"Coder crash on '{idea['hypothesis'][:60]}': {str(exc)[:150]}")
            continue

        # --- 2b. VERIFY SEMANTICS: did the patch do what it claimed? -----
        run_cfg = _compose_cfg(seed, base_cfg, idea.get("config"))
        verdict = None
        if contract:
            instrument.invalidate_cache()          # the patch may re-encode
            after_instr = instrument.measure(ds_for_eda, run_cfg)
            verdict = check_contract(contract, baseline_instr, after_instr)
            realised = instrument.diff(baseline_instr, after_instr)
            if verdict["satisfied"]:
                logger.info(f"  semantic contract SATISFIED "
                            f"({len(verdict['clauses'])} clauses)")
            else:
                # IMPLEMENTATION FAILURE. The patch did not do what it claimed,
                # so the experiment says NOTHING about the hypothesis. Spending
                # a training run here would produce a number that gets recorded
                # as evidence against a possibly-correct idea.
                logger.warning("  semantic contract VIOLATED — not training:")
                for f in verdict["failures"][:4]:
                    logger.warning(f"      {f}")
                led.record(
                    cycle=cycle, hypothesis=idea.get("hypothesis", ""),
                    source_technique=idea.get("source_technique", ""),
                    predicted_effect=idea.get("predicted_effect", ""),
                    stage="implementation_failure", module_changed=module or "",
                    code_diff=patch or "", status="implementation_failure",
                    metrics={}, error="; ".join(verdict["failures"])[:2000],
                    recovered=True, manual_intervention=False,
                    tokens_in=obs_tokens + idea_tokens + coder_tokens,
                    gpu_seconds=0.0, observations=observations,
                    conclusion=("The patch did not implement the stated "
                                "intervention. NO evidence about the hypothesis."),
                )
                jour.add_dead_end(
                    f"[cycle {cycle}] IMPLEMENTATION failure (not a scientific "
                    f"one): {idea.get('source_technique','?')} — "
                    f"{verdict['failures'][0][:120]}")
                _restore_module(module)
                continue

        incumbent = best_primary if best_primary > 0 else BASELINE["valid"]["primary"]
        result = _execute(data_dir, run_cfg, seed, incumbent)
        status = result["status"]
        metrics = result.get("metrics", {})
        error = result.get("error", "")
        gpu_seconds = result.get("wall_seconds", 0.0)

        if status == "ok":
            logger.info(f"  valid GAUC {metrics['GAUC']:.4f} "
                        f"nDCG@5 {metrics['nDCG@5']:.4f} "
                        f"primary {metrics['primary']:.4f} "
                        f"({gpu_seconds:.0f}s, {result.get('epochs_run')} epochs)")
        elif status == "timeout":
            logger.warning(f"  TIMEOUT — experiment exceeded its budget and was "
                           f"killed. {error.splitlines()[0][:120]}")
        else:
            logger.warning(f"  run failed: {error.splitlines()[0][:160]}")

        # --- 4. Reflect -------------------------------------------------
        try:
            failure_class = classify_failure(
                status, (verdict or {}).get("satisfied") if verdict else None,
                metrics, incumbent, result.get("history"))
            if status != "ok" or metrics.get("primary", 0) <= incumbent:
                logger.info(f"  failure class: {failure_class.upper()}"
                            + ("  (hypothesis NOT weakened)"
                               if failure_class != "scientific" else
                               "  (evidence against the hypothesis)"))
            keep, lesson, refl_tokens = decide_keep(
                cycle=cycle, idea=idea, metrics=metrics, error=error,
                status=status, led=led, jour=jour, model=llm_model,
                best_primary=best_primary,
                history=result.get("history", []),
                config=run_cfg,
                beliefs=beliefs,
                observations=observations,
                failure_class=failure_class,
            )
            tokens_used += refl_tokens
        except Exception as exc:
            logger.error(f"reflector failed: {exc}")
            keep, lesson, refl_tokens = False, f"Reflector crash: {exc}", 0

        # --- Apply the decision to disk ---------------------------------
        if keep:
            _accept_module(module)
            best_primary = metrics["primary"]
            best_checkpoint = {"cycle": cycle, "metrics": metrics,
                               "idea": idea, "module": module, "config": run_cfg}
            has_accepted_improvement = True
            instrument.invalidate_cache()          # features.py may have changed
            baseline_instr = instrument.measure(ds_for_eda, {"seed": seed})
            candidate = None                     # accepted: nothing left to revise
            logger.info(f"  KEEP  -> new best primary {best_primary:.4f}")

            # ---- EXPLOIT: a win is a new local research problem ----------
            # V4 stopped here and went back to open exploration. But the human
            # gain of +0.0012 from a structural change was followed by +0.0010
            # more from retuning the step size FOR that change -- a parameter
            # stale only because the intervention altered the gradient scale.
            # This works that branch before returning to exploration.
            try:
                ex = run_exploit(
                    idea, contract, realised if contract else None, run_cfg,
                    best_primary=best_primary,
                    keep_primary=metrics["primary"],
                    execute=lambda c: _execute(data_dir, c, seed, best_primary),
                    record=lambda **kw: led.record(
                        cycle=cycle,
                        source_technique=idea.get("source_technique", ""),
                        stage="exploit", module_changed=module or "",
                        status="ok", data_scale="full",
                        manual_intervention=False, **kw),
                    log=logger.info,
                    time_left=lambda: time.time() - started <= wall_clock_seconds,
                    llm_model=llm_model)
                tokens_used += ex["tokens"]
                if ex["checkpoint_cfg"] is not None:
                    best_primary = ex["best_primary"]
                    run_cfg = ex["run_cfg"]
                    base_cfg = {k: v for k, v in ex["run_cfg"].items()
                                if k != "seed"}
                    best_checkpoint = {"cycle": cycle, "metrics": {"primary": best_primary},
                                       "idea": idea, "module": module,
                                       "config": ex["checkpoint_cfg"]}
                if ex.get("summary"):
                    logger.info(f"    {ex['summary']}")
                    jour.add_insight(f"[cycle {cycle}] {ex['summary']}")
                    primary_history.append(best_primary)
            except Exception as exc:               # noqa: BLE001
                logger.warning(f"  exploit branch failed ({exc}); continuing")
        else:
            gap = incumbent - metrics.get("primary", -99)
            promising = status == "ok" and 0 <= gap < PROMISING_GAP
            attempts = (candidate["attempts"] + 1) if candidate else 0

            if promising and attempts < MAX_REVISIONS:
                # Hold the near miss for revision instead of discarding it.
                candidate = {"module": module, "code": patch, "metrics": metrics,
                             "idea": idea, "lesson": lesson, "config": run_cfg,
                             "incumbent": incumbent, "attempts": attempts}
                logger.info(f"  REVERT but HOLDING as candidate "
                            f"(short by {gap:.4f}, revision {attempts + 1} next)")
            else:
                if candidate:
                    logger.info(f"  revision budget spent on "
                                f"{candidate['idea'].get('source_technique')}; "
                                f"moving on")
                candidate = None
            reverted = _restore_module(module)
            logger.info(f"  REVERT ({'restored' if reverted else 'RESTORE FAILED'})"
                        f" -> {lesson[:100]}")

        # A diverged run (below the random floor) is a broken experiment, not
        # evidence of a plateau -- it must not feed the convergence rule.
        if status == "ok" and metrics["primary"] >= RANDOM_FLOOR:
            primary_history.append(metrics["primary"])
        elif status == "ok":
            logger.info(f"  diverged ({metrics['primary']:.4f} < {RANDOM_FLOOR}); "
                        f"excluded from the convergence series")

        led.record(
            cycle=cycle,
            hypothesis=idea.get("hypothesis", ""),
            source_technique=idea.get("source_technique", ""),
            predicted_effect=idea.get("predicted_effect", ""),
            parent_cycle=(cycle - 1) if idea.get("is_revision") else None,
            stage=("improve" if keep else
                   ("revise" if idea.get("is_revision") else "debug")),
            module_changed=module or "",
            code_diff=patch or "",   # full text; the ledger is the deliverable
            status=status,
            metrics=metrics if status == "ok" else {},
            data_scale="full",
            error=error,
            recovered=(status != "ok"),
            manual_intervention=False,
            tokens_in=obs_tokens + idea_tokens + coder_tokens + refl_tokens,
            tokens_out=0,
            gpu_seconds=gpu_seconds,
            conclusion=lesson,
            observations=observations,
        )

        # --- Convergence (organisers' rule) ------------------------------
        # A win of less than epsilon used to END the run on the cycle it
        # succeeded -- both winning runs converged the moment they improved
        # (+0.0017 and +0.0014, against epsilon=0.002). That makes the exploit
        # branch unreachable: the event that opens it is the event that stops
        # the loop. Convergence is therefore not evaluated on a winning cycle.
        if keep:
            logger.info("  (convergence not evaluated on a winning cycle)")
        elif has_converged(primary_history,
                           has_improved=has_accepted_improvement):
            stop_reason = "converged"
            logger.info(f"CONVERGED: validation primary has not improved by "
                        f">{EPSILON} over the last {N_CONSECUTIVE} iterations.")
            break

    _restore_all_modules()
    _release_lock()
    elapsed = time.time() - started
    totals = led.resource_totals()
    summary = {
        "stop_reason": stop_reason,
        "cycles": cycle,
        "wall_seconds": round(elapsed, 1),
        "start_primary": start_primary if start_primary > 0 else None,
        "best_primary": best_primary if best_primary > 0 else None,
        "improvement_over_start": (round(best_primary - start_primary, 4)
                                   if best_primary > 0 and start_primary > 0 else None),
        "beat_baseline": best_primary > BASELINE["valid"]["primary"],
        "baseline_primary": BASELINE["valid"]["primary"],
        "delta_vs_baseline": (round(best_primary - BASELINE["valid"]["primary"], 4)
                              if best_primary > 0 else 0.0),
        "submission_checkpoint": ("agent-best" if best_primary > 0
                                  else "official-baseline (nothing beat it)"),
        "best_checkpoint": best_checkpoint,
        "primary_history": primary_history,
        "tokens_by_role": dict(llm.USAGE.by_role),
        "model_routing": {r: llm.model_for(r, llm_model) for r in llm.ROLES},
        **totals,
    }

    logger.info("=" * 72)
    logger.info(f"stopped: {stop_reason} after {cycle} cycles, {elapsed:.0f}s")
    logger.info(f"best valid primary: {best_primary:.4f} "
                f"(baseline {BASELINE['valid']['primary']}, "
                f"delta {summary['delta_vs_baseline']})")
    logger.info("tokens by role:\n" + llm.USAGE.report())
    logger.info(f"tokens: {totals.get('tokens_in', 0) + totals.get('tokens_out', 0)} "
                f"| GPU-hours: {totals.get('gpu_hours', 0):.4f}")
    (workspace / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Autonomous ML Research Agent loop")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max-iterations", type=int, default=20)
    ap.add_argument("--wall-clock", type=int, default=7200, help="seconds")
    ap.add_argument("--token-budget", type=int, default=200_000)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--pipeline", default="classify",
                    choices=["classify", "library"],
                    help="classify = observe->classify->invent->pull-literature; "
                         "library = the older menu-driven ideator")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    print(json.dumps(run_loop(
        data_dir=args.data_dir,
        max_iterations=args.max_iterations,
        wall_clock_seconds=args.wall_clock,
        token_budget=args.token_budget,
        llm_model=args.model,
        seed=args.seed,
        workspace=args.workspace,
        pipeline=args.pipeline,
    ), indent=2, default=str))
