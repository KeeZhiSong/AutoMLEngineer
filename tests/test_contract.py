"""
Contract tests. These guard the invariants that fail SILENTLY -- the ones where
a broken run still produces a plausible-looking number.

    python3 -m pytest tests/test_contract.py -q --data-dir <KuaiRand-Pure/data>
or  python3 tests/test_contract.py --data_dir <KuaiRand-Pure/data>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import dataset, features            # noqa: E402
from solution.scoring import has_converged, score  # noqa: E402


def test_convergence_rule():
    """epsilon=0.002 over N=3. A plateau converges; real gains do not."""
    # A run that has never accepted an improvement is still SEARCHING, not
    # converged -- otherwise every search ends after exactly N iterations.
    assert not has_converged([0.60, 0.60, 0.60, 0.60], has_improved=False)

    # REGRESSION: the gate must key off "an improvement was ACCEPTED", never
    # off "the incumbent is a positive number". Anchoring the incumbent to the
    # starting code made the latter true from cycle 1 and killed a 25-cycle run
    # at cycle 4 with 92% of its token budget unspent.
    src = (ROOT / "agent" / "controller.py").read_text()
    assert "has_improved=has_accepted_improvement" in src, \
        "convergence gate is not keyed to accepted improvements"
    assert "has_improved=best_primary > 0" not in src, \
        "convergence gate regressed to the incumbent-sign test"
    assert not has_converged([0.60])
    assert not has_converged([0.60, 0.61, 0.62])
    # Flat after an early win -> converged.
    assert has_converged([0.60, 0.6300, 0.6301, 0.6302, 0.6303])
    # Still climbing by more than epsilon -> not converged.
    assert not has_converged([0.60, 0.61, 0.62, 0.64, 0.66])
    # Improvement smaller than epsilon is noise -> converged.
    assert has_converged([0.60, 0.6000, 0.6005, 0.6010, 0.6015])
    print("  convergence rule OK")


def test_scoring_matches_official():
    """score() must be the official implementation, not a lookalike."""
    users = ["a", "a", "a", "b", "b", "c", "c"]
    labels = [1, 0, 0, 0, 1, 0, 0]
    scores = [3.0, 2.0, 1.0, 1.0, 2.0, 5.0, 4.0]
    got = score(users, labels, scores)
    # user a: perfect ordering. user b: perfect. user c: all-negative -> nDCG 0.
    assert got["nDCG@5"] == (1.0 + 1.0 + 0.0) / 3, got
    assert got["GAUC"] == 1.0, got
    assert abs(got["primary"] - (got["GAUC"] + got["nDCG@5"]) / 2) < 1e-12
    print("  scoring OK (all-negative user counted as nDCG 0)")


def test_label_is_long_view():
    assert dataset.LABEL == "long_view", dataset.LABEL
    print("  label is long_view OK")


def test_no_feedback_leakage():
    """A feedback column used as an input feature would leak the target."""
    leaked = set(features.FIELDS) & set(dataset.FEEDBACK_COLUMNS)
    assert not leaked, f"label leakage via {leaked}"
    assert "long_view" not in features.FIELDS
    print("  no feedback leakage OK")


def test_alignment(data_dir: str):
    """Our loader must reproduce official.data.load() row-for-row, or every
    submission row_id silently misaligns."""
    ds = dataset.load(data_dir)
    counts = dataset.verify_alignment(ds, data_dir)
    assert counts == {"train": 1141112, "valid": 124909, "test": 170588}, counts
    print(f"  loader aligned with official OK {counts}")


def test_train_only_statistics(data_dir: str):
    """fit() on train must not change if valid/test are perturbed."""
    ds = dataset.load(data_dir)
    a = features.fit(ds.train)
    ds.valid.log.loc[:, "duration_ms"] = 999_999
    b = features.fit(ds.train)
    assert np.allclose(a.duration_edges, b.duration_edges)
    assert a.dim == b.dim
    print("  train-only statistics OK")


def test_revision_state_machine():
    """The revise loop must be bounded and must only hold genuine near misses."""
    from agent.ideator import MAX_REVISIONS, PROMISING_GAP
    from agent.controller import TIEBREAK_BAND
    from agent.reflector import MARGIN

    # MAX_REVISIONS is 0: measured across 20 runs, 93 revisions were attempted
    # and none was ever accepted, at a cost of 16% of the project's tokens. The
    # bound must stay non-negative and finite; whether it is currently 0 is a
    # budget decision, not a contract.
    assert MAX_REVISIONS >= 0 and MAX_REVISIONS < 10
    # The band logic below must stay correct whether or not revisions are on, so
    # raising MAX_REVISIONS again cannot silently reintroduce a broken gate.
    # A near miss must be a wider band than the noise margin, or nothing is ever
    # worth revising.
    assert PROMISING_GAP > MARGIN, (PROMISING_GAP, MARGIN)
    # The tie-break band must be wider than the accept margin, or a decision can
    # be made on a difference we never confirmed.
    assert TIEBREAK_BAND > MARGIN, (TIEBREAK_BAND, MARGIN)

    incumbent = 0.6016
    # inside the promising band -> hold for revision
    assert 0 <= incumbent - 0.5990 < PROMISING_GAP
    # far below -> discard, do not burn revisions on it
    assert not (incumbent - 0.5171 < PROMISING_GAP)
    state = "disabled" if MAX_REVISIONS == 0 else f"max {MAX_REVISIONS}"
    print(f"  revision state machine OK ({state}, "
          f"gap {PROMISING_GAP}, tiebreak {TIEBREAK_BAND})")


def test_revision_cannot_switch_technique():
    """A revision that drifts to another technique makes the ledger unreadable."""
    import agent.ideator as I
    src = I.propose_revision.__doc__ or ""
    assert "same technique" in src.lower()
    # The function force-overwrites these two fields after the LLM call.
    import inspect
    body = inspect.getsource(I.propose_revision)
    assert 'revision["source_technique"] = idea.get("source_technique"' in body
    assert 'revision["module"] = candidate["module"]' in body
    print("  revision pinned to its technique/module OK")


def test_seed_tiebreak():
    """A borderline result must be confirmed on a second seed before it is
    believed, and a crash on that second seed must not corrupt the first."""
    from unittest import mock
    from agent import controller

    calls = []

    def fake_run(data_dir, config=None, seed=0, **kw):
        calls.append(seed)
        p = 0.6020 if seed == 0 else 0.6000
        return {"status": "ok", "wall_seconds": 1.0,
                "metrics": {"GAUC": p, "nDCG@5": p, "primary": p}}

    with mock.patch.object(controller.runner, "run", fake_run):
        calls.clear()
        r = controller._execute("d", {}, 0, incumbent=0.6016)
        # Borderline -> averaged over TIEBREAK_SEEDS, not 2. Two seeds proved
        # insufficient: 0.6022/0.6025 averaged to baseline over five.
        assert calls == list(range(controller.TIEBREAK_SEEDS)), calls
        assert r["seeds_run"] == controller.TIEBREAK_SEEDS
        expected = (0.6020 + 0.6000 * (controller.TIEBREAK_SEEDS - 1)) / controller.TIEBREAK_SEEDS
        assert abs(r["metrics"]["primary"] - expected) < 1e-9, r["metrics"]["primary"]

        calls.clear()
        r = controller._execute("d", {}, 0, incumbent=0.5000)
        assert calls == [0] and r["seeds_run"] == 1

    def flaky(data_dir, config=None, seed=0, **kw):
        if seed != 0:                      # every confirmation seed crashes
            return {"status": "bug", "error": "boom", "wall_seconds": 0.1,
                    "metrics": {}}
        return {"status": "ok", "wall_seconds": 1.0,
                "metrics": {"GAUC": .602, "nDCG@5": .602, "primary": .602}}

    with mock.patch.object(controller.runner, "run", flaky):
        r = controller._execute("d", {}, 0, incumbent=0.6016)
        # Every extra seed crashes; the first result must survive intact.
        assert r["status"] == "ok" and r["seeds_run"] == 1
        assert abs(r["metrics"]["primary"] - 0.602) < 1e-9
    print("  seed tie-break OK (confirms borderline, survives a crashed seed)")


def test_no_leaky_statistics_file(data_dir: str):
    """video_features_statistic_pure.csv aggregates over the test window, so
    loading it would contaminate training with test-period labels. Assert we
    never read it, and that the file really does lack any time scoping."""
    src = (ROOT / "solution" / "dataset.py").read_text()
    loaded = [ln for ln in src.splitlines()
              if "read_csv" in ln and "statistic" in ln]
    assert not loaded, f"dataset.py loads the leaky statistics file: {loaded}"

    stats = Path(data_dir) / "video_features_statistic_pure.csv"
    if stats.exists():
        header = stats.read_text().split("\n", 1)[0].split(",")
        # No date/period column => the counters cannot be scoped to train.
        assert not [c for c in header if c.lower() in ("date", "day", "period")], \
            "statistics file now has a date column -- re-evaluate the hazard"
    print("  leaky statistics file not loaded OK")


def test_leakcheck_catches_the_real_leak():
    """Regression test for the exact code that scored a false 0.6449.

    The previous guard compared FIELD NAMES against the feedback list, so a
    derived column named `user_history_interest` that read `is_click` sailed
    through and the loop kept it. Check data ACCESS, not naming.
    """
    from solution.leakcheck import find_leaks, assert_clean

    real_leak = """
import pandas as pd
FIELDS = ["user_id", "video_id", "user_history_interest"]
def _compute_user_history_interest(split):
    log = split.log
    for user_id, group in log.groupby('user_id'):
        history = group['is_click'].rolling(window=5, min_periods=1).sum()
    return history
"""
    assert find_leaks(real_leak, "features.py"), "the real leak must be caught"

    # Every forbidden column, by subscript and by attribute.
    for col in ("is_like", "play_time_ms", "long_view"):
        assert find_leaks(f"x = log[{col!r}]", "features.py"), col
        assert find_leaks(f"x = log.{col}", "features.py"), col
    # The auxiliary-target API is not valid in an input module.
    assert find_leaks("v = split.feedback('is_click')", "features.py")

    # train.py is exempt: auxiliary objectives there are legitimate.
    assert not find_leaks(real_leak, "train.py")

    # Prose may discuss the rule without tripping it.
    assert not find_leaks('"""Never read is_click here."""\nx = 1', "features.py")

    # And the module currently on disk must be clean.
    assert_clean((ROOT / "solution" / "features.py").read_text(), "features.py")
    assert_clean((ROOT / "solution" / "model.py").read_text(), "model.py")
    print("  leakcheck catches the real leak OK")


def test_leakcheck_allows_train_only_target_encoding():
    """fit() sees ONLY train, so aggregating labels there is target encoding,
    not leakage. transform() sees valid/test and must never read labels.

    Added after the guard blocked a legitimate popularity-prior idea. A guard
    that is too strict costs forgone ideas rather than a bad number, which makes
    it harder to notice.
    """
    from solution.leakcheck import find_leaks

    legit = """
def fit(train):
    rate = train.log.groupby('video_id')['long_view'].mean()
    return FeatureState(video_rate=rate)
"""
    assert not find_leaks(legit, "features.py"), "train-only encoding must be allowed"

    # ...but the same read in transform() is leakage.
    bad = """
def transform(split, state):
    return split.log['long_view'].rolling(5).mean()
"""
    assert find_leaks(bad, "features.py"), "transform() reading labels must be caught"

    # model.py has no fit(); build() gets a FeatureState, never data.
    assert find_leaks("def fit(x):\n    return x.log['long_view']", "model.py")
    print("  fit()-scoped target encoding allowed, transform() still blocked OK")


def test_implausible_gain_threshold():
    """A gain larger than any plausible single change must be flagged, not banked."""
    from agent.reflector import IMPLAUSIBLE_GAIN, MARGIN
    from solution.scoring import BASELINE, ORACLE_CEILING

    assert IMPLAUSIBLE_GAIN > MARGIN
    # The entire baseline-to-oracle headroom on valid:
    headroom = ORACLE_CEILING["valid"] - BASELINE["valid"]["primary"]
    assert IMPLAUSIBLE_GAIN < headroom, "threshold must sit inside real headroom"
    # The observed leak (0.6449) must trip it.
    assert 0.6449 - BASELINE["valid"]["primary"] > IMPLAUSIBLE_GAIN
    print(f"  implausible-gain flag OK (>{IMPLAUSIBLE_GAIN} of {headroom:.3f} headroom)")


def test_row_count_invariant(data_dir: str):
    """transform() must return exactly one row per input row.

    Regression test for a real failure: a "synthetic augmentation" patch padded
    every user's list to a fixed length with UNK rows carrying label 0. The
    scored set went from 124,909 rows to 451,647, GAUC went 0.669 -> 0.890
    because ranking real impressions above synthetic filler is trivial, and the
    loop KEPT it. No feedback column was touched, so the leak guard was blind.
    """
    from solution import dataset, features, runner
    ds = runner._load_cached(data_dir)
    st = features.fit(ds.train)
    for name in ("train", "valid", "test"):
        enc = features.transform(ds[name], st)
        assert len(enc.y) == ds[name].n, (name, len(enc.y), ds[name].n)
        assert len(enc.X) == ds[name].n, (name, len(enc.X), ds[name].n)
        assert np.array_equal(enc.user_id, ds[name].user_id), f"{name}: order"

    # And the runner must REFUSE a transform that changes the row count.
    src = (ROOT / "solution" / "runner.py").read_text()
    assert "row-count violation" in src, "runner lost its row-count guard"
    print("  row-count invariant OK (scored set == evaluation split)")


def test_semantic_contract_discriminates():
    """A contract must fail a NO-OP patch. That is its entire purpose.

    The measured bottleneck: the agent named the right problem in 11 of 16
    scored cycles and no implementation cleared the margin. Syntax, import and
    leak guards all pass a patch that changes nothing.
    """
    from agent.specifier import check

    before = {"train_group_size_median": 1.0, "train_rows": 1141112,
              "valid_rows": 124909, "rows_covered_exactly_once": True}
    contract = {
        "postconditions": [{"metric": "train_group_size_median",
                            "op": "approx", "value": 5.0}],
        "invariants": [{"metric": "valid_rows", "op": "unchanged"},
                       {"metric": "rows_covered_exactly_once",
                        "op": "==", "value": True}],
    }

    good = {**before, "train_group_size_median": 5.0}
    assert check(contract, before, good)["satisfied"]

    noop = dict(before)                       # the patch did nothing
    assert not check(contract, before, noop)["satisfied"]

    padded = {**before, "train_group_size_median": 5.0, "valid_rows": 451647,
              "rows_covered_exactly_once": False}
    r = check(contract, before, padded)
    assert not r["satisfied"] and any("INVARIANT" in f for f in r["failures"])
    print("  semantic contract discriminates no-op and padding OK")


def test_contract_cannot_demand_the_impossible():
    """Postconditions must be over quantities a patch CONTROLS.

    A contract once demanded train_positive_rate become 0.3134 (the validation
    rate) when it is 0.3366 -- unreachable without resampling rows, which the
    mandatory invariants forbid. The cycle was blocked for failing a condition
    it could never have met. Data properties are invariants, never targets.
    """
    from agent.specifier import (CONTROLLABLE, FIXED_BY_DATA, MAX_INVARIANTS,
                                 MAX_POSTCONDITIONS)

    assert "train_positive_rate" in FIXED_BY_DATA
    assert "train_rows" in FIXED_BY_DATA and "valid_rows" in FIXED_BY_DATA
    assert "train_group_size_median" in CONTROLLABLE
    assert not set(CONTROLLABLE) & set(FIXED_BY_DATA), "a metric cannot be both"
    # over-constrained contracts block every cycle
    assert MAX_INVARIANTS <= 6 and MAX_POSTCONDITIONS <= 3
    print("  contract targets restricted to controllable quantities OK")


def test_contract_targets_cover_loss_changes():
    """A loss-changing hypothesis must have something legitimate to contract over.

    Regression for a real false block: a focal-loss patch was rejected for
    failing `pct_groups_mixed_label > 9.63`. Changing the loss does not move
    group composition -- grouping happens before the loss is called -- so a
    perfectly correct patch failed a condition it never controlled. The cause
    was that CONTROLLABLE held only data/batching quantities.
    """
    from agent.specifier import CONTROLLABLE

    loss_side = {"loss_fn_name", "grad_dz_absmean", "grad_dz_nonzero_pct",
                 "initial_loss"}
    assert loss_side <= set(CONTROLLABLE), \
        f"no loss-side contract targets: missing {loss_side - set(CONTROLLABLE)}"

    batching_side = {"train_group_size_median", "pct_groups_mixed_label"}
    assert batching_side <= set(CONTROLLABLE)
    print("  contract targets cover loss AND batching interventions OK")


def test_no_phantom_contract_metrics(data_dir: str):
    """Every metric the specifier offers must actually be produced.

    A contract over a metric instrument.measure() never returns can NEVER be
    satisfied, so it blocks that cycle forever. This happened: the loss probe
    was rewritten and three metric names (first_epoch_loss, loss_after_2_epochs,
    loss_scale_order) were left in the specifier's vocabulary pointing at
    nothing.
    """
    from solution import instrument, runner
    from agent.specifier import CONTROLLABLE, FIXED_BY_DATA

    ds = runner._load_cached(data_dir)
    produced = set(instrument.measure(ds, {"seed": 0}))
    offered = set(CONTROLLABLE) | set(FIXED_BY_DATA)
    phantom = offered - produced
    assert not phantom, f"specifier offers metrics never measured: {sorted(phantom)}"
    print(f"  no phantom contract metrics OK ({len(offered)} all produced)")


def test_changed_op_verifies_intervention_not_direction():
    """`changed` must pass a patch that worked but moved the wrong way.

    Two correct patches were rejected in one run:
      - loss redesign: required initial_loss < 0.693, got 0.866. The move PROVES
        the intervention landed; the contract guessed the direction wrong.
      - Transformer model: required model_type == "Transformer", got
        "TransformerModel". Rejected on a substring.
    A contract verifies that the change HAPPENED. Direction belongs in
    predicted_effect, which does not gate.
    """
    from agent.specifier import check

    before = {"initial_loss": 0.693158, "model_type": "FM",
              "train_rows": 1141112, "valid_rows": 124909,
              "rows_covered_exactly_once": True}
    inv = [{"metric": "valid_rows", "op": "unchanged"}]

    # the loss case
    directional = {"postconditions": [{"metric": "initial_loss", "op": "<",
                                       "value": 0.693158}], "invariants": inv}
    changed = {"postconditions": [{"metric": "initial_loss", "op": "changed"}],
               "invariants": inv}
    worked = {**before, "initial_loss": 0.866449}
    assert not check(directional, before, worked)["satisfied"], "should have false-blocked"
    assert check(changed, before, worked)["satisfied"], "`changed` must accept it"

    # the model-name case
    exact = {"postconditions": [{"metric": "model_type", "op": "==",
                                 "value": "Transformer"}], "invariants": inv}
    ch = {"postconditions": [{"metric": "model_type", "op": "changed"}],
          "invariants": inv}
    tf = {**before, "model_type": "TransformerModel"}
    assert not check(exact, before, tf)["satisfied"]
    assert check(ch, before, tf)["satisfied"]

    # and `changed` must STILL catch a genuine no-op
    assert not check(changed, before, dict(before))["satisfied"]
    assert not check(ch, before, dict(before))["satisfied"]
    print("  `changed` accepts working patches, still rejects no-ops OK")


def test_failure_classification():
    """Only a SCIENTIFIC failure may weaken a hypothesis.

    Before this existed, a broken implementation of a good idea wrote a false
    negative into beliefs.jsonl -- so a correctly diagnosed direction accrued
    evidence against itself every time the code was wrong.
    """
    from agent.reflector import classify_failure as cf

    assert cf("ok", False, {"primary": 0.61}, 0.60, [1] * 10) == "implementation"
    assert cf("ok", True, {"primary": 0.40}, 0.60, [1] * 10) == "optimisation"
    assert cf("ok", True, {"primary": 0.599}, 0.60, [1, 2]) == "optimisation"
    assert cf("timeout", None, {}, 0.60, None) == "optimisation"
    # contract held, trained properly, still worse -> real evidence
    assert cf("ok", True, {"primary": 0.599}, 0.60, [1] * 10) == "scientific"
    print("  failure classification OK (implementation/optimisation/scientific)")


def test_exploit_branch_reachable():
    """A win must NOT end the run on the cycle it happened.

    Both winning runs converged the moment they improved (+0.0017 and +0.0014
    against epsilon=0.002), because the rule asks whether improvement EXCEEDED
    epsilon and a sub-epsilon win does not. That made the exploit branch
    unreachable: the event that opens it was the event that stopped the loop.
    """
    import inspect
    from agent import controller

    src = inspect.getsource(controller.run_loop)
    assert "not evaluated on a winning cycle" in src, \
        "convergence is still evaluated on a winning cycle -- exploit unreachable"
    assert "run_exploit" in src, "no exploit branch"

    # and the rule itself still behaves: a sub-epsilon gain IS convergence
    from solution.scoring import has_converged
    assert has_converged([0.6014, 0.6031, 0.6031, 0.6031], has_improved=True)
    print("  exploit branch reachable after a win OK")


def test_exploit_branch_executes():
    """Drive the REAL exploit code path with stubs -- no LLM, no training.

    The branch fires only on a KEEP, and the one V5 run produced none, so this
    code had never executed. Source-text checks were what let that hide; this
    calls it. Numbers are the measured pointwise answer key: lr 3e-3 -> 0.6001,
    1e-3 -> 0.6014, 3e-4 -> 0.6022.
    """
    from agent.exploiter import run_exploit

    table = {3e-3: 0.6001, 1e-3: 0.6014, 3e-4: 0.6022}
    seen, logged = [], []

    def execute(cfg):
        seen.append(cfg["lr"])
        return {"status": "ok", "wall_seconds": 1.0,
                "metrics": {"primary": table[cfg["lr"]], "GAUC": 0.0, "nDCG@5": 0.0}}

    def plan(*_a, **_k):
        return {"mode": "retune", "param": "lr", "values": [1e-3, 3e-4],
                "reason": "step size stale after the intervention"}, 0

    common = dict(execute=execute, log=lambda _s: None, plan_fn=plan)

    out = run_exploit({}, None, None, {"seed": 0, "lr": 3e-3},
                      best_primary=0.6001, keep_primary=0.6001,
                      record=lambda **kw: logged.append(kw),
                      time_left=lambda: True, **common)
    assert seen == [1e-3, 3e-4], f"grid not applied in order: {seen}"
    assert len(logged) == 2, "trials not written to the ledger"
    assert out["checkpoint_cfg"]["lr"] == 3e-4, "did not adopt the best trial"
    assert abs(out["best_primary"] - 0.6022) < 1e-9
    assert out["run_cfg"]["lr"] == 3e-4, "winning config not carried out"
    assert "lr sweep" in out.get("summary", ""), "no dose-response summary"

    # a gain inside the accept band must NOT be adopted
    out2 = run_exploit({}, None, None, {"seed": 0, "lr": 3e-3},
                       best_primary=0.6018, keep_primary=0.6018,
                       record=lambda **kw: None, time_left=lambda: True, **common)
    assert out2["checkpoint_cfg"] is None, "adopted a gain inside the noise band"

    # wall-clock exhaustion stops before the first trial
    out3 = run_exploit({}, None, None, {"seed": 0, "lr": 3e-3},
                       best_primary=0.6001, keep_primary=0.6001,
                       record=lambda **kw: None, time_left=lambda: False, **common)
    assert out3["checkpoint_cfg"] is None and out3["trials"] == [(3e-3, 0.6001)]

    # the adopted value must not depend on the ORDER of the grid
    def mk(order):
        def _p(*_a, **_k):
            return {"mode": "retune", "param": "lr", "values": order, "reason": ""}, 0
        return _p
    picks = set()
    for order in ([1e-3, 3e-4], [3e-4, 1e-3]):
        o = run_exploit({}, None, None, {"seed": 0, "lr": 3e-3},
                        best_primary=0.6001, keep_primary=0.6001, execute=execute,
                        record=lambda **kw: None, log=lambda _s: None,
                        time_left=lambda: True, plan_fn=mk(order))
        picks.add(o["checkpoint_cfg"]["lr"])
    assert picks == {3e-4}, f"grid order changed the winner: {picks}"

    # a crashed trial must not abort the branch
    def flaky(cfg):
        if cfg["lr"] == 1e-3:
            return {"status": "crash", "metrics": {}}
        return execute(cfg)
    out4 = run_exploit({}, None, None, {"seed": 0, "lr": 3e-3},
                       best_primary=0.6001, keep_primary=0.6001,
                       record=lambda **kw: None, time_left=lambda: True,
                       execute=flaky, log=lambda _s: None, plan_fn=plan)
    assert out4["checkpoint_cfg"]["lr"] == 3e-4, "a crashed trial aborted the branch"
    print("  exploit branch executes: grid, ledger, win, band, timeout, crash, order OK")


def test_exploit_win_survives_the_next_cycle():
    """A parameter the exploit stage tuned must still be in force next cycle.

    It was not: run_cfg was rebuilt from the idea alone, so the tuned value was
    dropped while `incumbent` kept the score it earned. Every later experiment
    was then measured against a bar its own config could not reach.
    """
    from agent.controller import _compose_cfg

    base = {"lr": 3e-4}                       # what an exploit win adopted
    assert _compose_cfg(0, base, None)["lr"] == 3e-4, "tuned value dropped"
    assert _compose_cfg(0, base, {"group_size": 5})["lr"] == 3e-4, \
        "tuned value dropped when the idea set a different parameter"
    assert _compose_cfg(0, base, {"lr": 1e-3})["lr"] == 1e-3, \
        "an idea must still be able to set the parameter deliberately"
    assert _compose_cfg(7, base, None)["seed"] == 7
    print("  exploit-tuned parameters survive into later cycles OK")


def test_at_chance_result_is_not_scientific_evidence():
    """A run that learned nothing must not be allowed to weaken a belief.

    Both cases below are real, from run 10. Contrastive learning satisfied its
    contract, trained 5 epochs and scored 0.4926 against a 0.4834 random floor
    -- it cleared the floor by a hair and was filed as SCIENTIFIC, which is the
    only class permitted to weaken a hypothesis. What failed was the
    implementation, not the idea. Embedding dropout scored 0.5945 in the same
    run: that one IS evidence, and must stay scientific.
    """
    from agent.reflector import classify_failure

    hist = [{"epoch": i} for i in range(7)]
    assert classify_failure("ok", True, {"primary": 0.4926}, 0.6001, hist) \
        == "optimisation", "a result at chance was treated as evidence"
    assert classify_failure("ok", True, {"primary": 0.5945}, 0.6001, hist) \
        == "scientific", "a genuine near-miss stopped counting as evidence"
    print("  at-chance results excluded from scientific evidence OK")


def test_timeout_cannot_be_swallowed():
    """A guard's own exception must not be catchable by the code it guards.

    Both timeouts derived from Exception, and every probe in instrument.py sits
    inside `except Exception`. The alarm fired inside the first model build, the
    handler filed it as a probe error, and because signal.alarm is one-shot the
    NEXT build ran uncapped. One cycle spent 35 minutes past a 120s limit.
    """
    import time as _t
    from solution.instrument import _MeasureTimeout, _rearm
    from solution.runner import _CycleTimeout

    for T in (_MeasureTimeout, _CycleTimeout):
        assert not issubclass(T, Exception), \
            f"{T.__name__} is catchable by `except Exception`"
        swallowed = True
        try:
            try:
                raise T()
            except Exception:                  # what the probe handlers look like
                pass
            else:
                swallowed = False
        except T:
            swallowed = False
        assert not swallowed, f"{T.__name__} was swallowed by a probe handler"

    # and a consumed one-shot alarm must not leave a later probe uncapped
    try:
        _rearm(_t.monotonic() - 1)
    except _MeasureTimeout:
        pass
    else:
        raise AssertionError("_rearm did not fire on an already-expired deadline")
    print("  timeouts survive `except Exception`, alarm re-armed per probe OK")


def test_contract_cannot_forbid_its_own_intervention():
    """An invariant must not pin a quantity the postcondition requires to move.

    Run 13, real: three of the highest-scoring plans of the run were blocked by
    `embedding_dim_total unchanged` / `model_n_params unchanged` while the idea
    was "add a list-length feature". Adding a field necessarily adds embeddings.
    One was rejected for moving embedding_dim_total by ONE (40260 -> 40261).
    Contract satisfaction fell to 25%, the lowest of any run, and the blocks were
    OUR bug rather than the agent's.
    """
    from agent.specifier import sanitise_contract

    c = sanitise_contract({
        "postconditions": [{"metric": "n_feature_fields", "op": "changed"}],
        "invariants": [{"metric": "embedding_dim_total", "op": "unchanged"},
                       {"metric": "model_n_params", "op": "unchanged"}]})
    inv = {i["metric"] for i in c["invariants"]}
    assert "embedding_dim_total" not in inv, "contract forbids its own effect"
    assert "model_n_params" not in inv, "contract forbids its own effect"
    assert "train_rows" in inv, "mandatory invariants must survive"

    # a loss change must not be pinned on the quantities a loss change moves
    c2 = sanitise_contract({
        "postconditions": [{"metric": "loss_fn_name", "op": "changed"}],
        "invariants": [{"metric": "initial_loss", "op": "unchanged"}]})
    assert "initial_loss" not in {i["metric"] for i in c2["invariants"]}


def test_grouping_postcondition_dropped_when_unreachable():
    """`train_group_size_*` cannot move under row batching -- do not gate on it.

    Run 13, real: four cycles blocked on
    `train_group_size_median changed 1.0 -> got 1.0`. Under the reference
    batch_mode="row" the median is 1.0 by construction and eval_sized_groups()
    is never called, so no train.py edit alone can move it. The contract was
    unsatisfiable before the coder wrote a line.
    """
    from agent.specifier import sanitise_contract

    row = sanitise_contract(
        {"postconditions": [{"metric": "train_group_size_median", "op": "changed"}],
         "invariants": []}, {"batch_mode": "row"})
    assert not row["postconditions"], "gated on an unreachable target"

    user = sanitise_contract(
        {"postconditions": [{"metric": "train_group_size_median", "op": "changed"}],
         "invariants": []}, {"batch_mode": "user", "group_size": 5})
    assert user["postconditions"], "a reachable grouping target must survive"
    print("  contracts cannot forbid or over-reach their own intervention OK")



def test_cascade_is_mechanism_conditioned():
    """After a win, WHAT became stale depends on what actually changed.

    All three cases are real. Run 11 accepted a temporal-decay FEATURE, the
    generic cascade swept `lr` anyway, and 3e-4 tied the incumbent exactly --
    four training runs to establish that nothing was stale. Adding a
    deterministic feature does not change gradient scale; changing the batching
    or the objective does.
    """
    from agent.exploiter import stale_candidates

    feature, _ = stale_candidates({"n_feature_fields": 1})
    assert "lr" not in feature, \
        "a feature addition must not trigger a learning-rate sweep (run 11)"
    assert "k" in feature or "l2" in feature

    batching, _ = stale_candidates({"train_group_size_median": 1,
                                    "optimiser_steps_per_epoch": 1})
    assert "lr" in batching, \
        "changing batch construction DOES make the step size stale (+0.0010)"

    objective, _ = stale_candidates({"initial_loss": 1, "loss_fn_name": 1})
    assert "lr" in objective, "an objective change alters gradient magnitude"

    # nothing measured -> rule nothing out, rather than guess
    unknown, _ = stale_candidates(None)
    assert len(unknown) >= 4
    print("  exploit cascade conditioned on what actually changed OK")


def test_observe_coverage_reaches_the_decisive_tools():
    """OBSERVE must not be able to miss the two tools that reveal the answer.

    Measured across 14 logged cycles: `cold_start_rates` was picked in 100% of
    them -- a measured dead end -- while `list_length_distribution` was picked in
    14% and `usable_group_fraction` in 0%. Those two are the only tools that
    expose the train/eval list-size mismatch, and nothing downstream can name a
    problem it was never shown.
    """
    from agent.analyst import COVERAGE_SCHEDULE, TOOL_CATEGORY
    from solution import eda

    scheduled = {c["tool"] for sched in COVERAGE_SCHEDULE for c in sched}
    assert {"list_length_distribution", "usable_group_fraction"} <= scheduled, \
        "the decisive tools are not guaranteed to run"
    for t in scheduled:
        assert t in eda.TOOLS, f"scheduled tool {t} does not exist"
    for t in TOOL_CATEGORY:
        assert t in eda.TOOLS, f"categorised tool {t} does not exist"
    print("  OBSERVE coverage reaches the decisive tools OK")


def test_anomaly_board_remembers_across_cycles():
    """A repeated diagnosis must accumulate, not be rediscovered.

    Replays run 11: the list-length mismatch was named in cycles 6, 11, 12 and
    16 -- four fresh discoveries, no follow-through -- while a temporal drift was
    filed under the SAME problem class and must stay separate.
    """
    import tempfile
    from agent.anomalies import AnomalyBoard

    b = AnomalyBoard(tempfile.mkdtemp())
    lists_a = {"problem_class": "train/eval distribution mismatch",
               "statement": "training mean list length 43.54 vs validation 5.58"}
    lists_b = {"problem_class": "train/eval distribution mismatch",
               "statement": "63.699% of validation users have at most 5; train 43.54"}
    drift = {"problem_class": "train/eval distribution mismatch",
             "statement": "unseen users 3.617% in test vs 1.593% in valid"}
    for c, p in ((6, lists_a), (11, lists_b), (12, lists_a), (16, lists_a), (18, drift)):
        b.observe([p], c)

    assert len(b.items) == 2, f"expected 2 distinct anomalies, got {len(b.items)}"
    top = b.unresolved()[0]
    assert top["sightings"] == 4, "re-sightings did not merge"
    assert b.confidence(top) > 0.8, "repeated sightings must raise confidence"

    # a no-op must NOT retire an anomaly -- the idea was never actually tested
    b.record_attempt(6, "curriculum-learning", faithful=False, outcome="no-op")
    assert b.unresolved()[0]["status"] == "unresolved"
    before = b.confidence(b.unresolved()[0])
    b.record_attempt(12, "instance-weighting", faithful=True, outcome="0.6009")
    assert b.confidence(b.unresolved()[0]) < before, \
        "a faithful failure should lower confidence; a no-op should not"
    print("  anomaly board accumulates and separates correctly OK")


def test_planner_prefers_the_minimal_implementation():
    """Scoring must rank the direct fix above the elaborate one.

    Run 11 chose curriculum learning over a direct grouping change on the same
    diagnosis, four times, and every choice changed nothing measurable.
    """
    from agent.planner import score_plan

    minimal = {"directness": 5, "fidelity": 5, "isolation": 5, "cheapness": 5}
    middling = {"directness": 3, "fidelity": 4, "isolation": 4, "cheapness": 4}
    elaborate = {"directness": 2, "fidelity": 2, "isolation": 2, "cheapness": 2}
    assert score_plan(minimal) > score_plan(middling) > score_plan(elaborate)

    # a plan that cannot name the quantity it moves must lose to one that can,
    # even when it looks more direct -- no-ops are the dominant failure
    vague = {"directness": 5, "fidelity": 1, "isolation": 5, "cheapness": 5}
    concrete = {"directness": 3, "fidelity": 5, "isolation": 3, "cheapness": 3}
    assert score_plan(vague) == 0.0, \
        "a plan naming no measurable quantity cannot be contracted -- disqualify it"
    assert score_plan(concrete) > 0

    print("  planner prefers the minimal, verifiable implementation OK")



def test_exploiter_hardcodes_no_winning_values():
    """The exploiter must search, not recall.

    It tunes generic parameter NAMES over candidate grids; it must not contain
    the values a human found (lr 2e-4, group_size 5), or a 'clean' run would be
    handed the answer.
    """
    from agent import exploiter

    src = (ROOT / "agent" / "exploiter.py").read_text()
    for banned in ("2e-4", "0.0002", "0.6038", "0.6031"):
        assert banned not in src, f"exploiter contains a known answer: {banned}"

    # grids must straddle, not point at, an answer
    assert len(exploiter.TUNABLE["lr"]) >= 4
    assert len(exploiter.TUNABLE["group_size"]) >= 3
    assert exploiter.MAX_TRIALS <= 5, "an exploit branch must stay cheap"
    print("  exploiter searches rather than recalls OK")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    a = ap.parse_args()

    print("offline contract tests:")
    test_convergence_rule()
    test_scoring_matches_official()
    test_label_is_long_view()
    test_no_feedback_leakage()
    test_revision_state_machine()
    test_revision_cannot_switch_technique()
    test_seed_tiebreak()
    test_semantic_contract_discriminates()
    test_contract_cannot_demand_the_impossible()
    test_contract_targets_cover_loss_changes()
    test_changed_op_verifies_intervention_not_direction()
    test_failure_classification()
    test_exploit_branch_reachable()
    test_exploit_branch_executes()
    test_exploit_win_survives_the_next_cycle()
    test_timeout_cannot_be_swallowed()
    test_at_chance_result_is_not_scientific_evidence()
    test_contract_cannot_forbid_its_own_intervention()
    test_grouping_postcondition_dropped_when_unreachable()
    test_cascade_is_mechanism_conditioned()
    test_observe_coverage_reaches_the_decisive_tools()
    test_anomaly_board_remembers_across_cycles()
    test_planner_prefers_the_minimal_implementation()
    test_exploiter_hardcodes_no_winning_values()
    test_leakcheck_catches_the_real_leak()
    test_leakcheck_allows_train_only_target_encoding()
    test_implausible_gain_threshold()

    if a.data_dir:
        print("data contract tests:")
        test_alignment(a.data_dir)
        test_train_only_statistics(a.data_dir)
        test_no_leaky_statistics_file(a.data_dir)
        test_row_count_invariant(a.data_dir)
        test_no_phantom_contract_metrics(a.data_dir)
    else:
        print("  (skipping data tests -- pass --data_dir to run them)")
    print("\nall passed")
