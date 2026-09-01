# Autonomous ML Research Agent, KuaiRand-Pure

TikTok TechJam 2026, Track 2. An LLM-driven agent that runs the MLE iteration
loop on its own, read the problem, engineer features, train and tune, evaluate,
reflect, revise, and tries to beat the organisers' official baseline.

## The task (pinned by the Starter Kit, not by us)

| | |
|---|---|
| Dataset | **KuaiRand-Pure** (required; 1k/27k are bonus, not attempted) |
| Task | Within-user ranking over logged impressions, no full-catalogue retrieval |
| Relevance label | **`long_view`** (0/1) |
| Metrics | **GAUC** and **nDCG@5**; **primary = mean of the two** |
| Splits | train `20220408–21` / valid `20220422–28` / test `20220429–0508` |
| Baseline to beat | FM, **test primary 0.5946** / valid 0.6016 |
| Convergence | ε = 0.002 over N = 3 iterations |

## Quick start

```bash
# data (no registration required)
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz -C KuaiRand/

python3 run.py test                    # 10 contract tests
python3 run.py baseline                # reproduce the official FM
export OPENAI_API_KEY=...
python3 run.py loop --max-iterations 20
python3 run.py submit --out submission.csv
```

Dependencies: `numpy`, `pandas`, `openai`. The official baseline itself is
numpy-only.

## Layout

```
official/     Starter Kit, VERBATIM. SHA-256 in VENDORED.sha256. Never edited.
              evaluate.py is the sole scoring authority.
solution/
  dataset.py  FROZEN. Loads all 20 log columns; asserts row-order identity with
              official.data.load() so submission row_ids stay valid.
  scoring.py  FROZEN. The only doorway to the official metric.
  runner.py   FROZEN. Wires the pipeline; returns a crash as a result.
  features.py AGENT-EDITABLE  ─┐
  model.py    AGENT-EDITABLE   ├ what the loop rewrites
  train.py    AGENT-EDITABLE  ─┘
  eda.py      FROZEN. 9 dataset-inspection tools. Neutral numbers, no
              interpretation, a tool that flags a finding leaks the answer.
agent/        analyst.py     OBSERVE, 3 of 9 neutral measurement tools
              classifier.py  CLASSIFY, measurements -> a named problem
              anomalies.py   BOARD, problems persist across cycles
              inventor.py    INVENT, open-ended, optional literature pull
              planner.py     PLAN, 4 implementations, scored
              specifier.py   SPECIFY, the contract, before the code
              coder.py       CODE, rewrites one editable module
              reflector.py   JUDGE, arithmetic keep/revert + failure type
              exploiter.py   EXPLOIT, on a keep only
              controller.py  the loop itself
lib/          techniques.jsonl + beliefs.py (claims WITH contradicting evidence)
              llm.py: per-role model routing
tests/        contract tests for the silent-failure cases
artifacts/    human-best/ and agent-best*/, both reproducible by score
workspace*/   experiments.jsonl, anomalies.jsonl, INSIGHTS.md, summary.json
```

The shipped pipeline is **V6**. V7 and V8 were built, tested over three runs and
reverted, they fixed the failures they targeted without producing a win. That
work is preserved on branch `v7-v8-investigation`, and `RESULTS.md` explains
what it established.

## The research loop

One pass is a **cycle**. A run is 25 cycles, or until the convergence rule fires.
There is always an *incumbent*, the best configuration so far, and each cycle
tests exactly one change against it.

```
                          ┌──────────── RETRIEVAL (optional) ───────┐
                          │                                         ▼
OBSERVE ─► CLASSIFY ─► ANOMALY BOARD ─────────────────────────► INVENT ─► PLAN
                                                                            │
   ┌────────────────────────────────────────────────────────────────────────┘
   ▼
SPECIFY ─► CODE ─► VERIFY ─► RUN ─┬─► ERROR ─► RECOVERY ──┐
                                  └─► SUCCESS ─► JUDGE    │
                                                  │       │
                                    REJECT ◄──────┴──────►│ KEEP
                                                          ▼
                                          WIN ANALYSIS ─► EXPLOIT (retune │ dose)
                                                          ▼
                                                       MEMORY ─► next cycle
```

### 1 · OBSERVE: `agent/analyst.py`
Runs 3 of 9 tools from `solution/eda.py`. Every tool returns **neutral numbers
and no interpretation**. A tool that flags a finding has leaked the answer.

Tool families: distribution, ranking context, labels, temporal, cardinality,
cold-start/sparsity. **Cycles 1–2 are scheduled** to guarantee coverage; free
choice resumes at cycle 3, with unvisited families surfaced in the prompt.

*Why scheduled:* across 14 logged cycles the analyst chose `cold_start_rates` (a measured dead end) in **100%** of them,
and the two tools that reveal the
decisive train/eval list-size mismatch in **14% and 0%**. Nothing downstream can
name a problem it was never shown.

Writes: `observations` onto the cycle's ledger record.

### 2 · CLASSIFY: `agent/classifier.py`
Turns measurements into a **named problem** with a type, dimension and
magnitude. `train median 31, valid median 4` is a fact; *"train/eval mismatch,
dimension = ranking-context length, ratio ~8×"* is something an intervention can
attach to. May return **no problem**, in which case the cycle is skipped rather
than filled with something plausible.

### 3 · ANOMALY BOARD: `agent/anomalies.py`
Named problems persist across cycles. A re-sighting **merges** into the existing
entry (matched on problem class plus a shared statistic) and raises confidence;
a faithful failed attempt lowers it. Each entry carries its full attempt history.

*The load-bearing rule:* **a no-op does not retire an anomaly.** If the patch
changed nothing, the idea was never tested, so the problem stays open.

Writes: `workspace/anomalies.jsonl`.

### 4 · INVENT: `agent/inventor.py`
Proposes interventions **open-ended, with no technique library visible**. It may
then *request* literature for its problem class, pull, not push, so retrieval
informs a hypothesis the agent already owns. Retrieval is a second model call,
not a corpus lookup. Commits to one intervention with a kill criterion.

### 5 · PLAN: `agent/planner.py`
Enumerates **four implementations** of the committed hypothesis and scores them
on directness, fidelity, isolation and cost.

A plan that cannot name a measurable quantity it moves is **disqualified, not
penalised**. No contract could be written for it, so it could not be verified
before spending a training run.

### 6 · SPECIFY: `agent/specifier.py`
Writes the **semantic contract before the code exists**: postconditions the
patch must cause, plus invariants it must not break. If the coder wrote first it
would author a gate its own patch satisfies.

`sanitise_contract()` drops postconditions nothing can satisfy and invariants
that contradict the intervention, an "add a feature" idea cannot be gated on
the embedding table staying the same size.

### 7 · CODE: `agent/coder.py`
Rewrites exactly one of `features.py`, `model.py`, `train.py`. Syntax-, import-
and leak-checked before it may cost a training run. Everything else, loader,
scorer, runner, is frozen, so the agent can change the model but never what is
measured.

### 8 · VERIFY: `solution/instrument.py`
**24 measurements in 0.4 seconds, no training.** Checks the patch is valid, safe
(no feedback-column leakage, row alignment intact) and *faithful*, did it move
the quantity it claimed? A contract failure skips training entirely.

### 9 · RUN: `solution/runner.py`
Trains and scores through the frozen pipeline, 900s cap. A crash is returned as
a **result**, not an exception. `ERROR -> RECOVERY` restores the module from
snapshot and the loop continues.

### 10 · JUDGE: `agent/reflector.py`
Keep or revert, decided **arithmetically**: the result must beat the incumbent by
more than one published seed std (**0.0008**). Anything landing within 0.0016 is
automatically re-run on **5 seeds** first. The model writes the explanation,
never the verdict.

Failures are typed `implementation` / `optimisation` / **`scientific`**, and only
the last may weaken a belief, a patch that crashed says nothing about the idea.
A near miss within 0.010 is held for up to 2 revisions.

### 11 · EXPLOIT: `agent/exploiter.py` *(on a keep only)*
A win is a new local research problem. Up to 4 trials **retune** a parameter the
intervention made stale, or **dose** its strength to tell a plateau from a lucky
point. Which parameter is derived from what measurably changed: a new feature
points at capacity and regularisation; a change to the objective or batching
points at the step size.

Fired on five live wins; declined all five as inside the accept margin. **Zero
false wins.**

### 12 · MEMORY
`experiments.jsonl` (one record per cycle, with its observations and diff),
`anomalies.jsonl`, `beliefs.jsonl` (claims stored *with* contradicting
evidence), `INSIGHTS.md` / `DEAD_ENDS.md`, and `summary.json` at exit.

### Convergence
The organisers' default rule, implemented literally. Converged when validation
primary has not improved by more than **ε = 0.002** over **3 consecutive**
scored cycles. We use the published default rather than declaring our own ε, N
or minimum-iteration floor.

The rule is **cumulative**, as the organisers' FAQ specifies: the best score
over the last N scored iterations is compared against the best from *before*
that window, not against the immediately preceding value. `has_converged()` in
`solution/scoring.py` is exactly that:

```python
best_before = max(primary_history[:-n])
recent_best = max(primary_history[-n:])
return (recent_best - best_before) <= epsilon
```

**A cycle that produces no validation score does not advance or reset the
window.** The FAQ requires this, and the submitted run exercised it: cycle 4
failed its semantic contract, never trained, and appended nothing to
`primary_history`, so the window was unaffected. It still counts toward the
50-iteration cap.

The run stopped on this rule after 5 cycles:

```
best before the last 3 : 0.6023   [0.6023, 0.6023, 0.5981]
best in the last 3     : 0.6040   [0.6040, 0.6040, 0.6037]
improvement            : 0.0017  <=  0.002  ->  CONVERGED
```

Note the tension: no single step on this task has ever gained more than ε, so
every genuine win is sub-ε and the run stops shortly after succeeding. The
50-iteration budget was never the binding constraint; the stopping rule was.
Convergence is not evaluated on a winning cycle.

## What the agent proposed, and whether it came true

Every cycle must state a hypothesis, a named source technique and a **numeric
prediction** before any code is written. The prediction is recorded in
`experiments.jsonl` at proposal time, so it cannot be revised after the result.
That makes the run log scoreable against itself.

Here is the whole submitted run, its own predictions against its own outcomes:

| cycle | what it proposed | predicted | actual | verdict |
|---|---|---|---|---|
| 1 | `curriculum-learning` — grow training sequence length gradually, so the model adapts to varied list lengths | **+0.0050** | **+0.0009** | KEEP, 5.9x optimistic |
| 2 | `dynamic-interaction-modeling` — an `interaction_depth` feature for user engagement level | **+0.0030** | **−0.0042** | REVERT, wrong sign |
| 3 | `length-normalization` — pad training lists to the validation median to close a measured 6x mismatch | **+0.0040** | **+0.0017** | KEEP, 2.4x optimistic |
| 4 | `sample-weighting` — down-weight over-represented positives to align with the metric | **+0.0060** | *never tested* | contract violated, no training |
| 5 | `data-augmentation` — synthesise positive interactions from item-feature correlations | **+0.0020** | **−0.0002** | REVERT, wrong sign |

**Four predictions were tested. None came true.** Both wins were over-predicted,
by 5.9x and 2.4x; both losses were predicted positive and came back negative.
The agent's mean predicted gain was +0.0035 against a mean actual of +0.0006.

**That is the point, not an embarrassment.** The loop reached +0.0022 while its
own forecasts were wrong every single time, because *nothing downstream believes
the prediction*. The keep/revert gate is arithmetic on the measured score, the
tie-break re-runs on 5 seeds, and the semantic contract checks whether the patch
moved the quantity it named. A loop that acted on its predictions would have
kept cycle 5 and chased cycle 4 hardest of all. Calibration is a nice property;
**not requiring it is a design one.**

**Cycle 4 is the sharpest case.** It carried the largest prediction of the run,
+0.0060, and never ran: the contract check found `initial_loss` unchanged at
0.694438, so the patch had not implemented the intervention it described. The
loop recorded "no evidence about the hypothesis" rather than a negative result,
which is the distinction that keeps a botched implementation from being filed as
a refuted idea.

### One cycle end to end, in the agent's own words

Cycle 3, the change that produced the submitted configuration. Nothing here was
written by a human:

**Problem** (from its own EDA, via CLASSIFY):
> train/eval distribution mismatch: the percentage of users with list lengths of
> at most 5 is 63.699% in validation, while it is 10.618% in training, a 6x
> mismatch in how list lengths are distributed.

**Hypothesis:**
> Implement list length normalization via padding to the validation median to
> reduce distribution mismatch impact on ranking performance.

**Kill criterion**, committed before the run:
> If the implementation yields a performance change of less than +0.001 or shows
> a negative impact on within-user ordering, it should be abandoned.

**Prediction:** +0.004 primary.  **Measured:** +0.0017 — under its prediction,
over its kill criterion, so it survived on the rule it set for itself.

It considered **four implementations** and scored them before choosing (median
padding won on directness 5 / fidelity 4 / isolation 5), and it requested
literature on its own initiative, citing importance sampling and
covariate-shift reweighting for the problem class it had named. The full
reasoning chain, the candidates it rejected and the retrieved literature are in
`workspace_final/summary.json` under `best_checkpoint.idea`.

## Guards, each added because something got past the previous set

| guard | what it caught |
|---|---|
| AST leak scan (pre-execution) | 11 attempts to read a feedback column as an input, across 8 runs |
| row-count invariant | a patch that padded the scored set 124,909 -> 451,647 rows |
| semantic contract | patches that read as the intervention and changed nothing: 102 of 387 experiments |
| 5-seed tie-break | apparent wins of +0.0010 and +0.0012 that averaged to +0.0004 |
| failure classification | a broken implementation recorded as evidence against the idea it botched |
| contract sanitisation | contracts forbidding their own intervention |
| implausible-gain flag | any single-cycle jump > 0.02 of a 0.247 total headroom |
| per-cycle timeout | one experiment that ate 98% of a run's compute |
| divergence abort | training that falls below the random-scoring floor |
| exit-restore + preflight-by-score | a run launched from the wrong code, verified by SCORE, never by filename |
| loop lockfile | a second writer corrupting an in-flight experiment |

Before the leak guard existed, one such patch scored **0.6449** and was accepted.

## The search ledger (published page)

`docs/index.html` visualises every experiment across every run: a grid of one
cell per experiment, the score timeline, and the full run table. It is
**generated, never hand-written**:

```bash
python3 tools_build_ledger.py           # rebuild docs/index.html
python3 tools_build_ledger.py --check   # exit 1 if the page is stale
```

A GitHub Actions workflow regenerates it whenever a run log changes, so the
published page cannot drift from the data it describes. Enable GitHub Pages on
`main` + `/docs` to serve it.

## What we deliberately do not use

- `video_features_statistic_pure.csv`, **declined on an unverifiable window.**
  Its 51 per-video counters are undated, and neither the dataset nor the
  Starter Kit documents the period they cover. `long_time_play_cnt / show_cnt`
  is very close to a per-video `long_view` rate, and FAQ 2.9.3 forbids "feature
  statistics computed over" the test split. Since we cannot verify the counters
  exclude the test window, we did not use them, and a contract test asserts we
  never load the file.

  **We tested the leakage hypothesis and it did not hold up, which we report
  because the result cuts against us.** The question is whether the statistic
  carries information about the scored period beyond what train legitimately
  reveals. It does correlate with the test-period rate after controlling for the
  train-period rate (partial r = +0.276) — but that is confounded, because the
  statistic is a platform-wide, far lower-variance estimate of a stable quantity
  and would beat a noisy train estimate with no test data in it at all. Against
  a control that isolates exactly that effect — the statistic's extra
  explanatory power over a *held-out train week*, given the other train week,
  where leakage is impossible — the figure is **+0.312, higher than the +0.276
  it gets on test**, and the control is biased upward. So the evidence is
  consistent with the statistic being a better measurement, not a leak.

  The forfeit therefore rests on the window being undocumented rather than on
  demonstrated contamination. That is a weaker justification than we originally
  wrote here, and it is the accurate one.

  **And we measured what declining it cost.** Adding the statistic as two
  decile-bucketed FM fields, edges fitted on train only, through the frozen
  runner, 3 seeds each: reference **0.6014**, with the statistic **0.6019**, a
  delta of **+0.0005**. That is below one published seed std (0.0008), so it
  would not have cleared this loop's own keep threshold. The decision cost
  approximately nothing *here*. It would likely be worth more to a
  gradient-boosted model, which can use a continuous per-video rate far better
  than a bucketed FM field can, so this number sizes the forfeit for our
  pipeline and not in general.
- `user_features_pure.csv`, legal, but a term constant within a user cannot
  reorder that user's list. We tested the obvious remedy, an explicit user×item
  cross, and **measured exactly 0.0000**: an FM already computes every pairwise
  field interaction, so a hand-built cross of two existing fields is a coarser
  copy of something the model has. Only a genuinely new field can help.
- `log_random_*.csv`, legal as an *unbiased cross-check* only; never trained on.

## Status

The baseline is reproduced to within the published seed variance:

| | GAUC | nDCG@5 | primary | published |
|---|---|---|---|---|
| ours, valid | 0.6670 | 0.5358 | **0.6014** | 0.6016 |
| ours, test | 0.6621 | 0.5286 | **0.5953** | 0.5946 |

**A human-tuned configuration reaches the same 0.6038** by a different route:
listwise softmax with evaluation-length training groups (+0.0012), plus a
learning rate retuned for that new objective (+0.0010; the inherited 1e-3 had
been tuned for a pointwise loss). Against a matched control the paired t is 11.3
on 4 df, all five seeds positive, complete separation; the grouping effect also
held across four separate temporal windows (14/14 paired seeds). It is kept in
`artifacts/human-best/` as the benchmark the agent is measured against, and is
**not** what is submitted.

All three configurations are reproducible by score, not by assertion:

    cp artifacts/agent-best/*.py       solution/ && python3 tools_preflight.py --expect agent
    cp artifacts/human-best/*.py       solution/ && python3 tools_preflight.py --expect winning
    cp artifacts/agent-best-0.6031/*.py solution/  # the previous agent best

**The agent independently reaches 0.6038** from a 0.6014 reference, with no
answers in its technique library and zero manual interventions: 20-seed mean
0.6038, sd 0.0003, every seed above the baseline. **`submission.csv` is generated
from this configuration** and passes the organisers' checker.

It reaches the human score by a different route: curriculum learning over
sequence length plus a length-normalisation feature, rather than the human's
listwise objective and retuned learning rate. The run converged in 5 of the 50
permitted iterations (13 logged experiments, 8 of them EXPLOIT trials) on
105,304 tokens and 59 minutes. Across V4–V6 the loop
improves on **8 of 11 clean-reference runs**; the three generations before that
produced zero improvements in 70+ cycles.

The shipped pipeline is **V6**. V7 and V8 were built, tested over three runs and
reverted, they fixed the failures they targeted without producing a win. That
work is preserved on branch `v7-v8-investigation`; see `RESULTS.md`.

The test split was scored **exactly once**, after the configuration was frozen.
It was unreachable during the search: the runner is never called with
`evaluate_test`, so every keep/revert decision and the 20-seed confirmation ran
on validation alone. Scored once at the end, `submission.csv` reaches **test
primary 0.5972** against the baseline's 0.5946, a **score_dataset of +0.0026**
(GAUC +0.0029, nDCG@5 +0.0022). The validation gain of +0.0022 therefore held,
and grew slightly, on test.

Full results, the run-by-run reliability record, the seven directions closed by
measurement, and an honest accounting of every false win, the agent's two and
the harness's ten, are in [`RESULTS.md`](RESULTS.md).

## Where each deliverable lives

| deliverable | file |
|---|---|
| written project description | [`DEVPOST.md`](DEVPOST.md) |
| per-iteration run log | [`RUNLOG.md`](RUNLOG.md), generated from `workspace_final/experiments.jsonl` by `tools_build_runlog.py`; the unified diff applied in each cycle is in [`runlog_diffs/`](runlog_diffs/) |
| raw run logs, every run | `workspace_final/`, `workspace_*/`, `archive/runs/` |
| manual-intervention summary | [`RUNLOG.md`](RUNLOG.md) and [`RESULTS.md`](RESULTS.md): **0** |
| final model output | [`submission.csv`](submission.csv), test split, 170,588 rows, passes `official/submit.py --check`; scored once by `tools_score_test.py` at **test primary 0.5972**, score_dataset **+0.0026** |
| results table + resource usage | [`RESULTS.md`](RESULTS.md) |
| published search ledger | [`docs/index.html`](docs/index.html) |

```bash
python3 tools_build_runlog.py           # rebuild RUNLOG.md + runlog_diffs/
python3 tools_build_runlog.py --check   # exit 1 if it is stale
```

## Team member contributions

Solo entry. Kee Zhi Song built the whole submission: the frozen evaluation
harness around the organisers' `official/`, the agent loop and all eleven of its
stages, the guards, the human-tuned benchmark configuration the agent is
measured against, and every document here. There are no other contributors.

## Limitations, and what I would do with more time

**The agent diagnoses better than it implements.** In one 25-cycle run it named
the decisive problem four times and converted it into a working patch zero
times. Across the whole project it attempted the family the human fix belongs to
(listwise objectives, group sizing, list-length handling) in 50 experiments and
landed none of them. Selection and implementation are the constraint, not
perception.

**The margin is thin.** +0.0022 is about 3.0 standard errors once the
baseline's own 0.0008 seed variance is propagated, not the 6.8 it appears at
first glance. It is 20-seed confirmed and reproducible, but it is a small effect
on a saturated benchmark, not a breakthrough.

**It does not improve every run.** 8 of 11 clean-reference runs under V4-V6.
Run-to-run variance exceeds the difference between architecture generations, so
no version is separable from another by result alone.

**Seven directions are closed by measurement**, so the practical ceiling for
this approach looks like ~0.604 against an oracle of 0.8484. Getting past that
needs a different mechanism, not more tuning.

**The specification is the bottleneck, not the coder.** Given a detailed brief,
a strong model implements the decisive mechanism correctly (0.6037). Given the
planner's two-sentence brief, the same model produces no-ops. That is the
sharpest lever I did not get to pull.

### With more time

1. **Write better specifications, not better plans.** The planner picks the
   right intervention far more often than the coder implements it. A richer
   brief (the causal control point, the existing helpers to call, the gradient
   contract to satisfy) is cheaper than a bigger model.
2. **Let one intervention span the objective and its activating config.** The
   human fix is a listwise loss *and* evaluation-length grouping; neither helps
   alone. A loop that changes exactly one thing per cycle cannot express it.
3. **Give OBSERVE a reason to vary.** It self-corrects over a long run, but the
   opening cycles still decide much of what gets explored.
4. **Bonus benchmarks.** KuaiRand-1k and 27k were not attempted.

## Model routing

Each agent role can use a different model, because they need different things.
The keep/revert verdict is arithmetic, so the Reflector only writes prose and
does not need a strong model; the Coder is the measured bottleneck and does.

| role | task | default |
|---|---|---|
| analyst | pick 3 tools, report numbers | `gpt-4o-mini` |
| classifier | name the problem | `gpt-4o` |
| inventor | propose interventions | `gpt-4o` |
| **coder** | **write correct numpy** | `gpt-5.5` ← the bottleneck |
| reflector | explain a decision already made | `gpt-4o-mini` |

```bash
export AMRA_MODEL_CODER=o3            # point one role anywhere
export AMRA_MODEL_UNIFORM=gpt-4o      # or force every role to one model
```

Model ids beginning `claude-` route to Anthropic automatically (needs
`ANTHROPIC_API_KEY` and the `anthropic` SDK). Per-role token usage is reported
in `summary.json` under `tokens_by_role`, alongside the routing actually used.

### Why every default is an OpenAI model

Because that is the subscription I had, and it offered the best balance of price
and quality available to me, not because OpenAI models were compared against
others and found better for this task. **No cross-provider comparison was run**,
so nothing here should be read as a claim about relative model quality.

The routing layer is provider-agnostic by design: a role is a name, and any
model id can be pointed at it. The `gpt-4o` vs `gpt-5.5` comparison below is a
within-provider result and is the only model comparison this project actually
measured.

Measured, same architecture and budget with only the coder changed: `gpt-4o`
produced 3 crashes, 0 keeps and used 313K tokens; **`gpt-5.5` produced 0
crashes, a keep at 0.6030, and used 146K**. Cheaper overall, because better
code converges sooner. On an isolated grouped-listwise task where `gpt-4o`
crashed, `gpt-5.5` scored 0.6037 against 0.6036 for the human implementation.
`gpt-4o-mini` cannot write a correct grouped gradient at all.

Implementation, not diagnosis, is the constraint. More precisely it is the
SPECIFICATION handed to the coder: given a detailed brief `gpt-5.5` implements
the decisive mechanism correctly, given the planner's two sentences the same
model produces no-ops.
