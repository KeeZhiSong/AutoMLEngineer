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
Converged when validation primary has not improved by more than **ε = 0.002**
over **3 consecutive** cycles. Note the tension: no single step on this task has
ever gained that much, so every genuine win is sub-ε and the run stops shortly
after succeeding. Convergence is not evaluated on a winning cycle.

## Guards, each added because something got past the previous set

| guard | what it caught |
|---|---|
| AST leak scan (pre-execution) | 11 attempts to read a feedback column as an input, across 8 runs |
| row-count invariant | a patch that padded the scored set 124,909 -> 451,647 rows |
| semantic contract | patches that read as the intervention and changed nothing: 101 of 374 experiments |
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

- `video_features_statistic_pure.csv`, **leakage hazard.** Undated aggregate
  counters spanning the whole collection window, including test.
  `long_time_play_cnt / show_cnt` is close to a per-video `long_view` rate
  computed partly on test labels. A contract test asserts we never load it.
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
listwise objective and retuned learning rate. The run converged in 13 of the 50
permitted iterations on 105,304 tokens and 59 minutes. Across V4–V6 the loop
improves on **8 of 11 clean-reference runs**; the three generations before that
produced zero improvements in 70+ cycles.

The shipped pipeline is **V6**. V7 and V8 were built, tested over three runs and
reverted, they fixed the failures they targeted without producing a win. That
work is preserved on branch `v7-v8-investigation`; see `RESULTS.md`.

The test split has been scored **zero** times, the CSV holds test predictions,
but the test metric is deliberately withheld until the config is final.

Full results, the run-by-run reliability record, the seven directions closed by
measurement, and an honest accounting of every false win, the agent's two and
the harness's ten, are in [`RESULTS.md`](RESULTS.md).

## Limitations, and what I would do with more time

**The agent diagnoses better than it implements.** In one 25-cycle run it named
the decisive problem four times and converted it into a working patch zero
times. Across the whole project it attempted the family the human fix belongs to
(listwise objectives, group sizing, list-length handling) in 50 experiments and
landed none of them. Selection and implementation are the constraint, not
perception.

**The margin is thin.** +0.0015 is roughly two seed standard deviations. It is
5-seed confirmed and reproducible, but it is a small effect on a saturated
benchmark, not a breakthrough.

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
