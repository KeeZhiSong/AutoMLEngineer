# Autonomous ML Research Agent — KuaiRand-Pure

TikTok TechJam 2026, Track 2. An LLM-driven agent that runs the MLE iteration
loop on its own — read the problem, engineer features, train and tune, evaluate,
reflect, revise — and tries to beat the organisers' official baseline.

## The task (pinned by the Starter Kit, not by us)

| | |
|---|---|
| Dataset | **KuaiRand-Pure** (required; 1k/27k are bonus, not attempted) |
| Task | Within-user ranking over logged impressions — no full-catalogue retrieval |
| Relevance label | **`long_view`** (0/1) |
| Metrics | **GAUC** and **nDCG@5**; **primary = mean of the two** |
| Splits | train `20220408–21` / valid `20220422–28` / test `20220429–0508` |
| Baseline to beat | FM, **test primary 0.5946** / valid 0.6016 |
| Convergence | ε = 0.002 over N = 3 iterations |

> **CONFIRMED BY THE ORGANISERS (27 Aug 2026).** The problem statement's earlier
> "NDCG@10 / Recall@50, click = positive" text was incorrect and has been
> withdrawn. Their engineer: *"CWM does not contain a Recall implementation
> anywhere in the repository, its NDCG is reported at k=1/3/5, and it scores
> long_view2 rather than a click-based signal. There were no official
> NDCG@10 / Recall@50 scores to find, because that combination was never
> actually implemented."* Recall@50 is removed as not measurable on this task;
> the label is the native `long_view` column; CWM is an optional reference only.
> Everything below was inferred from the Starter Kit before this confirmation
> and needed no change.

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
              interpretation — a tool that flags a finding leaks the answer.
agent/        analyst → classifier → inventor → planner → specifier → coder
              → reflector → exploiter (on a keep only)
              anomalies.py: named problems persist across cycles
lib/          techniques.jsonl + beliefs.py (claims WITH contradicting evidence)
              llm.py: per-role model routing
tests/        contract tests for the silent-failure cases
artifacts/    human-best/ and agent-best*/ — both reproducible by score
workspace*/   experiments.jsonl, anomalies.jsonl, INSIGHTS.md, summary.json
```

The shipped pipeline is **V6**. V7 and V8 were built, tested over three runs and
reverted — they fixed the failures they targeted without producing a win. That
work is preserved on branch `v7-v8-investigation`, and `RESULTS.md` explains
what it established.

## How the loop works

1. **Analyst** measures the data. Cycles 1–2 are *scheduled* to cover the
   distribution, label, ranking-context and cardinality families; free choice
   resumes at cycle 3. Scheduling exists because it was measured that the
   analyst chose `cold_start_rates` — a dead end — in 100% of 14 logged cycles,
   and the two tools revealing the decisive fact in 14% and 0%. Nothing
   downstream can name a problem it was never shown.
2. **Classifier** turns the numbers into a *named problem* — e.g. "train/eval
   distribution mismatch, dimension = list length, ratio 6x". A fact suggests
   nothing; a named problem is what an intervention attaches to.
3. **Anomaly board** keeps named problems alive across cycles, merging
   re-sightings and recording what has been tried. Without it one run diagnosed
   the same decisive problem in four separate cycles as four fresh discoveries,
   and followed up on none. A no-op does **not** retire an anomaly: an idea that
   was never really tested is still open.
4. **Inventor** proposes interventions open-ended, with **no technique menu in
   front of it**, then may *request* literature for its problem class. Pull, not
   push — retrieval after invention informs a hypothesis the agent already owns.
5. **Planner** enumerates four ways to implement the committed hypothesis and
   scores them on directness, fidelity, isolation and cost. A plan that cannot
   name a measurable quantity it moves is disqualified, not merely penalised —
   it could not be given a contract, so it could not be verified.
6. **Specifier** writes a semantic contract *before the code exists* — a handful
   of measurable claims about what the patch must change. `instrument.py` checks
   them in 0.4s without training, so a patch that changes nothing never costs a
   run.
7. **Coder** rewrites exactly one module. Syntax-, import- and leak-checked
   before it is allowed to cost a training run.
8. **Reflector** decides keep/revert **arithmetically** — only if primary beats
   the incumbent by more than one published std (0.0008). The LLM writes the
   explanation, never the verdict. Failures are classified
   implementation / optimisation / **scientific**, and only the last may weaken
   a belief — a broken patch is not evidence against the idea it botched.
9. **Exploiter** (on a keep only) spends up to 4 trials retuning a parameter the
   intervention made stale. Which parameter is derived from *what measurably
   changed*: a new feature points at capacity and regularisation, a batching or
   objective change points at the step size. Fired on five live wins — one from a
   deliberately handicapped reference, to exercise the branch on demand — and
   declined all five as inside the accept margin. **Zero false wins.** In one
   case a trial looked like +0.0012 on seed 0 and averaged to +0.0004 over five
   seeds, which is exactly the error the margin exists to catch.

Guards, each added because something got past the previous set:

| guard | what it caught |
|---|---|
| AST leak scan (pre-execution) | 11 attempts to read a feedback column as an input, across 8 runs |
| row-count invariant | a patch that padded the scored set 124,909 → 451,647 rows |
| 5-seed tie-break | a "win" that averaged back to baseline over five seeds |
| implausible-gain flag | any single-cycle jump > 0.02 of a 0.247 total headroom |
| per-cycle timeout | one experiment that ate 98% of a run's compute |
| divergence abort | training that falls below the random-scoring floor |
| loop lockfile | a second writer corrupting an in-flight experiment |
| semantic contract | patches that read as the intervention and change nothing — the single most common failure |
| failure classification | a broken implementation recorded as evidence against the idea it botched |
| contract sanitisation | contracts forbidding their own intervention, e.g. "add a feature" gated on `embedding_dim_total unchanged` |
| exit-restore + preflight-by-score | a run launched from the wrong code, verified by SCORE rather than by filename |

## What we deliberately do not use

- `video_features_statistic_pure.csv` — **leakage hazard.** Undated aggregate
  counters spanning the whole collection window, including test.
  `long_time_play_cnt / show_cnt` is close to a per-video `long_view` rate
  computed partly on test labels. A contract test asserts we never load it.
- `user_features_pure.csv` — legal, but a term constant within a user cannot
  reorder that user's list. We tested the obvious remedy, an explicit user×item
  cross, and **measured exactly 0.0000**: an FM already computes every pairwise
  field interaction, so a hand-built cross of two existing fields is a coarser
  copy of something the model has. Only a genuinely new field can help.
- `log_random_*.csv` — legal as an *unbiased cross-check* only; never trained on.

## Status

The baseline is reproduced to within the published seed variance:

| | GAUC | nDCG@5 | primary | published |
|---|---|---|---|---|
| ours, valid | 0.6670 | 0.5358 | **0.6014** | 0.6016 |
| ours, test | 0.6621 | 0.5286 | **0.5953** | 0.5946 |

**Submitted result: valid primary 0.6038 over 5 seeds — +0.0022 over the
baseline.** Listwise softmax with evaluation-length training groups (+0.0012),
plus a learning rate retuned for that new objective (+0.0010; the inherited 1e-3
had been tuned for a pointwise loss). Against a matched control the paired t is
11.3 on 4 df, all five seeds positive, complete separation; the grouping effect
also held across four separate temporal windows (14/14 paired seeds).
`submission.csv` is generated from this config and passes the organisers' checker.
Both configurations are reproducible by score, not by assertion:

    cp artifacts/human-best/*.py solution/ && python3 tools_preflight.py --expect winning
    cp artifacts/agent-best/*.py  solution/ && python3 tools_preflight.py --expect agent

**The agent independently reaches 0.6031** from a 0.6014 reference, with no
answers in its technique library, 5-seed confirmed in-loop, zero manual
interventions — by a different route (a reweighted logloss). Across V4–V6 it
improves on **7 of 10 clean-reference runs**; the three generations before that
produced zero improvements in 70+ cycles.

The shipped pipeline is **V6**. V7 and V8 were built, tested over three runs and
reverted — they fixed the failures they targeted without producing a win. That
work is preserved on branch `v7-v8-investigation`; see `RESULTS.md`.

The test split has been scored **zero** times — the CSV holds test predictions,
but the test metric is deliberately withheld until the config is final.

Full results, the run-by-run reliability record, the seven directions closed by
measurement, and an honest accounting of every false win — the agent's two and
the harness's ten — are in [`RESULTS.md`](RESULTS.md).

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

Measured, same architecture and budget with only the coder changed: `gpt-4o`
produced 3 crashes, 0 keeps and used 313K tokens; **`gpt-5.5` produced 0
crashes, a keep at 0.6030, and used 146K** — cheaper overall, because better
code converges sooner. On an isolated grouped-listwise task where `gpt-4o`
crashed, `gpt-5.5` scored 0.6037 against 0.6036 for the human implementation.
`gpt-4o-mini` cannot write a correct grouped gradient at all.

Implementation, not diagnosis, is the constraint — and more precisely the
SPECIFICATION handed to the coder: given a detailed brief `gpt-5.5` implements
the decisive mechanism correctly, given the planner's two sentences the same
model produces no-ops.
