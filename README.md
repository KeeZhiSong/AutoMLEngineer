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
agent/        analyst → classifier → inventor → coder → reflector
lib/          techniques.jsonl + beliefs.py (claims WITH contradicting evidence)
tests/        contract tests for the silent-failure cases
workspace/    experiments.jsonl, INSIGHTS.md, DEAD_ENDS.md, summary.json
```

## How the loop works

1. **Analyst** picks 3 of 9 measurement tools and runs them on the data.
2. **Classifier** turns the numbers into a *named problem* — e.g. "train/eval
   distribution mismatch, dimension = list length, ratio 6x". A fact suggests
   nothing; a named problem is what an intervention attaches to.
3. **Inventor** proposes interventions open-ended, with **no technique menu in
   front of it**, then may *request* literature for its problem class. Pull, not
   push — retrieval after invention informs a hypothesis the agent already owns.
4. **Coder** rewrites exactly one module. Syntax-, import- and leak-checked
   before it is allowed to cost a training run.
5. **Reflector** decides keep/revert **arithmetically** — only if primary beats
   the incumbent by more than one published std (0.0008). The LLM writes the
   explanation, never the verdict.

Guards, each added because something got past the previous set:

| guard | what it caught |
|---|---|
| AST leak scan (pre-execution) | 5 attempts to read a feedback column as an input |
| row-count invariant | a patch that padded the scored set 124,909 → 451,647 rows |
| 5-seed tie-break | a "win" that averaged back to baseline over five seeds |
| implausible-gain flag | any single-cycle jump > 0.02 of a 0.247 total headroom |
| per-cycle timeout | one experiment that ate 98% of a run's compute |
| divergence abort | training that falls below the random-scoring floor |
| loop lockfile | a second writer corrupting an in-flight experiment |

## What we deliberately do not use

- `video_features_statistic_pure.csv` — **leakage hazard.** Undated aggregate
  counters spanning the whole collection window, including test.
  `long_time_play_cnt / show_cnt` is close to a per-video `long_view` rate
  computed partly on test labels. A contract test asserts we never load it.
- `user_features_pure.csv` — legal, but a term constant within a user cannot
  reorder that user's list. Only useful as a user×item cross.
- `log_random_*.csv` — legal as an *unbiased cross-check* only; never trained on.

## Status

The baseline is reproduced to within the published seed variance:

| | GAUC | nDCG@5 | primary | published |
|---|---|---|---|---|
| ours, valid | 0.6670 | 0.5358 | **0.6014** | 0.6016 |
| ours, test | 0.6621 | 0.5286 | **0.5953** | 0.5946 |

**Best result: valid primary 0.6034 (seed 0), 0.6028 over 5 seeds — +0.0012
over the baseline.** Listwise softmax with evaluation-length training groups.
Against a matched control that is +0.0025 with a paired t of 11.3 on 4 df, all
five seeds positive and complete separation; it also held across four separate
temporal windows (14/14 paired seeds). `submission.csv` is generated from this
config and passes the organisers' own checker.

The test split has been scored **zero** times — the CSV holds test predictions,
but the test metric is deliberately withheld until the config is final.

## Limitations

- **The +0.0012 was found by a human diagnostic, not by the agent.** The agent
  independently *names* the right problem from its own measurements; it has not
  yet converted that diagnosis into code that beats the reference. Both halves
  belong in any honest description of this system.
- The margin is thin: +0.0012 is ~3.0 standard errors once the baseline's own
  0.0008 seed variance is propagated, not the 6.8 it looks like if you treat the
  published number as exact.
- The convergence rule is gated on having accepted one improvement first; a
  literal reading ends any non-improving search at exactly 3 iterations. This is
  our interpretation, documented in `solution/scoring.py`.
- Bonus benchmarks (KuaiRand-1k / 27k) not attempted.
- The coder needs `gpt-4o`; `gpt-4o-mini` cannot write a correct grouped gradient.

## Model routing

Each agent role can use a different model, because they need different things.
The keep/revert verdict is arithmetic, so the Reflector only writes prose and
does not need a strong model; the Coder is the measured bottleneck and does.

| role | task | default |
|---|---|---|
| analyst | pick 3 tools, report numbers | `gpt-4o-mini` |
| classifier | name the problem | `gpt-4o` |
| inventor | propose interventions | `gpt-4o` |
| **coder** | **write correct numpy** | `gpt-4o` ← the bottleneck |
| reflector | explain a decision already made | `gpt-4o-mini` |

```bash
export AMRA_MODEL_CODER=o3            # point one role anywhere
export AMRA_MODEL_UNIFORM=gpt-4o      # or force every role to one model
```

Model ids beginning `claude-` route to Anthropic automatically (needs
`ANTHROPIC_API_KEY` and the `anthropic` SDK). Per-role token usage is reported
in `summary.json` under `tokens_by_role`, alongside the routing actually used.

Measured so far: `gpt-4o-mini` cannot write a correct grouped gradient at all.
`gpt-4o` names the right problem in 11 of 16 scored cycles and 0 of 16 of its
implementations cleared the accept margin — so implementation, not diagnosis, is
where a stronger model would pay.
