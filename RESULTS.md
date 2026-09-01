# Results — Autonomous ML Research Agent, KuaiRand-Pure

TikTok TechJam 2026, Track 2. All scores are **validation** `primary =
mean(GAUC, nDCG@5)` on the official split, computed by the organisers' unmodified
`official/evaluate.py` (SHA-256 pinned in `official/VENDORED.sha256`).

**The test split has been scored zero times.** `submission.csv` contains test
predictions; the metric is deliberately withheld so no selection decision can be
contaminated by it.

## Headline

| configuration | valid primary | vs baseline | how it was found |
|---|---|---|---|
| random scoring | 0.4834 | −0.1182 | floor |
| item popularity | 0.5807 | −0.0209 | floor |
| **official FM baseline** | **0.6016** | — | the bar |
| our reproduction of it | 0.6014 | −0.0002 | reproduces to within seed noise |
| **agent, fully unaided — SUBMITTED** | **0.6038** | **+0.0022** | loop, zero interventions |
| human-tuned (same score, different route) | 0.6038 | +0.0022 | two-step diagnosis |
| oracle ceiling | 0.8484 | +0.2468 | perfect ranking |

The ceiling is 0.8484, not 1.0: 27.1% of evaluation users have no positive label,
so their nDCG is 0 for any model. Remaining headroom above the baseline is 0.247.

## The submitted result, and how it was reached

Two sequential changes, not one complicated model.

| step | change | delta | why |
|---|---|---|---|
| 1 | listwise softmax with **evaluation-length training lists** (`group_size=5`, `batch_mode="user"`) | +0.0012 | training ranked ~31-item lists; evaluation ranks ~4. The objective was being optimised in the wrong regime. |
| 2 | **retune the step size for the new objective**, `lr` 1e-3 → 2e-4 | +0.0010 | grouping changed the gradient scale, so a learning rate tuned for a pointwise loss was stale |

Step 2 is the one that is easy to miss, and it is why the agent has an EXPLOIT
stage: a win creates a new local research problem.

Verification: 5 seeds, paired against a matched control, t = 11.3 on 4 df, all
five seeds positive, complete separation. The grouping effect also held across
four separate temporal windows (14/14 paired seeds).

Reproduce:

    cp artifacts/human-best/*.py solution/
    python3 tools_preflight.py --expect winning     # confirms BY SCORE

## The agent's own result

**0.6031, from a 0.6014 reference, with no task-specific answers in its
library and zero manual interventions.** Confirmed at 5 seeds by the loop itself
before acceptance — the tie-break fires automatically when a result lands within
2σ of the incumbent.

    cp artifacts/agent-best/*.py solution/
    python3 tools_preflight.py --expect agent

It reaches this by a different route from the human one (a reweighted loss, and
in a later run a class-weighted loss). It has **not** rediscovered the two-step
path above.

## Reliability

Every run below starts from the same reproduced 0.6014 reference with no
task-specific answers in the agent's library and zero manual interventions.

| architecture | runs | improved | best |
|---|---|---|---|
| V1–V3 — technique menu, measurement tools, problem naming | 3 | **0** | — |
| V4 — semantic contracts, failure classification | 3 | 3 | 0.6031 |
| V5 — + exploit cascade | 2 | 1 | 0.6022 |
| **V6 — + observation coverage, anomaly board, implementation planner** | **5** | **3** | **0.6031** |
| V7 — + codebase capability map, capability-aware scoring | 2 | 0 | — |
| V8 — + stronger coder on the V7 stack | 1 | 0 | — |

V4–V6 combined: **7 improvements in 10 runs.**

V1–V3 produced zero improvements in 70+ cycles; that architectural change is
real. n is small and "reliable" is not claimed.

**V7 and V8 were tested and reverted.** Both fixed the failures they targeted —
the direct grouping intervention went from ranked last (2.75) to ranked first
(5.00), and the compound objective-plus-config mechanism was assembled correctly
for the first time — and neither produced a win in three runs. The most likely
explanation is that V7 redirected effort from easy interventions that
occasionally landed toward the direct one, which is harder to implement. The
shipped pipeline is V6; the V7/V8 work is preserved on branch
`v7-v8-investigation` and its diagnostic value is described in Limitations.

### The coder is the measured constraint, and the model choice matters

Same architecture, same 25-cycle budget, only the coder changed:

| | `gpt-4o` | `gpt-5.5` |
|---|---|---|
| crashes from generated code | **3** | **0** |
| keeps | 0 | **1 (0.6030)** |
| tokens | 313K | **146K** |
| cycles to convergence | 25 (hit the cap) | 13 |

Cheaper overall despite a higher per-call cost, because better code reaches a
verdict sooner. On an isolated grouped-listwise task where `gpt-4o` crashed,
`gpt-5.5` scored 0.6037 against 0.6036 for the human implementation. `gpt-5.5`
is now the default coder.

## The submitted run

Deliverables 3 and 4 describe one run. This is it.

| | |
|---|---|
| run log | `workspace_final/experiments.jsonl` |
| configuration | `artifacts/agent-best/` |
| **iterations used** | **5 research cycles of the 50 cap**, stopped by convergence |
| logged experiments | **13** = the 5 cycles + 8 EXPLOIT trials; 12 were scored training runs |
| **total tokens (in + out)** | **105,304** |
| **agent wall-clock** | **59 minutes** |
| **GPU-hours** | **0.81** (CPU only; no accelerator was used) |
| **manual interventions** | **0** |

Two accepted interventions, in order:

| cycle | technique | module | primary |
|---|---|---|---|
| 1 | curriculum-learning | `train.py` | 0.6023 |
| 3 | length-normalization | `features.py` | 0.6040 |

### Validation-best score, required benchmark

Measured over **20 seeds**, so the metric rows and the primary come from the
same measurement:

| metric | ours (20-seed mean) | sd | official baseline | absolute delta |
|---|---|---|---|---|
| GAUC | 0.6702 | 0.0004 | 0.6674 | **+0.0028** |
| nDCG@5 | 0.5373 | 0.0002 | 0.5357 | **+0.0016** |
| primary (mean of the two) | **0.6038** | 0.0003 | 0.6016 | **+0.0022** |
| **score_dataset** (equal-weighted mean of the metric deltas) | | | | **+0.0022** |

Range over the 20 seeds: primary min 0.6033, max 0.6043, and **all 20 beat both
the baseline and the previous best agent result**. Per-seed values are in
`workspace_final/seed_sweep_20.json`.

For reference, seed 0 alone, the value the loop accepted at the time, reads
GAUC 0.6705 / nDCG@5 0.5375 / primary 0.6040. It sits slightly above the mean,
which is exactly why the submitted figure is the 20-seed one.

The value at acceptance was a single seed. It cleared the tie-break band by
0.0001 and so skipped confirmation, and a neighbouring configuration in the same
exploit sweep read 0.6040 on seed 0 and measured 0.6034 across five. It was
therefore measured at n=20 before being used for anything.

Bonus benchmarks (KuaiRand-1k, KuaiRand-27k) were not attempted.

> **On the two iteration counts.** `workspace_final/summary.json` reports
> `"cycles": 13`. That field carried the ledger's *record* count, not the loop
> counter: an accepted cycle writes one record for itself and one more for each
> EXPLOIT trial it opens, so the two diverge on exactly the runs that won. The
> loop ran **5 cycles** and logged **13 experiments**. Fixed at source in
> `lib/ledger.py` (the key is now `experiments`) and in `agent/controller.py`,
> which was spreading the totals over its own `cycles` value. The run itself is
> unaffected. `RUNLOG.md` derives both figures directly from the records.

Reproduce:

    cp artifacts/agent-best/*.py solution/
    python3 tools_preflight.py --expect agent      # confirms by score
    python3 make_submission.py --out submission.csv --hold-test-score

## Cost: what the revision mechanism was worth

The loop could revise a near miss up to twice before moving on. Measured across
every run in this repository, that mechanism fired **93 times and was never once
accepted**; the best a revision ever scored was 0.6019, below the incumbent it
was trying to beat. It consumed **675,815 tokens, 16% of the project's entire
spend**, for zero accepted results.

Disabling it (`MAX_REVISIONS = 0`) produced the cheapest keep-bearing run of the
project:

| run | revisions | tokens | contract satisfaction | keeps |
|---|---|---|---|---|
| 06-clean-run-14cyc | on | 142,612 | 50% | 1 |
| v6c_gpt55 | on | 146,488 | 62% | 1 |
| v6 | on | 165,813 | 60% | 1 |
| **final** | **off** | **105,304** | **92%** | **2** |

A 26-36% reduction against comparable runs. The saving appears in the run total
rather than the per-cycle rate, because revisions consumed whole extra cycles
against patches that had already failed.

93 samples of a mechanism that never fired is not proof it cannot work. It is
sufficient reason to stop paying for it. The anomaly board now does the job
revisions were meant to do: a near miss stays on the board with its attempt
history, available to a later cycle without re-running the loop against a patch
that already failed.

## Resource report

| | |
|---|---|
| research runs executed | 21 with a committed run log |
| experiments, all runs | **387** (219 completed, 102 stopped by contract, 49 crashed and recovered, 15 skipped, 2 timed out) |
| LLM tokens, all runs | **4,341,703** |
| GPU-hours, all runs | **5.20** (CPU-only; numpy, no accelerator) |
| single run cost | 82K–466K tokens, 0.02–0.46 GPU-hours |
| best single run | 0.6030 at 146K tokens, converged in 9 cycles (13 logged experiments) |
| manual interventions | **0** in every run |
| hardware | one laptop CPU; the baseline trains in ~15s |

Dependencies: `numpy`, `pandas`, `openai`. The pipeline itself is numpy-only.

Model routing — each role gets what it needs, not the most expensive option:

| role | model | why |
|---|---|---|
| analyst | `gpt-4o-mini` | *measured to be a bottleneck; see limitations* |
| classifier / inventor | `gpt-4o` | naming a problem is not the constraint |
| **coder** | **`gpt-5.5`** | **the measured constraint — 3 crashes to 0, and cheaper per run** |
| reflector | `gpt-4o-mini` | its verdict is arithmetic; it writes only prose |

## What the guards caught

Each exists because something got past the previous set.

| guard | caught |
|---|---|
| AST leak scan | **11 attempts** to read a feedback column as an input, across 8 runs |
| row-count invariant | a patch that padded the scored set 124,909 → 451,647 rows |
| semantic contract | patches that changed nothing measurable — the single most common failure |
| 5-seed tie-break | "wins" of +0.0010 and +0.0012 that averaged to +0.0004 |
| implausible-gain flag | single-cycle jumps > 0.02 of a 0.247 headroom |
| per-cycle timeout | one experiment consuming 98% of a run's compute |
| divergence abort | training below the 0.4834 random floor |

Before the leak guard existed, one such patch scored 0.6449 and was accepted.

## Limitations

1. **The agent now matches the human result, by a different route.** Both reach
   0.6038. The human path was a listwise objective on evaluation-length training
   lists plus a retuned learning rate; the agent reached the same score through
   curriculum learning over sequence length and a length-normalisation feature.
   It did not rediscover the human path, and that distinction should be kept.
2. **The margin is thin.** +0.0022 is ~3.0 standard errors once the baseline's
   own 0.0008 seed variance is propagated, not the 6.8 it appears.
3. **Seven directions are closed by measurement** — feature stacking, seed
   ensembling, temporal decay (real but subsumed), cold start, auxiliary click
   targets, hand-built user×item crosses, and the remaining inherited
   hyperparameters. The practical ceiling looks like ~0.604.
4. **The agent diagnoses better than it implements.** In one 25-cycle run it
   named the decisive problem four times and converted it to a working patch
   zero times. Contract satisfaction runs 25–88% across runs.
   The V7/V8 investigation narrowed this further: given a detailed
   specification `gpt-5.5` implements the decisive mechanism correctly (0.6037),
   but given the planner's two-sentence brief the same model produces no-ops.
   The binding constraint is the quality of the specification handed to the
   coder, not the coder alone.
5. **OBSERVE gates discovery.** Before V6, the analyst chose `cold_start_rates`
   in 100% of cycles and the two tools that reveal the decisive fact in 14% and
   0%. The cheapest model in the pipeline was deciding what the expensive ones
   could think about.
6. **The convergence rule sits above the effect size.** ε = 0.002, but no single
   step on this task has gained that much, so any real win is sub-ε and the run
   stops shortly after succeeding.
7. Bonus benchmarks (KuaiRand-1k / 27k) not attempted.

## Honest accounting of failures

The agent produced **two** classes of false win (label leakage, metric gaming).
The harness — us — produced **at least ten**, nearly all of one species:
measuring or preserving state against the wrong reference. That asymmetry is the
main engineering lesson of the project, and `handoff/08` documents every case.
