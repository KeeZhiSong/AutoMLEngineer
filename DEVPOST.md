## Inspiration

Plenty of agents write code. Far fewer do **research**, and the difference is not
code quality. It is knowing which of your own results to believe.

A research loop that cannot tell a real improvement from a lucky random seed will
confidently accumulate nonsense. On this benchmark the noise floor is one seed
standard deviation of 0.0008, and the entire remaining headroom above the
official baseline is 0.247. An agent that trusts a single measurement will
"improve" its way straight into a fantasy.

So I set out to build an agent that runs the whole machine-learning loop
unattended on a real benchmark, and, more importantly, one that is *hard to
fool*, including by itself, and including by me.

## What it does

### The problem statement, and how this addresses it

Track 2 asks for an autonomous agent that performs the machine-learning
engineering loop on the KuaiRand benchmarks: read the problem, engineer
features, train and tune, evaluate, reflect, revise, and beat the organisers'
official baseline, with autonomy and reproducibility as first-class criteria.

This solution addresses it directly:

| the ask | what this does |
|---|---|
| run the MLE loop autonomously | 11 stages from OBSERVE to MEMORY, driven by 5 LLM roles. **Zero manual interventions** across all 21 runs, verified by a per-record flag the loop writes itself |
| beat the official baseline | **0.6038** validation primary against **0.6016**, a **+0.0022** absolute gain, from a 0.6014 reference the agent reproduced itself |
| iterate and reflect | keep/revert decided arithmetically against one published seed std; failures typed so a botched implementation never counts as evidence against the idea it botched |
| stop sensibly | the organisers' own convergence rule, epsilon = 0.002 over N = 3 |
| be reproducible | every configuration is verified **by score**, never by asserting what a file contains. The scorer is the organisers' `evaluate.py`, vendored unmodified and SHA-256 pinned |
| be honest | the test split was unreachable during the search and scored **once**, after the config was frozen; every false win, mine and the agent's, is documented rather than quietly dropped |

### The task

The agent improves a recommender on **KuaiRand-Pure**. For each user it ranks
only the videos that user actually saw. No full-catalogue retrieval, no
candidate generation. The label is `long_view` (0/1) and the score is

```
primary = (GAUC + nDCG@5) / 2
```

The organisers' factorisation-machine baseline scores **0.6016** on validation.
That is a harder bar than it looks:

| configuration | valid primary |
|---|---|
| random scoring | 0.4834 |
| item popularity | 0.5807 |
| **official FM baseline** | **0.6016** |
| perfect ranking (oracle) | 0.8484 |

The ceiling is 0.8484 rather than 1.0 because **27.1% of evaluation users have no
positive label at all**. Their nDCG is zero for any model that could ever be
written. Real headroom above the baseline is 0.247, and the organisers' own
published ablations move this metric by less than 0.002. On this benchmark a
+0.001 gain is meaningful and a +0.05 jump means you have found a leak.

### The result

**0.6038, measured over 20 seeds** (sd 0.0003, min 0.6033, max 0.6043, every
seed above the baseline), reached from a 0.6014 reference with **zero manual
interventions**. That is **+0.0022**, and it **matches a human-tuned
configuration exactly**.

Broken out against the official baseline:

| metric | agent (20-seed mean) | official FM baseline | absolute delta |
|---|---|---|---|
| GAUC | 0.6702 | 0.6674 | **+0.0028** |
| nDCG@5 | 0.5373 | 0.5357 | **+0.0016** |
| primary | **0.6038** | 0.6016 | **+0.0022** |

Every number in that table is a 20-seed mean, so the metrics and the primary
come from one measurement rather than two. Seed 0 alone, the value the loop
accepted at the time, reads 0.6705 / 0.5375 / 0.6040; it sits a little above
average, which is why the submitted figure is the 20-seed one.

**On the hidden test set**, scored once after the configuration was frozen:

| metric | agent | official FM baseline | absolute delta |
|---|---|---|---|
| GAUC | 0.6639 | 0.6610 | **+0.0029** |
| nDCG@5 | 0.5304 | 0.5282 | **+0.0022** |
| primary | **0.5972** | 0.5946 | **+0.0026** |

The gain held on test and grew slightly, +0.0022 on validation against +0.0026
on test. Test was unreachable during the search, so nothing was ever selected
on it.

It got there by a different route from the human configuration. The human path
was a listwise objective trained on evaluation-length lists, plus a learning rate
retuned for it. The agent's two accepted interventions were:

| cycle | technique | module | primary |
|---|---|---|---|
| 1 | curriculum-learning | `train.py` | 0.6023 |
| 3 | length-normalization | `features.py` | 0.6040 |

It matched the score. It did not rediscover the method, and I keep those two
claims separate everywhere in this project.

## How I built it

One pass through the loop is a **cycle**. There is always an incumbent, the best
configuration so far, and each cycle tests exactly one change against it.

```
                    +------------- RETRIEVAL (optional) -------+
                    |                                          v
OBSERVE -> CLASSIFY -> ANOMALY BOARD --------------------> INVENT -> PLAN
                                                                      |
   +------------------------------------------------------------------+
   v
SPECIFY -> CODE -> VERIFY -> RUN -+-> ERROR -> RECOVERY --+
                                  +-> SUCCESS -> JUDGE    |
                                                |         |
                                  REJECT <------+------>  | KEEP
                                                          v
                                        WIN ANALYSIS -> EXPLOIT
                                                          v
                                                       MEMORY -> next cycle
```

Each stage exists because of a failure I measured, not because it seemed like
good architecture.

**OBSERVE** (`agent/analyst.py`) runs 3 of 9 tools that return neutral numbers
and no interpretation, because a tool that flags a finding has leaked the answer.
Left to choose freely, the model picked a *measured dead end* (`cold_start_rates`)
in 100% of 14 logged cycles, and the two tools that reveal the decisive fact in
14% and 0%. Nothing downstream can name a problem it was never shown, so cycles
1 and 2 are now scheduled for coverage and free choice resumes at cycle 3.

**CLASSIFY** (`agent/classifier.py`) turns measurements into a named problem with
a type, a dimension and a magnitude. `train median 31, valid median 4` is a fact;
*"train/eval mismatch, dimension = ranking-context length, ratio ~8x"* is
something an intervention can attach to. It may return no problem, in which case
the cycle is skipped rather than filled with something plausible.

**ANOMALY BOARD** (`agent/anomalies.py`) keeps named problems alive across
cycles. Without it, one run diagnosed the same decisive problem in cycles 6, 11,
12 and 16 as four fresh discoveries and followed up on none of them. The
load-bearing rule: **a no-op does not retire an anomaly.** If the patch changed
nothing, the idea was never tested, so the problem stays open.

**INVENT** (`agent/inventor.py`) proposes interventions with *no technique
library visible*, then may *request* literature for the problem it just named.
Pull, not push, so retrieval informs a hypothesis the agent already owns rather
than handing it one. It commits to a single intervention with a kill criterion.

**PLAN** (`agent/planner.py`) enumerates four implementations of that hypothesis
and scores them on directness, fidelity, isolation and cost. A plan that cannot
name a measurable quantity it moves is **disqualified, not penalised**: no
contract could be written for it, so it could never be verified before spending
a training run.

**SPECIFY** (`agent/specifier.py`) writes the semantic contract *before the code
exists*. If the coder wrote first, it would author a gate its own patch
satisfies. A sanitisation pass drops postconditions nothing can satisfy and
invariants that contradict the intervention, because an "add a feature" idea
cannot be gated on the embedding table staying the same size.

**CODE** (`agent/coder.py`) may rewrite only `features.py`, `model.py`,
`train.py`. The loader, the scorer and the runner are frozen, so the agent can
change the model but never what is measured.

**VERIFY** (`solution/instrument.py`) takes **24 measurements in 0.4 seconds**
with no training, checking that the patch is valid, safe (no feedback-column
leakage, row alignment intact) and *faithful*: did it move the quantity it
claimed? It stopped **102 of 387** experiments before they cost a training run.

**RUN** (`solution/runner.py`) trains and scores through the frozen pipeline
under a 900s cap, and returns a crash as a **result** rather than raising, so
`ERROR -> RECOVERY` restores the module from snapshot and the loop continues.

**JUDGE** (`agent/reflector.py`) decides arithmetically:

```
keep  <=>  primary > incumbent + 0.0008
```

where 0.0008 is one published seed standard deviation. Anything landing within
0.0016 is automatically re-run on five seeds first. The model writes the
explanation, never the verdict. Failures are typed `implementation`,
`optimisation` or `scientific`, and only the last may weaken a belief: a patch
that crashed says nothing about the idea it was trying to express.

**EXPLOIT** (`agent/exploiter.py`) fires only on a win, because a win is a new
local research problem. Up to four trials retune a parameter the intervention
made stale, or dose its strength to tell a plateau from a lucky point. It fired
on five live wins and declined all five as inside the accept margin: **zero false
wins**.

**MEMORY** writes `experiments.jsonl` (one record per experiment, with its
observations and the full patch), `anomalies.jsonl`, `beliefs.jsonl` (claims
stored *with* their contradicting evidence), `INSIGHTS.md`, `DEAD_ENDS.md`, and
`summary.json` at exit.

### The guards, and what each one caught

Every row here exists because something got past the previous set.

| guard | what it caught |
|---|---|
| AST leak scan, pre-execution | 11 attempts to read a feedback column as an input, across 8 runs |
| row-count invariant | a patch that padded the scored set from 124,909 to 451,647 rows |
| semantic contract | patches that read as the intervention and changed nothing: **102 of 387** experiments |
| 5-seed tie-break | apparent wins of +0.0010 and +0.0012 that averaged to +0.0004 |
| failure classification | a broken implementation recorded as evidence against the idea it botched |
| contract sanitisation | contracts that forbade their own intervention |
| implausible-gain flag | any single-cycle jump above 0.02 of a 0.247 total headroom |
| per-cycle timeout | one experiment that ate 98% of a run's compute |
| divergence abort | training that falls below the random-scoring floor |
| exit-restore, preflight-by-score | a run launched from the wrong code, caught by SCORE and never by filename |
| loop lockfile | a second writer corrupting an in-flight experiment |

Before the leak guard existed, one such patch scored **0.6449** and was accepted.

## Built with

### Development tools

- **VS Code** as the editor, with **Claude Code** in the terminal for
  refactoring and review.
- **git** and **GitHub** for version control. Two failed architecture
  generations are preserved on a branch (`v7-v8-investigation`) rather than
  deleted, because a negative result you cannot inspect is not a result.
- **GitHub Actions** (`.github/workflows/ledger.yml`) regenerates the published
  search-ledger page from the run logs on every push that touches a log, so the
  published page cannot drift from the data it describes.
- **macOS terminal**, `uv` for the virtual environment, standard Python
  tooling. Developed on Python 3.14.6.
- **No notebooks.** Every experiment runs through a script, so it is
  reproducible, loggable and diffable. Nothing in this project depends on cell
  execution order.

### APIs

**OpenAI Chat Completions** is the only external API. Five roles are each routed
to the model that role actually needs:

| role | model | job | tokens in the submitted run |
|---|---|---|---|
| analyst | `gpt-4o-mini` | pick measurement tools, report numbers | 6,412 |
| classifier | `gpt-4o` | turn measurements into a named problem | 19,264 |
| inventor | `gpt-4o` | propose interventions, request literature | 23,956 |
| **coder** | **`gpt-5.5`** | write correct numpy (the measured bottleneck) | **55,964** |
| reflector | `gpt-4o-mini` | explain a decision already made arithmetically | 1,318 |

The routing layer (`lib/llm.py`) is provider-agnostic: a role is just a name, any
model id can be pointed at it with an environment variable, and ids beginning
`claude-` route to the Anthropic SDK automatically.

OpenAI was the subscription I had, with the best price/quality balance available
to me. **No cross-provider comparison was run**, so nothing here is a claim about
relative model quality. The `gpt-4o` versus `gpt-5.5` comparison below is
within-provider and is the only model comparison this project actually measured.

### Libraries and frameworks

**`numpy`** and **`pandas`** only, plus the **`openai`** SDK (and `anthropic` on
the optional routing path). Everything else is the standard library: `ast` for
the leak scanner, `signal` for the per-cycle timeout, `difflib` for the run-log
diffs, `json` for the append-only ledger.

**No deep-learning framework.** The model is a factorisation machine written in
numpy, which is why 21 full research runs cost 5.2 CPU-core-hours on a laptop
and **zero GPU-hours**, with no accelerator involved at any point. The organisers' baseline is numpy-only and
I kept that property deliberately: it makes a full training run take about 15
seconds, and a loop that can afford to measure is a loop that can afford to
disbelieve itself.

### Datasets and assets

**KuaiRand-Pure** (Gao et al., CIKM 2022), the required benchmark, downloaded
from the public Zenodo mirror. No registration, no external data, no manually
labelled data, no pretrained weights.

Three files are loaded:

| file | use |
|---|---|
| `log_standard_4_08_to_4_21_pure.csv` | train split, 1,141,112 rows |
| `log_standard_4_22_to_5_08_pure.csv` | valid (124,909 rows) and test (170,588 rows) |
| `video_features_basic_pure.csv` | item-side static fields |

Three are **deliberately not used**, and the reasons are part of the solution:

- `video_features_statistic_pure.csv` is a **leakage hazard**. Its counters are
  undated aggregates spanning the whole collection window, test included, so
  `long_time_play_cnt / show_cnt` is close to a per-video `long_view` rate
  computed partly on test labels. A contract test asserts we never load it.
- `user_features_pure.csv` is legal but inert. Ranking happens *within* a user,
  so any term constant across that user's impressions cannot reorder their list.
  I tested the obvious remedy, an explicit user-by-item cross, and measured
  **exactly 0.0000**: an FM already computes every pairwise field interaction, so
  a hand-built cross of two existing fields is a coarser copy of something the
  model already has.
- `log_random_*.csv` is legal as an unbiased cross-check only, and was never
  trained on.

The organisers' Starter Kit is vendored **verbatim** in `official/` with its
SHA-256 recorded in `official/VENDORED.sha256`, and `official/evaluate.py` is the
sole scoring authority. Nothing in the agent can change what is measured.

## Challenges I ran into

**The agent tried to cheat, twice.** One patch read a feedback column as an input
feature and scored **0.6449**, pure label leakage, accepted before the guard
existed. Another padded the scored set from 124,909 rows to 451,647, changing
*what was measured* rather than the model. The fixes are an AST scan of what the
code actually reads (11 attempts blocked since) and a row-count invariant in the
frozen layer.

**But I produced more false wins than the agent did.** The agent produced two
classes. I produced at least ten, nearly all of one species: measuring or
preserving state against the wrong reference. My favourite is that the code which
generated our own submission was never saved to disk. I found this, rebuilt it
from the documented recipe, and verified the rebuild *by score* rather than by
reading the file. **It is easier to build a system that catches an agent
deceiving you than one that catches yourself.**

**The most instructive moment was a win that wasn't.** An exploit trial scored
0.6026 on the first seed against a 0.6014 incumbent, a clean +0.0012, past the
threshold. It landed inside the tie-break band, so the loop re-ran it on five
seeds:

```
0.6026  0.6019  0.6018  0.6014  0.6015   ->  mean 0.6018  (+0.0004)
```

Declined. The first measurement was not wrong, it just was not representative.
Later I chased a nearly identical sub-margin result myself and lost an hour
before it evaporated at twenty seeds. **The arithmetic gate was more disciplined
than the human operating it.**

**The same rule then protected a real result.** The final run's winning patch was
accepted on a *single* seed of 0.6040, having cleared the band by 0.0001. A
neighbouring configuration in the same sweep also read 0.6040 on seed 0 and
measured 0.6034 across five. So I tested the accepted one at 20 seeds before it
touched the submission: **mean 0.6038, sd 0.0003**. The caution was right to
apply, and this time the result survived it.

**Two architecture generations failed and were reverted.** A codebase capability
map (V7) fixed exactly the failure it targeted, moving the direct intervention
from ranked last to ranked first among candidates, and produced no win across
three runs. V8 put a stronger coder on the same stack, with the same outcome.
Both are preserved on a branch and reported rather than quietly dropped. The
shipped pipeline is V6.

## What I learned

**Diagnosis was never the bottleneck.** The clearest evidence is the family the
human fix belongs to: listwise objectives, group sizing, list-length handling.
Across the project the agent attempted that family **55 times without a single
acceptance**. In the final run it attempted it 10 more times and landed **two**,
which is how it reached 0.6038. The idea had been right for weeks. Selection and
implementation were what changed.

**More precisely, the constraint is the specification handed to the coder.**
Given a detailed brief, `gpt-5.5` implements the decisive mechanism correctly and
scores 0.6037. Given the planner's two-sentence brief, the same model produces
no-ops. That is the sharpest lever I did not get to pull.

**Model choice matters exactly where the bottleneck is.** Same architecture, same
25-cycle budget, only the coder changed:

| | `gpt-4o` | `gpt-5.5` |
|---|---|---|
| crashes from generated code | **3** | **0** |
| keeps | 0 | **1 (0.6030)** |
| tokens | 313K | **146K** |
| cycles to convergence | 25 (hit the cap) | 13 |

*Cheaper overall* despite a higher per-call price, because code that works
reaches a verdict sooner. `gpt-4o-mini` cannot write a correct grouped gradient
at all.

**Measure your own machinery, not just your model.** The loop could revise a near
miss twice before moving on. That mechanism fired **93 times across the project
and was never once accepted**; the best a revision ever scored was 0.6019, below
the incumbent it was trying to beat. It cost **675,815 tokens, 16% of total
spend**, for zero accepted results. Disabling it produced the cheapest
keep-bearing run of the project: **105,304 tokens** against 143K to 166K for
comparable runs, with the highest contract satisfaction (92%) and the best
result. The anomaly board now does the job revisions were meant to do.

**Negative results are the real output.** Seven directions are closed by
measurement, each with a mechanism rather than a shrug: feature stacking, seed
ensembling, temporal decay, cold start, auxiliary click targets, hand-built
user-by-item crosses, and the remaining inherited hyperparameters. My favourite
is the auxiliary click target, which *hurt* by -0.0020 because
`P(long_view = 1 | click = 0) = 0.003`, so clicked-but-not-long-viewed rows are
precisely the confusable class the metric exists to separate, not weak positives.

## Accomplishments I'm proud of

- **The agent matches a human-tuned configuration**, unaided, by a different
  route, and I can prove both halves of that sentence.
- **Zero manual interventions in 21 runs**, and that is a measured field on 387
  records rather than a claim I typed.
- **Zero false wins survived to the submission.** Every one was caught by a
  guard, and each guard is in the repository with the incident that motivated it.
- **The published ledger cannot lie.** `docs/index.html` visualises all 387
  experiments and is regenerated by CI from the logs, so it cannot drift from the
  data it describes.

## Resource usage

The submitted run:

| | |
|---|---|
| total tokens (input + output) | **105,304** |
| **agent wall-clock** (the scored compute measure) | **59 minutes** of the 6 h ceiling |
| iterations used | **5 research cycles of the 50 cap**, stopped by convergence |
| logged experiments | 13 = the 5 cycles + 8 exploit trials; 12 were scored training runs |
| GPU-hours | **0** — no GPU was used at any point |
| training compute | 0.81 CPU-core-hours |
| manual interventions | **0** |

The whole project, across 21 recorded runs:

| | |
|---|---|
| experiments | **387** (219 completed, 102 stopped by contract, 49 crashed and recovered, 15 skipped, 2 timed out) |
| accepted improvements | 10 |
| tokens | **4,341,703** |
| GPU-hours | **0** — no GPU was used at any point |
| training compute | 5.20 CPU-core-hours, one laptop CPU |
| manual interventions | **0** |

## What's next

1. **Write better specifications, not better plans.** The planner picks the right
   intervention far more often than the coder implements it. A richer brief (the
   causal control point, the existing helpers to call, the gradient contract to
   satisfy) is cheaper than a bigger model.
2. **Let one intervention span the objective and its activating config.** The
   human fix is a listwise loss *and* evaluation-length grouping, and neither
   helps alone. A loop that changes exactly one thing per cycle cannot express
   that, which is a structural limit rather than a tuning one.
3. **Give OBSERVE a reason to vary.** It self-corrects over a long run, but the
   opening cycles still decide much of what ever gets explored.
4. **The bonus benchmarks.** KuaiRand-1k and KuaiRand-27k were not attempted.

## Honest limitations

- **The margin is thin.** +0.0022 is about 3.0 standard errors once the
  baseline's own 0.0008 seed variance is propagated, not the 6.8 it appears at
  first glance. It is 20-seed confirmed and reproducible, but it is a small
  effect on a saturated benchmark, not a breakthrough.
- **It does not improve every run.** 8 of 11 clean-reference runs under V4 to V6.
  Run-to-run variance exceeds the difference between architecture generations, so
  no version is separable from another by result alone, and n is small.
- **The practical ceiling for this approach looks like ~0.604** against an oracle
  of 0.8484. Getting past it needs a different mechanism, not more tuning.
- **The test split was scored once, at the very end.** It was unreachable for
  the whole search, so no selection decision could be contaminated by it. Scored
  once after the configuration was frozen, the submission reaches **test primary
  0.5972** against the baseline's 0.5946: **score_dataset +0.0026**, with the
  validation gain of +0.0022 holding and growing slightly. That is a real
  generalisation check rather than a claim, but it is still a small effect on a
  saturated benchmark.

## Team

Solo entry. Kee Zhi Song built the whole submission: the frozen evaluation
harness around the organisers' Starter Kit, the agent loop and all eleven of its
stages, the guards, the human-tuned benchmark configuration the agent is measured
against, and every document here.

**Repository:** <https://github.com/KeeZhiSong/AutoMLEngineer>
