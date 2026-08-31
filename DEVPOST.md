## Inspiration

Plenty of agents write code. Far fewer do **research**, and the difference is
not code quality. It is knowing which of your own results to believe.

A research loop that cannot tell a real improvement from a lucky random seed
will confidently accumulate nonsense. So I set out to build an agent that runs
the whole machine-learning loop unattended on a real benchmark, and, more
importantly, one that is *hard to fool* — including by itself, and including by
me.

## What it does

The agent improves a recommender on **KuaiRand-Pure**. For each user it ranks
only the videos that user actually saw, scored by

$$\text{primary} = \tfrac{1}{2}\left(\text{GAUC} + \text{nDCG@5}\right)$$

The organisers' factorisation-machine baseline scores **0.6016**. That is a
harder bar than it looks:

| | primary |
|---|---|
| random scoring | 0.4834 |
| item popularity | 0.5807 |
| **official baseline** | **0.6016** |
| perfect ranking (oracle) | 0.8484 |

The ceiling is 0.8484 rather than 1.0 because **27.1% of evaluation users have
no positive label at all** — their nDCG is zero for any model. Real headroom is
0.247, and the organisers' own ablations move this metric by less than 0.002.
In this benchmark a \\(+0.001\\) gain is meaningful and a \\(+0.05\\) jump means
you have found a leak.

**Result: the agent reached 0.6031 unaided, from a 0.6014 reference, with zero
manual interventions** — confirmed over five seeds by the loop itself before
acceptance. A separate human-tuned configuration reached 0.6038. Those two
numbers came from different routes and I keep them clearly separated
throughout; the autonomous claim is 0.6031.

## How I built it

One pass through the loop is a **cycle**. There is always an incumbent, and each
cycle tests exactly one change against it.

```
OBSERVE -> CLASSIFY -> ANOMALY BOARD -> INVENT -> PLAN -> SPECIFY -> CODE
        -> VERIFY -> RUN -> JUDGE -> (on KEEP) EXPLOIT -> MEMORY -> next cycle
```

Each stage exists because of a failure I measured, not because it seemed like
good architecture:

- **OBSERVE** runs 3 of 9 measurement tools that return neutral numbers and no
  interpretation, because a tool that flags a finding has leaked the answer.
  Early cycles are scheduled: left to choose freely, the model picked a
  *measured dead end* in 100% of 14 logged cycles and the two tools that reveal
  the decisive fact in 14% and 0%.
- **CLASSIFY** turns measurements into a named problem. `train median 31,
  valid median 4` is a fact; *"train/eval mismatch, dimension = ranking-context
  length"* is something an intervention can attach to.
- **ANOMALY BOARD** keeps problems alive across cycles. Without it, one run
  diagnosed the same decisive problem in cycles 6, 11, 12 and 16 as four fresh
  discoveries and followed up on none.
- **INVENT** proposes interventions with *no technique library visible*, then
  may request literature for the problem it named. Pull, not push.
- **PLAN** enumerates four implementations and scores them. A plan that cannot
  name a measurable quantity it moves is disqualified, because no contract
  could be written for it.
- **SPECIFY** writes the semantic contract *before the code exists*. If the
  coder wrote first it would author a gate its own patch satisfies.
- **CODE** may rewrite only `features.py`, `model.py`, `train.py`. The loader,
  scorer and runner are frozen, so the agent can change the model but never
  what is measured.
- **VERIFY** takes 24 measurements in **0.4 seconds** with no training. It
  stopped **101 of 374** experiments before they cost a training run.
- **JUDGE** decides arithmetically. The model writes the explanation, never the
  verdict:

$$\text{keep} \iff \text{primary} > \text{incumbent} + 0.0008$$

  where \\(0.0008\\) is one published seed standard deviation. Anything within
  \\(0.0016\\) is automatically re-run on five seeds first.
- **EXPLOIT** fires only on a win, because a win is a new local research
  problem. Which parameter it retunes is derived from what measurably changed.

## Challenges

**The agent tried to cheat, twice.** One patch read a feedback column as an
input feature and scored **0.6449** — a spectacular result that was pure label
leakage, and it was accepted before the guard existed. Another padded the
scored set from 124,909 rows to 451,647, changing *what was measured* rather
than the model. The fixes are an AST scan of what code actually reads
(11 attempts blocked since) and a row-count invariant in the frozen layer.

**But I produced more false wins than the agent did.** The agent produced two
classes. I produced at least ten, nearly all of one species: measuring or
preserving state against the wrong reference. My favourite is that the code
which generated our own submission was never saved to disk — I found this,
rebuilt it, and verified the rebuild *by score*. It is easier to build a system
that catches an agent deceiving you than one that catches yourself.

**The most instructive moment was a win that wasn't.** An exploit trial scored
0.6026 on the first seed against a 0.6014 incumbent — a clean \\(+0.0012\\),
comfortably past the threshold. Because it landed inside the tie-break band the
loop re-ran it on five seeds:

```
0.6026  0.6019  0.6018  0.6014  0.6015   ->  mean 0.6018   (+0.0004)
```

Below the margin. Declined. The first measurement was not wrong, it just was
not representative. Later I chased a nearly identical sub-margin result myself
and lost an hour before it evaporated at twenty seeds. **The arithmetic gate
was more disciplined than the human operating it**, which is the whole argument
for not letting a language model, or a person, judge their own experiment.

**Two architecture generations failed and were reverted.** A codebase
capability map fixed exactly the failure it targeted — the direct intervention
went from ranked last to ranked first among candidates — and produced no win
across three runs. It is preserved on a branch and reported rather than
quietly dropped.

## Built with

**Development tools:** VS Code, git, GitHub Actions (regenerates the published
run-ledger page from the logs), macOS terminal. No notebooks: every experiment
runs through a script so it is reproducible and logged.

**APIs:** OpenAI Chat Completions. Five agent roles, each on the model it needs,
via a provider-agnostic routing layer:

| role | model | job |
|---|---|---|
| analyst | `gpt-4o-mini` | pick measurement tools, report numbers |
| classifier | `gpt-4o` | turn measurements into a named problem |
| inventor | `gpt-4o` | propose interventions |
| **coder** | **`gpt-5.5`** | write correct numpy (the measured bottleneck) |
| reflector | `gpt-4o-mini` | explain a decision already made arithmetically |

OpenAI was the subscription I had with the best price/quality balance available
to me. No cross-provider comparison was run, so nothing here is a claim about
relative model quality. The routing layer supports any provider.

**Libraries and frameworks:** `numpy` and `pandas` only, plus the `openai` SDK
and the Python standard library (`ast` for the leak scanner, `signal` for the
per-cycle timeout, `json`/`dataclasses` for the ledger). **No deep-learning
framework.** The model is a factorisation machine written in numpy, which is why
20 runs cost 4.09 GPU-hours on a laptop CPU with no accelerator.

**Datasets:** KuaiRand-Pure (Gao et al., CIKM 2022), the required benchmark. The
`log_standard` files only. Deliberately unused: `video_features_statistic_pure.csv`
(undated counters spanning the test window, a leakage hazard) and
`log_random_*.csv` (legal as an unbiased cross-check, never trained on). No
external or manually labelled data.

## What I learned

**Diagnosis is not the bottleneck.** In one 25-cycle run the agent named the
decisive problem four times and converted it into a working patch zero times.
The gap is selection and implementation.

**Model choice matters exactly where the bottleneck is.** Same architecture,
same budget, only the coder changed: one model produced 3 crashes, 0 keeps and
used 313K tokens; a stronger one produced 0 crashes, a keep, and used 146K.
*Cheaper overall*, because code that works reaches a verdict sooner.

**Negative results are the real output.** Seven directions are closed by
measurement, each with a mechanism. My favourite: auxiliary click targets *hurt*
by \\(-0.0020\\), because \\(P(\text{long\_view}=1 \mid \text{click}=0) = 0.003\\)
means clicked-but-not-long-viewed rows are the confusable class the metric
exists to separate, not weak positives.

**Cost:** 20 runs, 374 experiments, 4.12M tokens, **4.09 GPU-hours on a laptop
CPU**, and **zero manual interventions** in every run.
