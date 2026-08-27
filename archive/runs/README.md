# Archived run logs
Each directory is one autonomous run: `experiments.jsonl` (per-cycle hypothesis, code diff, metrics, errors), `beliefs.jsonl`, `INSIGHTS.md`/`DEAD_ENDS.md`, `summary.json`.

## `01-library-pipeline-20cyc` (20 cycles)

Menu-driven ideator. 20 cycles. No improvement. The run whose failure motivated the CLASSIFY architecture.

## `02-clean-replay-library-25cyc` (25 cycles)

Answers stripped from the library, 25 cycles. Measured the list-length gap 3x at high confidence and never acted on it -- the observation->action failure.

## `03-classify-pipeline-25cyc` (25 cycles)

First CLASSIFY run. Named the right problem repeatedly; every implementation landed at or below the reference.

## `04-tier1-smoke-6cyc` (6 cycles)

First semantic-contract run. 6/6 blocked -- 3 leaks, 2 genuine no-ops, 1 impossible contract. Exposed the contract defects.

## `05-autonomy-benchmark-21cyc` (21 cycles)

Full V4 stack from the 0.6014 reference. FIRST autonomous improvement: 0.6022 via temporal decay, 0 interventions. Confirmed +0.0007 paired (t=2.28), later shown subsumed by the human grouping+lr config.
