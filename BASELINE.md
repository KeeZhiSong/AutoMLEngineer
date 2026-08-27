# Official Baseline Reproduction — KuaiRand-Pure

Task Requirement #1 ("Reproduce the official baseline") — **complete**.

## Task definition (from the starter kit — authoritative)

| | |
|---|---|
| Task | Within-user ranking over logged impressions (no full-catalogue retrieval) |
| Relevance label | **`long_view`** (native column, 0/1) |
| Metrics | **GAUC**, **nDCG@5**; **primary = mean of the two** |
| Splits | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Zero-positive users | nDCG counted as 0.0 and included in the mean |
| GAUC population | only users with `0 < positives < impressions`, weighted by positive count |
| nDCG gain | `2^rel − 1` |
| Convergence | ε = 0.002, N = 3 consecutive validation iterations |

Note: the prose in problem statement §2.3 still reads "NDCG@10 / Recall@50, click =
positive". The starter kit supersedes it — §2.4 states the label and K values are
"pinned in the Starter Kit so every team solves the same task", and `evaluate.py`
implements `long_view` + GAUC/nDCG@5. Worth confirming at the 28 Aug webinar.

## Reproduction results

Run on this machine, single CPU core, numpy only.

```
python3 baseline.py --data_dir <KuaiRand-Pure/data> --model {random,pop,fm}
```

Split row counts matched the published spec exactly:
`train 1,141,112 / valid 124,909 / test 170,588`.

| Model | Split | GAUC | nDCG@5 | primary | Published primary | Δ |
|---|---|---|---|---|---|---|
| random (seed 0) | valid | 0.4990 | 0.4663 | 0.4827 | 0.4834 | −0.0007 |
| random (seed 0) | test | 0.4999 | 0.4514 | **0.4757** | 0.4753 | +0.0004 |
| item popularity | valid | 0.6387 | 0.5227 | **0.5807** | 0.5807 | 0.0000 |
| item popularity | test | 0.6308 | 0.5121 | **0.5715** | 0.5715 | 0.0000 |
| **FM (official)** | valid | 0.6671 | 0.5358 | **0.6015** | 0.6016 | −0.0001 |
| **FM (official)** | test | 0.6621 | 0.5286 | **0.5953** | 0.5946 | +0.0007 |

All deltas are within the published 5-seed std of 0.0008. `item_popularity` matches
to four decimal places. The random rung passes the starter kit's harness self-check
(`primary ≈ 0.475 ± 0.001`).

FM early-stopped at epoch 11 (patience 4, best at epoch 7), ~1.2 s/epoch, ~14 s total.

## Submission path verified

```
python3 submit.py --make  --split valid <out.csv>   ->  124,909 rows written
python3 submit.py --score --split valid <out.csv>   ->  alignment OK, primary 0.6015
```

Round-trips to the same score, so the CSV schema and row alignment are correct.

## The target

**Beat FM: test primary 0.5946.** Scoring is the equal-weighted mean of each
metric's absolute improvement over this baseline on the hidden test set.

Headroom is smaller than it looks. The oracle ceiling (true labels as scores) is
test primary **0.8645**, not 1.0 — 27.1% of test users are all-negative and score
nDCG = 0 for any model. FM has already captured ~30.7% of the achievable range;
remaining headroom is 0.27, not 0.41.

## Organizer-tested dead ends (do not re-explore)

| Tried | Result |
|---|---|
| Adding static features (all 13 CWM fields) | primary 0.5940 vs 0.5950 for 5 fields — no gain |
| Increasing capacity (k = 8 / 16 / 32) | 0.5895 / 0.5902 / 0.5887 — flat |

Structural reason: `user_id × video_id` crossing already absorbs most of the
learnable signal, and 1.14M rows will not support more capacity.

**Critical constraint:** pure user-side first-order terms contribute *exactly zero*.
Ranking happens within a user, so any term constant across that user's impressions
leaves the intra-group order unchanged. User-side features can only act through
crosses with item-side features.

## Organizer-flagged unexplored directions

Their stated priority order:

1. **Loss function** — currently pointwise logloss while the metrics are ranking
   metrics. Pairwise (BPR) or listwise softmax over the user's impressions aligns
   the objective with the scoring. Flagged as most likely to work.
2. **User behaviour sequences** — completely unused today. DIN / SIM-style interest
   modelling is a blank direction.
3. **Multi-task** — `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`,
   `play_time_ms` as auxiliary tasks for the `long_view` main task.
4. **Watch-time modelling** — CWM-style censored regression (watch time is truncated
   when a video completes, so a one-sided loss beats squared error).
5. **Different model** — DeepFM / DCN / xDeepFM. Lower priority since capacity is
   demonstrably not the bottleneck.
6. **Time features and distribution drift** — `hourmin`, `date`, train→test drift.
7. **Unbiased validation (advanced)** — `log_random_4_22_to_5_08_pure.csv` (1.18M
   rows of randomized exposure) as a debiased validation set.

## Vendored files

`official/` holds the starter kit unmodified. SHA-256 in `official/VENDORED.sha256`:

```
1bf54f5f3a9f590eab2f87f09a3c27422031867a20a5328d56cbd8c7db36e541  data.py
ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de  evaluate.py
ab01bb2b970ae2a9f2ead299f5240b71ff4126c2d9bb0e0c4de6c7e245dc148c  submit.py
c8f7fc60178413e247e78bb231e7550eeef52101b6493fcf1a4d2b0e5fe18f8a  baseline.py
```

`evaluate.py` is never modified — it is the sole authority on scoring.
