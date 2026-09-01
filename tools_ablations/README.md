# Ablations kept for the record

Experiments run by hand, not by the agent, to size decisions the write-up makes
claims about. Kept so the numbers in `RESULTS.md` are reproducible rather than
asserted.

| file | what it measures |
|---|---|
| `features_with_statistic.py` | a `solution/features.py` variant that adds `video_features_statistic_pure.csv` as two decile-bucketed fields, edges fitted on train only |
| `measure_declined_statistic.py` | runs that variant against the reference through the frozen runner, 3 seeds each, and prints the delta |

`measure_declined_statistic.py` writes the variant to `solution/features.py`
because `runner.run` reloads that module from disk on every call. It restores
the committed file in a `finally` block, including on interrupt.

Result, recorded in `RESULTS.md`: reference 0.6014, with the statistic 0.6019,
**delta +0.0005** — below one published seed std, so it would not have cleared
the loop's own keep threshold.

> A first version of this measurement monkey-patched `features.transform` in
> memory instead of writing to disk. `runner.run` reloaded the module and wiped
> the patch, so both arms scored identically to four decimals and the honest
> reading was "the statistic adds nothing". It was a no-op wearing the costume
> of a result — the exact failure the agent's semantic contract exists to catch,
> reproduced by the human operating it.
