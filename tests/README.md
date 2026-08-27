# Contract tests

    python3 run.py test          # or: python3 tests/test_contract.py --data_dir <dir>

These guard the invariants that fail **silently** — where a broken run still
produces a plausible-looking number. They are not unit tests for correctness of
the model; they are tripwires for the ways this project can lie to itself.

| Test | What breaks if it fails |
|---|---|
| `convergence_rule` | The loop stops too early or never stops |
| `scoring_matches_official` | We score something other than GAUC/nDCG@5 |
| `label_is_long_view` | We optimise the wrong target entirely |
| `no_feedback_leakage` | A feedback column becomes an input; validation inflates |
| `revision_state_machine` | Revisions unbounded, or near misses discarded |
| `revision_cannot_switch_technique` | The ledger becomes unattributable |
| `seed_tiebreak` | Decisions get made on differences inside the noise |
| `alignment` | Submission `row_id`s misalign — every score is wrong |
| `train_only_statistics` | Feature stats leak from valid/test |
| `no_leaky_statistics_file` | Test-period aggregates contaminate training |

`--data_dir` is required for the last three; the rest run offline.
