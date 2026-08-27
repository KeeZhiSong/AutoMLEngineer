# Libraries & Utilities

## Files
- **ledger.py** — Append-only experiment ledger + journal (INSIGHTS.md, DEAD_ENDS.md)
  - ExperimentLedger: Records every cycle (hypothesis, metrics, tokens, code diff, lessons)
  - Journal: Tracks insights and dead-ends for ideator context
  - Signal readers: best_metrics, tried_techniques, resource_totals
  
- **techniques.jsonl** — 40 curated recsys technique cards
  - Covers: ranking losses, multi-task, sequential, debiasing, feature interaction, training
  - Each card: name, problem, approach, when to use, key insight, paper reference
  - Used by ideator for RAG-grounded proposal

## Usage
```python
from ledger import ExperimentLedger, Journal
led = ExperimentLedger("workspace")
jour = Journal("workspace")
led.record(cycle=1, hypothesis=..., metrics=...)
jour.add_insight("What we learned")
```
