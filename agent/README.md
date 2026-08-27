# Autonomous Agent Loop

## Files
- **controller.py** — Main orchestration loop (ideator → coder → executor → reflector)
- **ideator.py** — Proposes ideas grounded in technique library + ledger context
- **coder.py** — Generates code patches, validates syntax, rolls back on error
- **reflector.py** — Evaluates metrics, keeps/reverts, journals lessons

## Usage
```bash
python ../pipeline/pipeline.py  # Runs controller loop
```

## Key Components
- Zero manual interventions (fully autonomous)
- Error recovery (crashes logged as bugs, loop continues)
- Structured LLM outputs (Pydantic validation)
- Resource metering (tokens, GPU-hours tracked)
