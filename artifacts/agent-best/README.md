# Agent's best configuration — found unaided

Produced by the run in `workspace_final/`, from a 0.6014 reference with no
task-specific answers in its technique library and zero manual interventions.

    valid     : 0.6038 over 20 seeds (sd 0.0003, min 0.6033, max 0.6043)
                GAUC 0.6702 (sd 0.0004) / nDCG@5 0.5373 (sd 0.0002)
                all 20 seeds above baseline; per-seed values in
                workspace_final/seed_sweep_20.json
    vs baseline 0.6016 : +0.0022
    seed 0    : 0.6040
    run       : converged in 5 cycles (13 logged experiments, 8 of them exploit
                trials), 105,304 tokens, 59 minutes, 0.81 GPU-hours

Two accepted interventions, in order:

    cycle 1   curriculum-learning    train.py     0.6023
    cycle 3   length-normalization   features.py  0.6040

**This matches the human-tuned result (0.6038) exactly**, by a different route.
The human path was a listwise objective on evaluation-length training lists plus
a retuned learning rate. The agent reached the same score through curriculum
learning over sequence length and a length-normalisation feature.

The single-seed value at acceptance was 0.6040, which cleared the tie-break band
by 0.0001 and so was accepted without confirmation. It was then measured over 20
seeds before being used for anything: mean 0.6038, every seed above the previous
agent best of 0.6031.

The previous best (0.6031, weighted-loss-imbalance) is kept in
`artifacts/agent-best-0.6031/`.

Reproduce:
    cp artifacts/agent-best/*.py solution/
    python3 tools_preflight.py --expect agent
