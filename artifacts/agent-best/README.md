# Agent's best configuration — found unaided

Produced by the clean run in `workspace/` (cycle 11), from a 0.6014 reference
with no task-specific answers in its technique library.

    technique : weighted-loss-imbalance
    module    : train.py  (loss_fn = weighted_logloss)
    valid     : 0.6031 over 5 seeds (0.6024 0.6033 0.6037 0.6030 0.6029)
    vs baseline 0.6016 : +0.0015
    seed 0    : 0.6024

Confirmed at 5 seeds BY THE LOOP ITSELF before it was accepted -- the tie-break
fires automatically when a result lands within 2 sigma of the incumbent, so this
is not a single-seed artefact.

Zero manual interventions. The contract for this cycle targeted `initial_loss`,
which is the exact metric that falsely blocked a loss change two runs earlier --
the `changed`-over-direction fix is what let it through.

Reproduce:
    cp artifacts/agent-best/*.py solution/
    python3 tools_preflight.py --expect agent
