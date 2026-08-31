# Agent's V6 + gpt-5.5 result — found unaided

Produced by the run in `workspace_v6c_gpt55/`, from a 0.6014 reference with no
task-specific answers in its library and zero manual interventions.

    technique : focal-loss-weighting
    module    : train.py
    valid     : 0.6030 (5-seed confirmed by the loop before acceptance)
    vs baseline 0.6016 : +0.0014
    converged after 9 cycles (13 logged experiments), 146K tokens, 0 crashes

The exploit stage then fired on the learning rate and measured a clean peak:

    lr 3e-05 -> 0.5986   1e-04 -> 0.6027
       3e-04 -> 0.6035   1e-03 -> 0.6027

`3e-4` scores 0.6035 over 5 seeds, +0.0005 on the incumbent -- INSIDE the 0.0008
accept margin, so the loop correctly declined it. It is not a result. Two
sub-margin leads of exactly this shape were chased earlier in the project
(`l2=1e-5`, `group_size=3`) and both evaporated at n=20. It would need a
20-seed paired test before it meant anything.

Reproduce:
    cp artifacts/agent-best-v6c/*.py solution/
