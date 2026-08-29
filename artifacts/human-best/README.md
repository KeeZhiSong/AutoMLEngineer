# Human-tuned configuration — the submission

Reconstructed 28 Aug 2026. **The original code that produced `submission.csv`
was never saved** -- no listwise loss existed anywhere on disk, only the CSV and
the prose recipe. This is a rebuild from that recipe, verified by score.

    module    : train.py  (loss_fn = listwise_softmax, normalised targets)
    config    : batch_mode="user", group_size=5, lr=2e-4  (baked in as defaults)
    valid     : 0.6037 over 5 seeds (0.6036 0.6038 0.6038 0.6033 0.6039)
    published : 0.6038 -- reproduced to within 0.0001
    seed 0    : 0.6036

## The two steps, and why the second is easy to forget

1. **Match training lists to evaluation lists.** Training groups held ~31 items;
   evaluation lists hold ~4. A listwise objective trained on 31-item lists is
   optimising a different problem from the one being scored. `group_size=5`
   closes that gap: 0.6016 -> 0.6028, **+0.0012**.
2. **Retune the step size for the new objective.** Changing the grouping changed
   the gradient scale, which made the inherited `lr=1e-3` stale -- it had been
   tuned for a pointwise loss. `lr=2e-4`: **+0.0010** more, reaching 0.6038.

Step 2 is the one an agent skips, and it is why the V5 EXPLOIT stage exists.

## Known open question

`group_size=3` scores 0.6042 over the same 5 seeds, beating `group_size=5` on
4 of 5 paired seeds (mean +0.0005). That is BELOW the 0.0008 accept margin and
paired t is only ~2.1 on 4 df, so it is not established. Do not submit it
without more seeds.

Reproduce:
    cp artifacts/human-best/*.py solution/
    python3 tools_preflight.py --expect winning
