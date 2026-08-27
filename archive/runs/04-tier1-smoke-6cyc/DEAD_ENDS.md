# Dead Ends

- [2026-08-27 13:35] [cycle 1] IMPLEMENTATION failure (not a scientific one): skewed-feedback-loss — pct_rows_contributing_gradient > 17.9 -> got 17.9
- [2026-08-27 13:37] [cycle 2] IMPLEMENTATION failure (not a scientific one): embedding-initialization-nearest-neighbors — unique_users_per_batch > 6249 -> got 6249
- [2026-08-27 13:38] Coder crash on 'Add a feature for each user's historical average long_view r': Label leakage in features.py:
  - line 105: reads 'long_view', a feedback/label column -- it is an outcome of the same impression being ranked and can
- [2026-08-27 13:39] Coder crash on 'Introduce augmentation for user feedback by generating synth': Label leakage in features.py:
  - line 82: calls .feedback() -- that API supplies auxiliary TARGETS for train.py, never model inputs

Feedback columns
- [2026-08-27 13:40] [cycle 5] IMPLEMENTATION failure (not a scientific one): dynamic-loss-weighting — train_positive_rate approx 0.3134 -> got 0.33662
- [2026-08-27 13:41] Coder crash on 'Introduce a new feature capturing recent long views per item': Label leakage in features.py:
  - line 87: reads 'long_view', a feedback/label column -- it is an outcome of the same impression being ranked and cann
