# blend 009 + 010 (leaderboard candidate, not a pipeline run)

`tools/blend_test.py`: 0.3 × run 009 + 0.7 × run 010 stage-1 probabilities on their shared test features,
threshold 0.60. On real validation records in a test-density context: 0.9764 (dense 0.9658) vs 010 alone
0.9759 (0.9650) — superseded by run 012 (0.9774 / 0.9670). Output in `output/` (not in git).
