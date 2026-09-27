# 013 — run 012 without synthetic decoys (hedge)

Identical to [012](../012_keys-synth-safecap/) (TF-IDF top-10 + 5 exact-key passes, cap that never drops a
record's best candidate, 49 features, LightGBM) but trained **without** synthetic decoys
(`modal_app.py --no-synth`). Built so the leaderboard can decide whether synthetic decoys transfer to the
real test: 012 vs 013 differ only in that.

| Metric | 009 (no synth) | **013 (no synth)** | 012 (synth) |
|---|---|---|---|
| Blocking recall | 0.9673 | **0.9706** | 0.9702 |
| Oracle F0.5 | 0.9887 | **0.9899** | 0.9897 |
| Validation F0.5, own context | 0.9781 | **0.9794** | 0.9741 (incl. synthetic decoys) |
| Dense validation, own context | 0.9684 | **0.9701** | 0.9614 (incl. synthetic decoys) |
| Real records, test-density context | 0.9730 | — | 0.9774 |
| Test candidates / S1 | 11.9 | 13.5 | 13.5 |
| Leaderboard | _pending_ | _pending_ | _pending_ |

436 trees, threshold 0.70. Test matches / S1: France 3.18, India 3.26, US 3.32.
If 013 beats 012 on the leaderboard, rebuild the zip from it: `./package.sh runs/013_keys-safecap-nosynth wsg`
(and drop the synthetic-decoy paragraph from `docs/methodology.md` / the synth step from `docs/package_README.md`).
