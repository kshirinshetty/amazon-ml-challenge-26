# 005 — deeper retrieval (top-10) (neutral result)

Same as 003 except blocking: retrieval K = 3 → **10**, prune gains a rank cutoff (`RMAX`) and
REL 0.7 → **0.8** (cap 30). Why: 4.50% of true pairs were ranked 4+ with K=3 (`tools/misses.py`:
30% of misses have an empty address; India misses 6.7% vs US 3.4%).

Prune grid (train recall / candidates per S1), K=10:

| RMAX | REL 0.7 cap 30 | REL 0.8 cap 30 | REL 0.9 cap 30 |
|---|---|---|---|
| 3 | 0.9524 / 6.46 | 0.9517 / 5.77 | 0.9494 / 5.39 |
| 5 | 0.9565 / 7.55 | 0.9556 / 6.47 | 0.9523 / 5.82 |
| **10** | 0.9592 / 8.90 | **0.9582 / 7.31** | 0.9539 / 6.31 |

| Metric | 003 | 005 |
|---|---|---|
| Blocking recall (validation) | 0.9524 | **0.9581** |
| Oracle F0.5 | 0.9834 | **0.9852** |
| **Validation F0.5** | 0.9712 | 0.9711 |
| Test candidates / S1 | 7.8 | 9.4 (France 11.4) |

The extra true pairs are the hard ones (no address, lost name words) and the matcher could not accept them
confidently, so F0.5 did not move while the candidate set grew. **Not submitted.** Kept as the blocking
for run 006, whose new ambiguity features target exactly those records.
