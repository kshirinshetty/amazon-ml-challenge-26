# 011 — more exact-key passes (+ synthetic decoys) — neutral

Run 010 + three more key passes in `src/block.py`: consonant skeleton of the sorted name words, spacing-free
name, first two name words; the address key loosened to "house number + first real word after it"
(`109/110 Durga Bari` → `109 durga`). Coverage estimate (`tools/pass_coverage2.py`): extra keys match up to
145,791 of the 257,484 still-missed train pairs.

| Metric | 010 | 011 |
|---|---|---|
| Blocking recall (after prune) | 0.9662 | 0.9675 |
| Own validation F0.5 (with synthetic decoys) | 0.9727 | 0.9729 |
| Test candidates / S1 | 11.9 | 13.3 |

Neutral. `tools/prune_loss.py` found why: recall is **0.9759 before pruning and 0.9676 after** — the
30-per-S1 cap, ordered by TF-IDF score, dropped 63,868 true pairs at hub S1s with hundreds of candidates
(median group 165), including records' *own best* candidate. Fixed for run 012 (the cap now trims only
runner-up candidates: recall 0.9703 at 13.3 candidates / S1). Not submitted.
