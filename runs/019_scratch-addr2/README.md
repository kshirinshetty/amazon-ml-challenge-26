# 019 — full pipeline from scratch on a new Modal workspace + second address match

018's pipeline (address pass, noise features, 1023-leaf LightGBM) rerun end to end (download → predict, `--no-synth`)
on a fresh Modal workspace (the first one's $30 credit was used up), with the address pass keeping a **second**
address match when it scores ≥ 90% of the record's best (`ADDR_K, ADDR_REL = 2, 0.9`; local India sample +0.1 pt).

| Metric | 018 | **019** |
|---|---|---|
| Blocking recall (train) | 0.9785 | **0.9792** (grid) / 0.9790 (fit) |
| Oracle F0.5 | 0.9928 | 0.9930 |
| Validation F0.5 | 0.9843 | 0.9845 |
| **Dense validation** | 0.9766 | **0.9769** |
| Candidates / S1 (train / test) | 11.63 / 14.55 | 12.31 / 15.29 |

605 trees, threshold 0.70. Not submitted on its own; its features and model are the base of 020–022.
