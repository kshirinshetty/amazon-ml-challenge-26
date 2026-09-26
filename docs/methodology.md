# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** wsg  
**Team Members:** Siddartha Aralakuppe Yogesha, Kshirin Shetty  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A blocking + gradient-boosting pipeline built around two findings from the data. (1) Every Source 2/3
record matches at most one Source 1 entity, so candidate generation runs *from* each S2/S3 record into a
per-country Source 1 index (sparse TF-IDF top-10 plus exact-key passes), and each record is finally assigned
to at most one entity. (2) About a quarter of Source 2/3 records (≈42% on test) are **decoys** — near-copies
of a real business with an extra marker word or a shifted house number — so most features are designed to
expose exactly that, and they are computed **without labels**, which makes them work unchanged on France
(absent from training). A LightGBM classifier over 49 pair features, with its threshold chosen on a
validation set that mimics the test's decoy density, reaches validation F0.5 = **0.978** (0.968 on the
test-density validation) with 96.7% blocking recall.

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the full training set (2.21M S1, 5.03M S2, 5.29M S3 records; 7.64M true pairs):

| Finding | Number | Consequence |
|---|---|---|
| Matches per S1 | 0 → 5.6%, 1 → 5.4%, 2–6 → ~85%, max 11 | predicting nothing is right for ~6% of entities |
| S1s matched by one S2/S3 record | at most 1, always | retrieve from the S2/S3 side; assign each record to one S1 |
| S2/S3 records matching no S1 (decoys) | 26% train; **≈42% test** (label-free estimate) | the model must reject near-copies; validate at test density |
| Matches across countries | never | block within country; `country` stays an open label |
| Indic-script names / addresses | 13% / 14% of India S2 | transliterate before comparing |
| Test | 1.73M S1: India 47%, US 38%, **France 15%** | no country-specific features |

**Decoys.** Decoys copy a real S1 and change it slightly: a marker word (US `midtown`, `eastgate`,
`holdings` — ~100% of records containing them are decoys; India `overseas`, `infratech`; France `holding`,
`participations`), a different legal form (US records with `ltd`: 72% decoys) and/or a shifted house number
(true records keep the S1's first house number 84% of the time, decoys 9%).

**Test density.** Test has 5.75 S2/S3 records per S1 vs 4.67 in train; the house-number mismatch rate of
records vs their best S1 implies ≈42% decoys (train 26%, estimate exact on train). A validation set with half
the true records removed reproduces the leaderboard (0.9638 vs 0.961) and is used for model selection.

**Noise.** Names: legal-suffix churn, prefixes/suffixes (`Sri`, `Center`), typos, reordering, accents,
glued phone numbers, domain/handle forms (`internationalforteanimation.com`), `X formerly/dba Y` aliases,
Indic scripts. Addresses: abbreviations, state names vs codes, zero-padded or ranged house numbers,
placeholders (`None`, `N/A`, `<NULL>`), reordering, dropped components.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (multi-pass reverse retrieval → LightGBM → one-S1-per-record assignment)  
**Core Innovation:** label-free decoy detection and test-density-aware validation; reverse-direction,
multi-pass blocking.

**Normalization** (`normalize.py`): alias stripping (`X formerly Y` → `Y`), `unidecode` transliteration,
removal of domains/phones/punctuation, repeated-letter collapse; **spelling maps learned from matched training
pairs** (e.g. `praivet → private`, `eksports → exports`, `sacrmento → sacramento`; only S2/S3-side spellings,
lossy maps skipped); **filler words detected per split and country without labels** (tokens over-represented
in S2/S3 vs S1); address abbreviations and state names → codes. Filler/legal words are removed for blocking
but kept in a "full" name for the decoy features.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. **Reverse TF-IDF retrieval** per country: each S2/S3 record retrieves its top-10 S1s by
     `0.6·cos(name char 4-grams) + 0.4·cos(address word 1–2-grams)`; n-grams in >0.5% of S1s are dropped
     for speed (exact sparse top-k with `sparse_dot_topn`).
  2. **Exact-key passes** for what TF-IDF misses when every n-gram of a name is common: normalized name
     words sorted; first house number + following street word. Keys shared by >50 S1s are skipped; per
     record and key the 2 S1s closest on the *other* field are kept.
  3. **Pruning**: a record keeps rank-2..10 S1s only if their score is ≥0.8× its best; key-pass pairs are
     kept; at most 30 candidates per S1.
- **Candidate pairs generated:** 20.5M on test — **11.9 per S1** (reduction ratio >99.999% vs all pairs).
- **How we ensured true matches were not lost:** recall was measured on training labels for every change.
  v0 (char 3-grams, heavier pruning) 85.1% → char 4-grams + address bigrams 95.3% → top-10 95.8% →
  **exact-key passes 96.7%**. A loss decomposition showed never-retrieved true pairs as the largest
  remaining loss, which is what motivated the key passes (they recover 43% of the TF-IDF misses).

---

## 4. Matching Model

**Features used** (49, all label-free):
- Name: rapidfuzz `ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio`, Jaro-Winkler; TF-IDF
  cosine; extra words on each side of the *full* name; label-free **decoy score** of the extra words
  (per country: share of records whose best S1 lacks the word *and* whose house number disagrees; 0.82–0.88
  correlation with the true decoy share); ambiguity (S1s sharing the name; share of the record's words known
  in the country's S1 vocabulary — detects made-up trade names); transliteration flag.
- Address: rapidfuzz `ratio`, `token_set_ratio`, `partial_ratio`; TF-IDF cosine; house-number overlap,
  first-number equality/distance, whether the S1's first house number appears in the record; S1s sharing the
  exact address.
- Other: retrieval score/rank, record's best and second-best score, gap and margin; key-pass flags;
  records pointing at the S1 and this record's rank among them; **sibling agreement** (other records of the
  S1 sharing this record's house number; this record's words no other sibling has); source id.

**Model type:** LightGBM binary classifier (`learning_rate 0.05, num_leaves 511, min_data_in_leaf 50,
feature_fraction 0.8, bagging 0.8`), early stopping, trained on candidate pairs of 80% of training S1s.  
**Threshold selection method:** each S2/S3 record is assigned to its highest-probability S1 candidate and
kept if p ≥ t; t maximizes macro F0.5 (the exact leaderboard metric, singletons included) on a held-out 20%
of training S1s at test decoy density.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.978** on held-out training S1s; **0.968** at test decoy density
  (the leaderboard proxy). Public leaderboard history: 0.933 → 0.945 → 0.957 → 0.961.
- **Where the remaining loss is** (test-density validation, fixing one error type perfectly):
  never-retrieved true pairs +2.1 pts (before the key passes), retrieved-but-rejected true pairs +0.9
  (mostly records with no address), false merges +0.6 (almost all decoys).
- **Common false positives (wrong merges):** decoys whose only differences are a small house-number shift or
  a marker word; records with no address whose name is shared by several S1s; made-up trade names at an S1's
  exact address.
- **Common false negatives (missed matches):** records with no address and a common name; names that lost
  their distinctive word; Indic-script names whose transliteration is far from the English spelling.

---

## 6. Conclusion

Reverse, multi-pass blocking keeps the candidate set small (~12 per S1) while retaining 96.7% of true pairs,
and label-free features that describe how decoys are made let a single LightGBM model separate near-copies
from true variants in any country. The most useful lessons: validate at the test's decoy density, and
measure where the loss is before optimizing — blocking recall mattered more than model complexity.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — see its `README.md` for exact commands.

| File | Entry point | Output |
|---|---|---|
| `src/normalize.py` | `python src/normalize.py` | `work/{train,test}.parquet`, `work/train_gt.parquet`, learned maps |
| `src/block.py` | `python src/block.py train\|test` then `--prune` | `work/{split}_cands.parquet` (= candidate_pairs) |
| `src/features.py` | `python src/features.py train\|test` | `work/{split}_feats.parquet` |
| `src/match.py` | `python src/match.py fit DIR` then `predict DIR` | `DIR/model.txt`, `DIR/metrics.json`, `DIR/output/*.tsv` |

Dependencies are pinned in `requirements.txt`. No external data, APIs or pretrained models; LightGBM is
MIT-licensed and trained from scratch on the provided data.

### B. Additional Results

| Run | Change | Blocking recall | Candidates / S1 | Val F0.5 (dense) | Leaderboard |
|---|---|---|---|---|---|
| 001 | TF-IDF reverse blocking + LightGBM (29 features) | 0.953 | 17.3 | 0.949 | 0.933 |
| 002 | learned cleaning, pruning, per-S1 decision rule | 0.952 | 7.8 | 0.957 | 0.945 |
| 003 | label-free decoy features | 0.952 | 7.8 | 0.971 | 0.957 |
| 006 | ambiguity features, larger LightGBM, top-10 retrieval | 0.958 | 9.4 | 0.975 (0.964) | 0.961 |
| 009 | exact-key blocking passes | 0.967 | 11.9 | 0.978 (0.968) | — |

Tried and rejected: France self-training on confident test pairs (no change), stage-2 stacking on
out-of-fold probabilities (+0.4 on validation, −0.1 on the leaderboard), stricter thresholds (0.959/0.956).
