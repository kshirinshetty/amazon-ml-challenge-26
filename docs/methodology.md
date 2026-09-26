# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** wsg  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

A two-stage blocking + classifier pipeline. **Blocking** inverts the problem: because every Source 2/3
record belongs to at most one Source 1 entity, each S2/S3 record retrieves only its top-3 most similar
S1 records (same country) with sparse TF-IDF cosine (name character 4-grams + address word
uni/bigrams). This keeps ~17 candidates per S1 entity while retaining 95.3% of true pairs.
A **LightGBM** classifier scores each pair on 29 string-similarity and ranking-context features, and each
S2/S3 record is assigned to its single best S1 only if its probability clears a threshold tuned directly
for macro F0.5 on held-out S1 entities (validation F0.5 = **0.949**).

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the full training set (2.21M S1, 5.03M S2, 5.29M S3 records; 7.64M true pairs):

| Finding | Number | Consequence for the design |
|---|---|---|
| Matches per S1 entity | 0 → 5.6% (singletons), 1 → 5.4%, 2–6 → ~85%, max 11 | predicting nothing is right only for ~6% of entities |
| S1 entities matched by one S2/S3 record | **at most 1**, always | 1-to-many structure: retrieve from the S2/S3 side, assign each record to its best S1 |
| S2/S3 records matching no S1 | ~26% (2.7M distractors) | the model must be able to reject a record's best candidate |
| Matches across countries | never | block within country (country kept as an open label; France handled like any other) |
| Indic-script names (S2/S3, India) | 13% / 7.5% of names, ~14% of addresses (Devanagari, Gujarati, Tamil, Bengali, Kannada…) | transliterate everything to ASCII before comparing |
| Missing addresses | ~3% of S2/S3 (`None`, `N/A`, empty) | name-only matching path; weak spot (see §5) |
| Test set | 1.73M S1: India 47%, US 38%, **France 15%** (unseen in training) | no country-specific features or one-hots |

Noise observed in names: legal-suffix churn (`LLC`/`L.L.C.`/`Corp`/`Pvt Ltd`/`(Limited)`), injected
prefixes/suffixes (`Sri`, `Dr`, `--`, `Center`, `Services`, `Co`), duplicated words (`Moore Moore`),
word reordering, character typos (`Privase`, `Irhnia`), accents (`FÁCT`), phone numbers glued on
(`- 9832661323`), web-domain / handle forms (`internationalforteanimation.com`, `@internationalforte`),
`X DBA Y` trade names, and names written in Indic scripts. Noise in addresses: abbreviations
(`St`/`Street`, `Dr`/`Drive`), state names vs codes (`Arizona`/`AZ`, `Karnataka`/`KA`, even
`Telangana` vs `Andhra Pradesh`), zero-padded and ranged house numbers (`06240`, `4631-4635`),
`##`/`N/A`/`<NULL>` placeholders, component reordering, split/merged words (`DOWNER SGROVE`), dropped
components (no street number, no city).

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (sparse retrieval → gradient-boosted pair classifier → constrained assignment)  
**Core Innovation:** *Reverse-direction blocking with a one-S1-per-record assignment.* Retrieval runs from
each S2/S3 record into the S1 index, so the candidate set of an S1 entity is exactly the set of records
that consider it one of their top-3 matches. This (a) bounds the total candidate count at 3 × #S2/S3
records, (b) mirrors the data's 1-to-many structure, and (c) lets the final assignment resolve
conflicts by giving each record to at most one S1 entity.

Pipeline: `normalize.py → block.py → features.py → match.py fit → match.py predict`.

**Normalization** (`normalize.py`, applied identically to all sources and countries):
- `unidecode` transliteration of every non-ASCII string (Indic scripts, French accents), lowercase.
- Names: remove web-domain suffixes, `www.`, `@`, digit runs ≥ 6 (phones), `&` → `and`, punctuation;
  drop legal-form/filler tokens (`llc inc corp co company ltd limited pvt private plc lp llp pc pllc sas
  sasu sarl sa eurl sci snc the and of dba`); collapse repeated letters (`raam maarkettiNg` → `ram
  marketing`), checking the stop list both before and after collapsing.
- Addresses: placeholders (`None`, `N/A`) → empty; punctuation → space; strip leading zeros;
  canonical short forms for street types / directions (`street→st`, `road→rd`, `r→rue`…) and US/Indian
  state names → codes (multi-word states such as `tamil nadu → tn` handled first).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** within-country sparse TF-IDF retrieval, fitted on that country's S1 records:
  - name: TF-IDF over **character 4-grams** (`char_wb`) of the normalized name with spaces removed
    (robust to typos, word reordering and domain-style names like `internationalforteanimation`);
  - address: TF-IDF over **word unigrams + bigrams** of the normalized address;
  - both L2-normalized and combined as `score = 0.6·cos_name + 0.4·cos_address`;
  - n-grams present in more than 0.5% of a country's S1 records are dropped (`max_df=0.005`); this is
    what makes an exact sparse top-K search over millions of records tractable (`sparse_dot_topn`,
    multi-threaded), and removes the least discriminative terms;
  - every S2/S3 record keeps its **top-3** S1 records.
- **Candidate pairs generated:** 29.9M on test (1.73M S1 × 9.97M S2/S3 records) — **17.3 candidates per
  S1 entity**, a reduction ratio of 99.9998% versus all S1 × S2/S3 pairs; 30.96M on train.
- **How you ensured true matches were not lost:** recall was measured on the training labels for every
  design choice (sample of 30k matched records per country, then the full set):

  | Blocking variant | Recall@3 (US) | Recall@3 (India) |
  |---|---|---|
  | name char 3-grams + address words, `max_df=0.002` (v0) | 0.879 | 0.808 |
  | name char 4-grams + address words, `max_df=0.002` | 0.951 | — |
  | name char 4-grams + address uni+bigrams, `max_df=0.002` | 0.952 | 0.915 |
  | **name char 4-grams + address uni+bigrams, `max_df=0.005` (used)** | **0.967** | **0.931** |
  | same, `max_df=0.01` (≈2× slower) | — | 0.938 |

  Full training set with the chosen setting: recall@1 = 0.932 (4.7 candidates / S1),
  recall@2 = 0.946 (9.4), **recall@3 = 0.953 (14.0)**. The v0 failure mode is instructive: with
  character 3-grams nearly every common gram exceeds the document-frequency cut, leaving only rare,
  accidental grams (e.g. `Dr TITAN IMPORT` retrieved `Indrtech Infra` through the gram "drt");
  4-grams and word bigrams are rare by construction and survive the cut.

---

## 4. Matching Model

**Features used** (29, all computed per candidate pair; no country or source-specific encodings except
the S2/S3 source id):
- Name features: rapidfuzz `ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio` on normalized
  names; Jaro-Winkler on the space-free name; TF-IDF name cosine; name lengths; transliteration flag
  (record's raw name was non-ASCII).
- Address features: rapidfuzz `ratio`, `token_set_ratio`, `partial_ratio`; TF-IDF address cosine;
  house/unit-number overlap (intersection, union, Jaccard of digit tokens; first-number equality);
  address lengths.
- Other (retrieval context): combined retrieval score, rank of this S1 in the record's top-3, the
  record's best and second-best scores, gap to its best score and best-vs-second margin; how many
  records retrieved this S1 (total and as rank-1) and this record's rank among them.

**Model type:** LightGBM binary classifier (`num_leaves=255`, `learning_rate=0.1`,
`min_data_in_leaf=100`, `feature_fraction=0.9`, `bagging_fraction=0.8`), trained on candidate pairs of
40% of training S1 entities (12.4M pairs, label = pair in ground truth), early-stopped on held-out pairs
(219 trees, ~1 min on 32 cores).  
**Threshold selection method:** each S2/S3 record is assigned to its highest-probability S1 candidate
(a record matches at most one S1); the assignment is kept only if the probability ≥ *t*. *t* is chosen by
sweeping 0.10–0.95 and maximizing the exact leaderboard metric (macro F0.5 per S1 entity, singletons
included) on a disjoint 20% of training S1 entities → *t* = 0.60.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.949** on held-out training S1 entities (~441k entities, split by hash of
  the S1 id). Ceiling with a perfect matcher on the same candidates: 0.984.

  | threshold | 0.30 | 0.40 | 0.50 | **0.60** | 0.70 | 0.80 | 0.90 |
  |---|---|---|---|---|---|---|---|
  | F0.5 | 0.935 | 0.943 | 0.948 | **0.949** | 0.948 | 0.946 | 0.938 |

  Test-set sanity check (no labels): France, never seen in training, gets 3.37 matches per S1 and 5.8%
  empty predictions vs 3.24 / 6.5% for India and 3.28 / 6.2% for the US — consistent behaviour across
  countries.
- **Common false positives (wrong merges):** near-duplicate businesses at the same or adjacent address
  where only a small detail differs — a changed house/unit number (`9101` vs `9104 Crowne Springs Cir`),
  an extra name token (`Parameswara Technology` vs `Parameswara Technology Overseas`), or a different
  first word (`Brown & Cross Origin` vs `Jacquez & Cross Origin`).
- **Common false negatives (missed matches):** in a 300-sample of missed pairs, 46% never reached the
  model (not among the record's top-3 — dominated by records with an empty address and a noisy name
  such as `Lucknow Corp Center`, or with an Indic-script name), 42% scored below the threshold (typos or
  changed house numbers on otherwise identical records), and 12% were assigned to a different S1
  (mostly Indic-script names whose transliteration is far from the English spelling, e.g.
  `बिग इन्वेस्टमेंट` vs `Big Investment`).

---

## 6. Conclusion

Reverse-direction sparse retrieval plus a one-S1-per-record assignment gives small candidate sets
(17 per S1) with 95% recall, and a gradient-boosted classifier over classic string-similarity features
with a threshold tuned for the exact metric reaches 0.949 macro F0.5 on held-out entities. The largest
remaining gains are in blocking recall for address-less and Indic-script records and in separating
near-duplicate businesses that share an address.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — see its `README.md` for exact commands.

| File | Entry point | Output |
|---|---|---|
| `src/normalize.py` | `python src/normalize.py` | `work/{train,test}.parquet`, `work/train_gt.parquet` |
| `src/block.py` | `python src/block.py train\|test` | `work/{split}_cands.parquet` (+ recall@K on train) |
| `src/features.py` | `python src/features.py train\|test` | `work/{split}_feats.parquet` |
| `src/match.py` | `python src/match.py fit DIR` then `predict DIR` | `DIR/model.txt`, `DIR/metrics.json`, `DIR/output/{matching_results,candidate_pairs}.tsv` |

Dependencies are pinned in `requirements.txt` (polars, scikit-learn, sparse-dot-topn, rapidfuzz,
unidecode, lightgbm). No external data, APIs or pretrained models are used; LightGBM is MIT-licensed.

### B. Additional Results

Top features by LightGBM gain (run 001): gap to the record's best retrieval score 71.5%, address
`token_set_ratio` 5.4%, name Jaro-Winkler 3.5%, retrieval rank 2.9%, record's 2nd-best score 2.1%,
address length 1.7%, address `partial_ratio` 1.5%, name `ratio` 1.5%, #records retrieving the S1 1.4%,
name TF-IDF cosine 1.3%.

Experiment history (each run's code snapshot, metrics and error samples are versioned in our repository):

| Run | Change | Blocking recall | Candidates / S1 | Val F0.5 |
|---|---|---|---|---|
| 000 | char 3-gram name / address word blocking, `max_df=0.002` | 0.851 | 13.9 (train) | — |
| 001 | char 4-gram name / address uni+bigram blocking, `max_df=0.005`; LightGBM on 29 features | 0.953 | 17.3 (test) | 0.949 |
