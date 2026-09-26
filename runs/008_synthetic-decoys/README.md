# 008 — synthetic decoys at test density

Run 006's pipeline + `src/synth.py`: 2,636,097 synthetic decoys added to the training split (1.19 per S1),
bringing train to the test's ~2.4 decoys per S1 (`tools/shift.py`). Made only from the provided training
data, following the decoy recipe observed in train:

- half: a real decoy re-shifted to a new house number (±1–30);
- half: a true record turned into a decoy — house number shifted (80%) and/or a marker word added from
  words ≥90% decoy in train (US `eastgate lakeside northside westgate greater southside riverside midtown`,
  India `hardware medicals overseas infratech bakery pharmacy sweets restaurant provision`).

Synthetic decoys match no S1 (never in `train_gt`), so they only act as negatives — and they make every S1's
group context (records pointing at it, sibling agreement, label-free decoy scores) look like test.

## Results

| Validation | run 006 model | run 008 model |
|---|---|---|
| with synthetic decoys (circular: 008 learned exactly these) | 0.9371 | 0.9697 |
| **real records only, group context at test-like density** (`tools/compare_real.py`) | 0.9687 | **0.9730** |
| same, dense | 0.9552 | **0.9612** |

With test-like decoy density around each S1 the 006 model degrades (0.9747 → 0.9687) and 008 holds up
(0.9730). Blocking 0.9574 recall, 1237 trees. **Not submitted**; combined with run 009's blocking in run 010.
