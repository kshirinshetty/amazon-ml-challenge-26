# Starter prompt for the next session

Paste this to the coding agent:

---

We're competing in the Amazon ML Challenge 2026 (business entity resolution, team **wsg**). The repo is
`/home/sidd/Desktop/amazon-ml` (GitHub `kshirinshetty/amazon-ml-challenge-26`, branch `main`).

**Read these first, in order:** `docs/dev/HANDOFF.md` (everything: setup, pipeline, every experiment, lessons, ideas),
the root `README.md` (runs table), `runs/014_relative-tiebreak/README.md` (current best), and
`docs/problem_statement.md`.

**Current state:** best public leaderboard **0.970** (run 014); #1 is 0.9908. The final submission zip is built from
014 at `submission/wsg_submission.zip`. Our "dense" validation (`val_f05_dense` in `runs/*/metrics.json`) predicts
the leaderboard within ~0.003 — optimize it.

**Goal:** push the leaderboard score as high as possible toward 0.99 before the deadline (≈ 2026-09-27 23:00 IST —
confirm with me).

**Hard constraints:**
- Never run full-data stages on the laptop (15 GB RAM, it OOMs). All full-data work goes through
  `modal run modal_app.py ...` (see HANDOFF §3). Local = small samples only.
- Modal credit is nearly used up ($25.83 of $30 as of 17:00 IST) — check `modal billing summary` before every
  launch, estimate cost, and ask me before anything over ~$1.
- No external data / APIs / geocoding (disqualification). Final model must be MIT/Apache and ≤ 8B params.
- One `runs/NNN_name/` folder per experiment with a README (why / what / results); commit and push after each run;
  no Claude co-author trailer in commits. Never commit submission TSVs.
- Validate every file before I upload it (`docs/student_resource/validate_submission.py --check-ids`, no NUL bytes,
  1,732,545 lines). Rebuild the zip with `./package.sh runs/<best> wsg` whenever the best run changes.

**Suggested first steps:** (1) confirm the deadline and remaining budget with me; (2) review
`runs/014_relative-tiebreak/errors.tsv` and HANDOFF §8–9; (3) propose the 1–2 highest-value experiments that fit the
budget (cheap fit-only ones first: ensembling, prune-cap relaxation, targeted features), with expected dense-val gain;
(4) run them, compare on dense validation, and hand me the file to upload only if dense val beats 0.9717.

---
