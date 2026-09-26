#!/usr/bin/env sh
# Build the final submission zip from a run folder (see docs/problem_statement.md, "Final Submission Package").
#   ./package.sh runs/001_tfidf-lgbm-baseline <team_name> [output_dir]  ->  submission/<team_name>_submission.zip
set -eu
RUN=${1:?usage: ./package.sh RUN_DIR TEAM_NAME}; TEAM=${2:?usage: ./package.sh RUN_DIR TEAM_NAME}
OUT=${3:-output}; PKG=submission/${TEAM}_submission; CODE=$PKG/code/business_entity_resolution
rm -rf "$PKG" "$PKG.zip" && mkdir -p "$PKG/output" "$CODE"
cp "$RUN/$OUT/matching_results.tsv" "$RUN/$OUT/candidate_pairs.tsv" "$PKG/output/"
cp -r "$RUN/src" "$CODE/src"
cp docs/package_README.md "$CODE/README.md"
uv export --frozen --no-hashes --no-emit-project -q > "$CODE/requirements.txt"
cp docs/methodology.md "$PKG/Documentation_template.md"
python3 docs/student_resource/validate_submission.py --matching "$PKG/output/matching_results.tsv" \
    --candidate "$PKG/output/candidate_pairs.tsv" --test-dir data/student_resource/dataset/test
(cd "$PKG" && zip -qr "../${TEAM}_submission.zip" .)  # contents at zip root, as in the spec
ls -lh "$PKG.zip"
