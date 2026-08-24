#!/usr/bin/env bash
# Build an anonymised mirror of this repository for double-blind review.
#
# It does NOT touch this repository. Rewriting history here would break every
# existing clone and the published Zenodo record still carries the author's
# name regardless, so the mirror is a separate artifact: a fresh repository
# with one commit, authored by a neutral identity, with the identifying
# metadata removed.
#
# What leaks identity, and what this handles:
#
#   git commit authorship   59 commits by two addresses -> single neutral commit
#   .github/FUNDING.yml     a sponsor username          -> removed
#   README.md               name, repo URL, Zenodo DOI  -> withheld
#   training logs           absolute paths under $HOME  -> already scrubbed
#
# What it cannot handle, and you must decide about:
#
#   The Zenodo record is public under the author's name. A reviewer who
#   searches a distinctive phrase from the corpus description will find it.
#   Most venues permit citing a pre-existing public resource; none permit the
#   paper pointing at it. The paper withholds both URLs -- keep it that way.
#
# Usage: ./anonymise_for_review.sh [output-dir]

set -euo pipefail
OUT="${1:-../tesseract-gt-builder-anon}"
SRC="$(git rev-parse --show-toplevel)"

[ -e "$OUT" ] && { echo "refusing to overwrite $OUT"; exit 1; }

# The corpus itself is 397k tracked files and belongs on the archive, not in
# a review mirror; reviewers need the code, the test set and the results.
EXCLUDE='gt_v2_fixedwidth/* gt/* preds_*/*'

echo "exporting tracked files to $OUT (excluding bulk corpus data)"
mkdir -p "$OUT"
# shellcheck disable=SC2086
git archive HEAD $(printf -- ':(exclude)%s ' $EXCLUDE) | tar -x -C "$OUT"

cd "$OUT"
rm -rf .github

# The README is the repository's public identity; the mirror needs a
# different one rather than a redacted version of the same document.
cat > README.md <<'EOF'
# Anonymous submission: synthetic Tamil OCR corpus and pipeline

This is an anonymised mirror provided for double-blind review. Author,
repository and archive identifiers are withheld and will be restored at
camera-ready.

See the paper for what this contains. Everything needed to reproduce the
reported results is here; the entry points are `regenerate_corpus.py`,
`prepare_lstmf.py`, `build_dawgs.py`, `make_testset.py` and
`tamil_ocr_eval.py`, and the reproduction recipe is in the paper's appendix.
EOF

# Anything left that names the author, the host, or the archive.
LEAKS=$(grep -rlniE "khaleel|jageer|syedkhaleel|zilogic|zenodo\.[0-9]+" \
        --binary-files=without-match --exclude-dir=.git . 2>/dev/null || true)
if [ -n "$LEAKS" ]; then
    echo "!! identifying strings remain in:"
    echo "$LEAKS" | sed 's/^/     /'
    echo "   edit these before publishing the mirror"
fi

git init -q
git add -A
GIT_AUTHOR_NAME="Anonymous" GIT_AUTHOR_EMAIL="anonymous@example.invalid" \
GIT_COMMITTER_NAME="Anonymous" GIT_COMMITTER_EMAIL="anonymous@example.invalid" \
    git commit -q -m "Anonymised submission mirror"

echo
echo "mirror at $OUT"
echo "  commits:  $(git rev-list --count HEAD) by $(git log --format='%an' | sort -u)"
echo "  files:    $(git ls-files | wc -l)"
echo
echo "  The corpus and test-set images are excluded; point reviewers at the"
echo "  anonymous archive record for those. Publish this via"
echo "  https://anonymous.4open.science/ and put that URL in the paper in"
echo "  place of the withheld one."
