"""
Tamil OCR evaluation: grapheme CER, code-point CER, and WER.

Replaces the SequenceMatcher-based metric in cer_wer_tamil.py, which counted
only reference-side spans and therefore charged nothing for inserted
characters. All rates here are true Levenshtein edit distance.

Two reporting levels:

  micro   corpus-level rate, sum(edits) / sum(reference units) across all
          lines. This is the number to report in a paper.
  macro   mean of per-line rates. Reported alongside micro because a large
          gap between the two indicates the error is concentrated in a few
          lines rather than spread across the corpus.

Usage:
    # paired single files (one document per file)
    python tamil_ocr_eval.py --ground_truth gt.txt --prediction pred.txt

    # line-aligned files (line i of gt pairs with line i of pred)
    python tamil_ocr_eval.py --ground_truth gt.txt --prediction pred.txt --by-line

    # tesstrain-style directory: *.gt.txt paired with *.txt predictions
    python tamil_ocr_eval.py --gt_dir test/gt --pred_dir test/pred

    # add bootstrapped confidence intervals and a confusion listing
    python tamil_ocr_eval.py --gt_dir test/gt --pred_dir test/pred \
        --bootstrap 2000 --confusions 25
"""

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Tamil combining marks: dependent vowel signs U+0BBE-U+0BCC, virama (pulli)
# U+0BCD, and the AU length mark U+0BD7. A grapheme cluster is one base
# character followed by any run of these.
_TAMIL_COMBINING = set(range(0x0BBE, 0x0BCE)) | {0x0BD7}

# Zero-width formatting characters. Tesseract emits ZWNJ liberally around
# Tamil conjuncts; these are invisible and carry no recognition information,
# but scored naively they dominate the error rate. On the sample pairs in
# cer_wer/, stripping them moves grapheme CER from 20.76% to 4.83%.
_ZERO_WIDTH = dict.fromkeys([0x200B, 0x200C, 0x200D, 0xFEFF])

# Active normalization profile, recorded so the paper can state it exactly.
_PROFILE = {"strip_zero_width": True, "collapse_whitespace": True}


def normalize(text: str) -> str:
    """Canonical composition plus the configured cleanup.

    NFC, not NFKC: NFKC applies compatibility folding, which rewrites Tamil
    numerals and some punctuation and so silently changes the denominator of
    every rate below.

    The two cleanup steps are on by default and are reported in the output.
    Both are decisions a reader must be able to see, because each moves the
    headline number materially.
    """
    text = unicodedata.normalize("NFC", text)
    if _PROFILE["strip_zero_width"]:
        text = text.translate(_ZERO_WIDTH)
    if _PROFILE["collapse_whitespace"]:
        text = re.sub(r"\s+", " ", text)
    return text


def graphemes(text: str) -> list:
    """Split Tamil text into user-perceived characters.

    'கி' is one grapheme and two code points. Grapheme CER is the more
    meaningful rate for Tamil, because a single missed vowel sign should count
    as one error rather than being diluted across code points.

    Uses open-tamil when available so results match find_cfr.py; falls back to
    an equivalent combining-mark rule otherwise.
    """
    try:
        from tamil import utf8

        return utf8.get_letters(text)
    except ImportError:
        pass

    out = []
    for ch in text:
        if out and ord(ch) in _TAMIL_COMBINING:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def edit_distance(ref: list, hyp: list):
    """Levenshtein distance with unit cost for substitution, deletion and
    insertion. Returns (distance, aligned_pairs).

    aligned_pairs holds the (reference, hypothesis) pairs on the optimal path,
    with None marking a deletion or insertion. It is what feeds the confusion
    listing.
    """
    n, m = len(ref), len(hyp)
    # d[i][j] = distance between ref[:i] and hyp[:j]
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        ri = ref[i - 1]
        di, dprev = d[i], d[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            di[j] = min(dprev[j] + 1, di[j - 1] + 1, dprev[j - 1] + cost)

    # Backtrace one optimal alignment.
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                pairs.append((ref[i - 1], hyp[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            pairs.append((ref[i - 1], None))  # deletion
            i -= 1
            continue
        pairs.append((None, hyp[j - 1]))  # insertion
        j -= 1

    pairs.reverse()
    return d[n][m], pairs


def score_pair(ref_text: str, hyp_text: str):
    """All three metrics for one reference/hypothesis pair."""
    ref_text = normalize(ref_text).strip()
    hyp_text = normalize(hyp_text).strip()

    g_ref, g_hyp = graphemes(ref_text), graphemes(hyp_text)
    c_ref, c_hyp = list(ref_text), list(hyp_text)
    w_ref, w_hyp = ref_text.split(), hyp_text.split()

    g_dist, g_pairs = edit_distance(g_ref, g_hyp)
    c_dist, _ = edit_distance(c_ref, c_hyp)
    w_dist, _ = edit_distance(w_ref, w_hyp)

    return {
        "grapheme_edits": g_dist,
        "grapheme_ref": len(g_ref),
        "codepoint_edits": c_dist,
        "codepoint_ref": len(c_ref),
        "word_edits": w_dist,
        "word_ref": len(w_ref),
        "alignment": g_pairs,
    }


def _rate(edits: int, total: int) -> float:
    if total == 0:
        return 0.0 if edits == 0 else 1.0
    return edits / total


def aggregate(per_line: list) -> dict:
    """Corpus-level (micro) and per-line-mean (macro) rates."""

    def micro(e_key, r_key):
        return _rate(sum(r[e_key] for r in per_line), sum(r[r_key] for r in per_line))

    def macro(e_key, r_key):
        if not per_line:
            return 0.0
        return sum(_rate(r[e_key], r[r_key]) for r in per_line) / len(per_line)

    return {
        "lines": len(per_line),
        "cer_grapheme_micro": micro("grapheme_edits", "grapheme_ref"),
        "cer_grapheme_macro": macro("grapheme_edits", "grapheme_ref"),
        "cer_codepoint_micro": micro("codepoint_edits", "codepoint_ref"),
        "wer_micro": micro("word_edits", "word_ref"),
        "wer_macro": macro("word_edits", "word_ref"),
        "total_graphemes": sum(r["grapheme_ref"] for r in per_line),
        "total_words": sum(r["word_ref"] for r in per_line),
    }


def bootstrap_ci(per_line: list, e_key: str, r_key: str, n_resamples: int,
                 alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for a micro-averaged rate, resampling lines.

    Resampling lines (not characters) is the right unit here: lines are the
    independent observations, characters within a line are not.
    """
    if len(per_line) < 2:
        return None
    rng = random.Random(seed)
    n = len(per_line)
    rates = []
    for _ in range(n_resamples):
        sample = [per_line[rng.randrange(n)] for _ in range(n)]
        rates.append(_rate(sum(r[e_key] for r in sample),
                           sum(r[r_key] for r in sample)))
    rates.sort()
    lo = rates[int((alpha / 2) * n_resamples)]
    hi = rates[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return lo, hi


def confusion_counts(per_line: list):
    """Substitution, deletion and insertion counts over the grapheme alignment."""
    subs, dels, ins = Counter(), Counter(), Counter()
    for r in per_line:
        for a, b in r["alignment"]:
            if a is None:
                ins[b] += 1
            elif b is None:
                dels[a] += 1
            elif a != b:
                subs[(a, b)] += 1
    return subs, dels, ins


def read(path) -> str:
    return Path(path).read_text(encoding="utf-8")


def collect_pairs(args):
    """Build the list of (name, reference, hypothesis) triples to score."""
    if args.gt_dir:
        gt_dir, pred_dir = Path(args.gt_dir), Path(args.pred_dir)
        pairs, missing = [], []
        for gt_file in sorted(gt_dir.glob("*.gt.txt")):
            stem = gt_file.name[: -len(".gt.txt")]
            pred = pred_dir / f"{stem}.txt"
            if not pred.exists():
                missing.append(stem)
                continue
            pairs.append((stem, read(gt_file), read(pred)))
        if missing:
            print(f"[warn] {len(missing)} ground-truth files had no prediction "
                  f"(first: {missing[0]})", file=sys.stderr)
        return pairs

    gt, pred = read(args.ground_truth), read(args.prediction)
    if args.by_line:
        gt_lines = gt.splitlines()
        pred_lines = pred.splitlines()
        if len(gt_lines) != len(pred_lines):
            print(f"[warn] line-count mismatch: {len(gt_lines)} reference vs "
                  f"{len(pred_lines)} hypothesis. Pairing by index; the tail is "
                  f"scored against empty strings.", file=sys.stderr)
        n = max(len(gt_lines), len(pred_lines))
        gt_lines += [""] * (n - len(gt_lines))
        pred_lines += [""] * (n - len(pred_lines))
        return [(f"line_{i+1}", g, p)
                for i, (g, p) in enumerate(zip(gt_lines, pred_lines))]

    return [(Path(args.ground_truth).name, gt, pred)]


def main():
    ap = argparse.ArgumentParser(
        description="Grapheme CER, code-point CER and WER for Tamil OCR.")
    src = ap.add_argument_group("input")
    src.add_argument("--ground_truth", help="reference text file")
    src.add_argument("--prediction", help="hypothesis text file")
    src.add_argument("--by-line", action="store_true",
                     help="treat each line as a separate observation")
    src.add_argument("--gt_dir", help="directory of *.gt.txt references")
    src.add_argument("--pred_dir", help="directory of *.txt hypotheses")

    norm = ap.add_argument_group("normalization (reported in output)")
    norm.add_argument("--keep-zero-width", action="store_true",
                      help="score ZWNJ/ZWJ/ZWSP as real characters "
                           "(default: strip them)")
    norm.add_argument("--keep-whitespace", action="store_true",
                      help="score line breaks and runs of spaces literally "
                           "(default: collapse to single spaces)")

    ap.add_argument("--bootstrap", type=int, default=0, metavar="N",
                    help="bootstrap resamples for 95%% CIs (e.g. 2000)")
    ap.add_argument("--confusions", type=int, default=0, metavar="K",
                    help="print the K most frequent grapheme confusions")
    ap.add_argument("--json", metavar="PATH", help="write full results as JSON")
    args = ap.parse_args()

    if args.gt_dir and not args.pred_dir:
        ap.error("--gt_dir requires --pred_dir")
    if not args.gt_dir and not (args.ground_truth and args.prediction):
        ap.error("supply either --gt_dir/--pred_dir or --ground_truth/--prediction")

    _PROFILE["strip_zero_width"] = not args.keep_zero_width
    _PROFILE["collapse_whitespace"] = not args.keep_whitespace

    pairs = collect_pairs(args)
    if not pairs:
        print("No reference/hypothesis pairs found.", file=sys.stderr)
        return 1

    per_line = []
    for name, ref, hyp in pairs:
        r = score_pair(ref, hyp)
        r["name"] = name
        per_line.append(r)

    agg = aggregate(per_line)

    profile = ", ".join([
        "NFC",
        "zero-width stripped" if _PROFILE["strip_zero_width"] else "zero-width kept",
        "whitespace collapsed" if _PROFILE["collapse_whitespace"] else "whitespace literal",
    ])
    print(f"Normalization:       {profile}")
    print(f"Lines scored:        {agg['lines']}")
    print(f"Reference graphemes: {agg['total_graphemes']}")
    print(f"Reference words:     {agg['total_words']}")
    print()
    print(f"CER (grapheme, micro):  {agg['cer_grapheme_micro'] * 100:6.2f}%")
    print(f"CER (grapheme, macro):  {agg['cer_grapheme_macro'] * 100:6.2f}%")
    print(f"CER (codepoint, micro): {agg['cer_codepoint_micro'] * 100:6.2f}%")
    print(f"WER (micro):            {agg['wer_micro'] * 100:6.2f}%")
    print(f"WER (macro):            {agg['wer_macro'] * 100:6.2f}%")

    if args.bootstrap:
        ci = bootstrap_ci(per_line, "grapheme_edits", "grapheme_ref", args.bootstrap)
        wci = bootstrap_ci(per_line, "word_edits", "word_ref", args.bootstrap)
        if ci:
            print()
            print(f"95% CI, grapheme CER: [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%] "
                  f"({args.bootstrap} resamples over lines)")
            print(f"95% CI, WER:          [{wci[0]*100:.2f}%, {wci[1]*100:.2f}%]")
            agg["cer_grapheme_ci95"] = ci
            agg["wer_ci95"] = wci

    if args.confusions:
        subs, dels, ins = confusion_counts(per_line)
        print()
        print(f"Top {args.confusions} grapheme substitutions (reference -> hypothesis):")
        for (a, b), n in subs.most_common(args.confusions):
            print(f"  {a} -> {b}   {n}")
        print(f"\nTop deletions:  " +
              ", ".join(f"{c}({n})" for c, n in dels.most_common(10)))
        print(f"Top insertions: " +
              ", ".join(f"{c}({n})" for c, n in ins.most_common(10)))
        agg["substitutions"] = [
            {"ref": a, "hyp": b, "count": n} for (a, b), n in subs.most_common()
        ]

    if args.json:
        payload = {
            "aggregate": agg,
            "per_line": [
                {k: v for k, v in r.items() if k != "alignment"} for r in per_line
            ],
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
