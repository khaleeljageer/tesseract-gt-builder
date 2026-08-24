#!/usr/bin/env python3
"""Build the dictionary DAWGs and pack them into a fine-tuned model.

tesstrain reads its wordlist from $(OUTPUT_DIR)/$(MODEL_NAME).wordlist and
its punctuation and number patterns from the matching .punc and .numbers.
When those files are absent -- which is the default, since nothing creates
them -- combine_lang_model is handed three paths that do not exist and
emits a traineddata with no dictionary at all. That is what happened to v3:
3.1 MB against stock tam's 6.0 MB, the whole difference being the DAWGs.

Tesseract without a dictionary still recognises, it just loses the language
model that resolves ambiguous glyphs, and on this test set that costs about
0.7 points of CER and 0.5 of WER.

Grafting stock tam's DAWGs across is safe but not best. Safe because
combine_lang_model builds our unicharset as a strict superset of stock's
with the IDs preserved -- all 99 of stock's units keep their positions and
ours appends nine (– ' ' ௌ — ஔ + # \\) -- so the IDs inside stock's DAWG
still denote the characters it was built for. Not best because stock's
251,899 words were chosen for a different corpus: building the wordlist
from our own 2.4M tokens instead is worth a further 0.44 CER points, and
the union of the two is worse than ours alone, so this is a matter of
matching the vocabulary rather than of having more of it.

Usage:
    python3 build_dawgs.py --checkpoint ~/tesstrain/data/tam_v3/checkpoints/tam_v3_4.959_1059_1400.checkpoint \\
                           --traineddata ~/tesstrain/data/tam_v3/tam_v3.traineddata \\
                           --out model/tam_v3.traineddata
"""

import argparse
import collections
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "experiments"))
import corpus  # noqa: E402

TAMIL = re.compile(r"[஀-௿]")
EDGE = re.compile(r"^[^஀-௿\w]+|[^஀-௿\w]+$")


def wordlist(raw_dir, min_freq=1):
    """Unique repaired Tamil words, most frequent first is not needed --
    wordlist2dawg sorts internally -- but the frequency filter is, if you
    want to trade coverage for size."""
    freq = collections.Counter()
    for path in sorted(Path(raw_dir).glob("*.txt")):
        for token in path.read_text(encoding="utf-8").split():
            word = corpus.repair_tamil(EDGE.sub("", token))
            if word and TAMIL.search(word) and corpus.well_formed(word):
                freq[word] += 1
    return sorted(w for w, c in freq.items() if c >= min_freq)


def run(*cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"{cmd[0]} failed:\n{proc.stderr}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="checkpoint to package")
    ap.add_argument("--traineddata", required=True,
                    help="the run's tam_v3.traineddata, for --stop_training")
    ap.add_argument("--stock", default="/usr/share/tesseract-ocr/5/tessdata/tam.traineddata",
                    help="source of the punc and number DAWGs")
    ap.add_argument("--raw-dir", default="raw_data")
    ap.add_argument("--min-freq", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # The recognition model first: --stop_training strips the optimiser
        # state and leaves a traineddata carrying lstm, unicharset, recoder.
        run("lstmtraining", "--stop_training",
            "--continue_from", args.checkpoint,
            "--traineddata", args.traineddata,
            "--model_output", str(tmp / "model.traineddata"))

        # Its unicharset is the one the DAWGs must be built against.
        run("combine_tessdata", "-e", str(tmp / "model.traineddata"),
            str(tmp / "uc.lstm-unicharset"))

        words = wordlist(args.raw_dir, args.min_freq)
        (tmp / "words").write_text("\n".join(words) + "\n", encoding="utf-8")
        print(f"{len(words):,} words from {args.raw_dir}")
        run("wordlist2dawg", str(tmp / "words"),
            str(tmp / "d.lstm-word-dawg"), str(tmp / "uc.lstm-unicharset"))

        # Punctuation and number patterns are not corpus-specific and stock's
        # are ID-compatible, so take them rather than inventing our own.
        run("combine_tessdata", "-e", args.stock,
            str(tmp / "d.lstm-punc-dawg"), str(tmp / "d.lstm-number-dawg"))

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp / "model.traineddata", out)
        run("combine_tessdata", "-o", str(out),
            str(tmp / "d.lstm-word-dawg"),
            str(tmp / "d.lstm-punc-dawg"),
            str(tmp / "d.lstm-number-dawg"))

    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    subprocess.run(["combine_tessdata", "-d", str(out)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
