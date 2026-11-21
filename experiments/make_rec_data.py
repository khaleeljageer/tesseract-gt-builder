#!/usr/bin/env python3
"""Write PaddleOCR recognition label files for TamilSynth (IJDAR revision, X5).

train.txt  the 196,311 lines of the Tesseract runs' list.train (same lines,
           so the two recognisers see identical training data)
val.txt    2,000 lines sampled (seed 0) from the Tesseract runs' list.eval
test.txt   the 300 real-document test lines
tamilsynth_dict.txt  every character in train.txt (test set not consulted)

    python3 make_rec_data.py --gt-dir GT --list-train L --list-eval L \
        --testset TESTSET --out OUT
"""
import argparse
import random
import unicodedata
from pathlib import Path


def stems(path):
    return [ln.strip().rsplit("/", 1)[1][:-len(".lstmf")]
            for ln in open(path) if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--list-train", required=True)
    ap.add_argument("--list-eval", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-val", type=int, default=2000)
    args = ap.parse_args()

    gt, out, test = Path(args.gt_dir), Path(args.out), Path(args.testset)
    out.mkdir(parents=True, exist_ok=True)
    chars = set()

    def write(name, names, collect):
        n = 0
        with open(out / name, "w", encoding="utf-8") as fh:
            for s in names:
                text = " ".join((gt / f"{s}.gt.txt").read_text(encoding="utf-8").split())
                if not text:
                    continue
                fh.write(f"{gt / (s + '.tif')}\t{text}\n")
                n += 1
                if collect:
                    chars.update(text)
        return n

    print("train", write("train.txt", stems(args.list_train), True))
    ev = stems(args.list_eval)
    random.Random(0).shuffle(ev)
    print("val", write("val.txt", ev[:args.n_val], False))

    chars.discard(" ")
    (out / "tamilsynth_dict.txt").write_text(
        "".join(c + "\n" for c in sorted(chars)), encoding="utf-8")
    print("dict", len(chars))

    unseen = set()
    with open(out / "test.txt", "w", encoding="utf-8") as fh:
        for p in sorted((test / "gt").glob("*.gt.txt")):
            s = p.name[:-len(".gt.txt")]
            text = " ".join(p.read_text(encoding="utf-8").split())
            fh.write(f"{test / 'images' / (s + '.tif')}\t{text}\n")
            unseen.update(set(text) - chars - {" "})
    print("test characters absent from the training dictionary:",
          [f"U+{ord(c):04X} {unicodedata.name(c, '?')}" for c in sorted(unseen)])


if __name__ == "__main__":
    main()
