#!/usr/bin/env python3
"""Recognise the 300 test lines with a PaddleOCR text-line recogniser (X5).

Writes <pred_dir>/<stem>.txt, one per line image, in the layout
tamil_ocr_eval.py and sweep_checkpoints.py expect, so PaddleOCR is scored by
exactly the same code as Tesseract.

    # zero-shot, pretrained PaddleOCR Tamil model
    python3 paddle_predict.py --model-name ta_PP-OCRv5_mobile_rec \
        --testset ../testset --pred-dir ../preds_paddle_ta_v5

    # a model exported from our own training run
    python3 paddle_predict.py --model-name ta_PP-OCRv5_mobile_rec \
        --model-dir OUT/inference --testset ../testset --pred-dir ../preds_paddle_ft
"""
import argparse
from pathlib import Path

from paddleocr import TextRecognition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="1 avoids padding narrow lines to a batch's widest")
    args = ap.parse_args()

    kw = {"model_name": args.model_name}
    if args.model_dir:
        kw["model_dir"] = args.model_dir
    rec = TextRecognition(**kw)

    images = sorted((Path(args.testset) / "images").glob("*.tif"))
    assert len(images) == 300, len(images)
    out = Path(args.pred_dir)
    out.mkdir(parents=True, exist_ok=True)
    for res in rec.predict([str(p) for p in images], batch_size=args.batch_size):
        stem = Path(res["input_path"]).stem
        text = " ".join(str(res["rec_text"]).split())
        (out / f"{stem}.txt").write_text(text, encoding="utf-8")
    print(f"{len(list(out.glob('*.txt')))} predictions in {out}")


if __name__ == "__main__":
    main()
