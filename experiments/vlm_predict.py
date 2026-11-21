#!/usr/bin/env python3
"""Zero-shot transcription of the 300 test lines by a local open-weight
vision-language model served by Ollama (multimodal comparator, R2.8).

Runs entirely on the local machine: no test image leaves it. Decoding is
greedy (temperature 0, fixed seed), the prompt is fixed, and the only
post-processing is mechanical (code fences, surrounding quotes, line breaks
joined by a space); every line that needed it is counted in the metadata.

Cost recorded per line: wall-clock seconds, Ollama's own eval/prompt timings,
and peak GPU memory (nvidia-smi polled every 0.2 s while the model runs).
`--cpu` forces CPU-only inference (num_gpu 0) for the cost comparison with
Tesseract, normally on a subset (`--limit`).

    python3 vlm_predict.py --model gemma3:12b --images testset/images \
        --out results/vlm_preds/gemma3_12b [--names names.txt] [--cpu]
"""
import argparse
import base64
import io
import json
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from PIL import Image

URL = "http://localhost:11434/api/generate"
PROMPT = ("This image is one line of printed Tamil text. Transcribe it exactly, "
          "character for character, in Tamil Unicode. Output only the text of "
          "the line: no translation, no explanation, no quotation marks.")


def png_b64(path):
    im = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def clean(text):
    raw = text
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    if len(t) >= 2 and t[0] in "\"'“‘«" and t[-1] in "\"'”’»":
        t = t[1:-1].strip()
    t = " ".join(s.strip() for s in t.splitlines() if s.strip())
    return t, t != raw.strip()


class GpuPeak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                self.peak = max(self.peak, int(out.split()[0]))
            except Exception:
                pass
            time.sleep(0.2)


def call(model, img, cpu):
    body = {"model": model, "prompt": PROMPT, "images": [img], "stream": False,
            "think": False, "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 0, "num_predict": 400,
                        "num_ctx": 4096}}
    if cpu:
        body["options"]["num_gpu"] = 0
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 400 and "think" in e.read().decode():
            body.pop("think")
            req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--names", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    names = (a.names.read_text().split() if a.names else
             sorted(p.stem for p in a.images.glob("*.tif")))
    if a.limit:
        names = names[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    # warm-up (load weights) on the first image, not timed
    call(a.model, png_b64(a.images / f"{names[0]}.tif"), a.cpu)
    # Timings are only valid if the whole model (layers and vision projector)
    # sits on the GPU, alone; Ollama silently splits it when VRAM is short.
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    loaded = [l for l in ps.splitlines()[1:] if l.strip()]
    if not a.cpu and (len(loaded) != 1 or "100% GPU" not in loaded[0]):
        raise SystemExit(f"model not alone and fully on GPU:\n{ps}")
    gpu = GpuPeak()
    gpu.start()
    meta = {"model": a.model, "cpu": a.cpu, "prompt": PROMPT, "lines": {}}
    t_all = time.time()
    for i, n in enumerate(names):
        img = png_b64(a.images / f"{n}.tif")
        t0 = time.time()
        r = call(a.model, img, a.cpu)
        wall = time.time() - t0
        text, cleaned = clean(r.get("response", ""))
        (a.out / f"{n}.txt").write_text(text + "\n", encoding="utf-8")
        meta["lines"][n] = {"raw": r.get("response", ""), "cleaned": cleaned,
                            "wall_s": round(wall, 3),
                            "eval_count": r.get("eval_count"),
                            "prompt_eval_count": r.get("prompt_eval_count"),
                            "done_reason": r.get("done_reason")}
        print(f"{i+1}/{len(names)} {n} {wall:.2f}s {text[:60]}", flush=True)
    gpu.stop.set()
    walls = [v["wall_s"] for v in meta["lines"].values()]
    meta["summary"] = {
        "n": len(walls), "total_s": round(time.time() - t_all, 1),
        "mean_s_per_line": round(sum(walls) / len(walls), 3),
        "median_s_per_line": round(sorted(walls)[len(walls) // 2], 3),
        "peak_gpu_mib": gpu.peak,
        "n_cleaned": sum(v["cleaned"] for v in meta["lines"].values()),
        "n_empty": sum(not (a.out / f"{n}.txt").read_text().strip() for n in names),
        "n_truncated": sum(v["done_reason"] == "length" for v in meta["lines"].values()),
    }
    try:
        show = subprocess.run(["ollama", "show", a.model], capture_output=True,
                              text=True).stdout
        meta["ollama_show"] = show
        meta["ollama_ps"] = subprocess.run(["ollama", "ps"], capture_output=True,
                                           text=True).stdout
    except Exception:
        pass
    (a.out / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(meta["summary"]))


if __name__ == "__main__":
    main()
