"""Thin wrappers around tesstrain and tesseract inference.

Nothing here is clever; it exists so the ablation runner has one place to call
and one place to fail loudly when the toolchain is absent.

Environment:
    TESSTRAIN_DIR   checkout of https://github.com/tesseract-ocr/tesstrain
    TESSDATA_DIR    directory holding the start model (e.g. tessdata_best)
"""

import os
import shutil
import subprocess
from pathlib import Path

# Line images: tell Tesseract it is looking at exactly one text line.
# Leaving this at the default causes layout analysis to run on a 30px-tall
# strip, which is a common and silent source of inflated error rates.
PSM_SINGLE_LINE = "7"


class ToolchainError(RuntimeError):
    pass


def require_toolchain(training=True):
    """Fail early and specifically rather than midway through a sweep."""
    problems = []

    if shutil.which("tesseract") is None:
        problems.append(
            "tesseract not on PATH. Install: apt install tesseract-ocr "
            "(and tesseract-ocr-tam for the stock Tamil model).")

    if training:
        tesstrain = os.environ.get("TESSTRAIN_DIR")
        if not tesstrain:
            problems.append(
                "TESSTRAIN_DIR not set. git clone "
                "https://github.com/tesseract-ocr/tesstrain and export "
                "TESSTRAIN_DIR=/path/to/tesstrain")
        elif not (Path(tesstrain) / "Makefile").exists():
            problems.append(f"no Makefile in TESSTRAIN_DIR={tesstrain}")

        if shutil.which("lstmtraining") is None:
            problems.append(
                "lstmtraining not on PATH. Install the training tools: "
                "apt install tesseract-ocr libtesseract-dev "
                "(or build tesseract with --enable-training).")

        tessdata = os.environ.get("TESSDATA_DIR")
        if not tessdata:
            problems.append(
                "TESSDATA_DIR not set. Point it at a tessdata_best checkout "
                "containing tam.traineddata.")

    if problems:
        raise ToolchainError(
            "Toolchain incomplete:\n  - " + "\n  - ".join(problems))


def train(gt_dir, model_name, out_dir, start_model="tam",
          max_iterations=10000, extra_make_args=None, log_path=None):
    """Run tesstrain's `make training` for one variant.

    Returns the path to the produced .traineddata.
    """
    require_toolchain(training=True)
    tesstrain = Path(os.environ["TESSTRAIN_DIR"])
    tessdata = Path(os.environ["TESSDATA_DIR"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "make", "training",
        f"MODEL_NAME={model_name}",
        f"START_MODEL={start_model}",
        f"TESSDATA={tessdata}",
        f"GROUND_TRUTH_DIR={Path(gt_dir).resolve()}",
        f"DATA_DIR={out_dir.resolve()}",
        f"MAX_ITERATIONS={max_iterations}",
    ]
    cmd += list(extra_make_args or [])

    log_path = Path(log_path) if log_path else out_dir / f"{model_name}.train.log"
    with open(log_path, "w") as log:
        log.write(f"$ cd {tesstrain} && {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=tesstrain, stdout=log,
                              stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        raise RuntimeError(
            f"tesstrain failed for {model_name} (exit {proc.returncode}). "
            f"See {log_path}")

    produced = out_dir / model_name / f"{model_name}.traineddata"
    if not produced.exists():
        alt = out_dir / f"{model_name}.traineddata"
        if alt.exists():
            produced = alt
        else:
            raise FileNotFoundError(
                f"training reported success but no .traineddata found under "
                f"{out_dir}. See {log_path}")
    return produced


def recognise(image_dir, out_dir, model, tessdata_dir, psm=PSM_SINGLE_LINE,
              jobs=None):
    """Run inference over every .tif in image_dir, writing <stem>.txt.

    Skips images that already have a prediction, so an interrupted sweep
    resumes rather than restarting.
    """
    require_toolchain(training=False)
    image_dir, out_dir = Path(image_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*.tif"))
    todo = [p for p in images if not (out_dir / f"{p.stem}.txt").exists()]

    for path in todo:
        dest = out_dir / path.stem          # tesseract appends .txt itself
        cmd = ["tesseract", str(path), str(dest),
               "--tessdata-dir", str(tessdata_dir), "-l", model, "--psm", psm]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[warn] tesseract failed on {path.name}: "
                  f"{proc.stderr.strip().splitlines()[:1]}")
            dest.with_suffix(".txt").write_text("", encoding="utf-8")

    return {"images": len(images), "recognised": len(todo),
            "skipped_existing": len(images) - len(todo)}
