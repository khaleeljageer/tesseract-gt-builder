#!/usr/bin/env python3
"""Build the X4 (IJDAR revision) training lists from the released images.

The release assigns typefaces round-robin (gt.manifest.json): line i, file
page_{i//50+1:06d}_line_{i%50+1:03d}, is rendered in fonts[i % 29]. Every
subset is therefore a subset of the existing images, with no re-rendering.

  x4_s50   random 50% of lines, all 29 faces
  x4_s34   random 75,215 lines (nested in x4_s50), all 29 faces
  x4_f10   every line in 10 faces drawn at seed 0 (75,215 lines): the
           size-matched font-count contrast to x4_s34

    python3 experiments/x4_lists.py LSTMF_DIR OUT_DIR
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
man = json.loads((HERE / "gt.manifest.json").read_text())
assert man["assignment"] == "round-robin"
fonts, n = man["fonts"], man["crops_written"]
per = man["config"]["lines_per_page"]
gt, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)


def name(i):
    return str(gt / f"page_{i // per + 1:06d}_line_{i % per + 1:03d}.lstmf")


faces = sorted(random.Random(0).sample(fonts, 10))
f10 = [i for i in range(n) if fonts[i % len(fonts)] in faces]
order = list(range(n))
random.Random(0).shuffle(order)
lists = {"x4_s50": sorted(order[: n // 2]),
         "x4_s34": sorted(order[: len(f10)]),
         "x4_f10": f10}
for k, idx in lists.items():
    missing = [i for i in idx if not Path(name(i)).exists()]
    assert not missing, (k, len(missing))
    (out / f"{k}.txt").write_text("".join(name(i) + "\n" for i in idx))
    print(k, len(idx))
(out / "x4_faces.json").write_text(json.dumps(
    {"f10_faces": faces, "all_faces": fonts, "seed": 0}, indent=2))
print("f10 faces:", faces)
