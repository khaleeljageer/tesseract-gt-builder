# Tesseract GT Builder for Tamil OCR

Generates synthetic ground-truth data for training Tesseract's LSTM recogniser
on Tamil, and evaluates the resulting models.

Tamil text is rendered across 27 Unicode typefaces at 300 dpi, then re-segmented
into line crops paired with exact transcriptions in `tesstrain` layout. The
corpus this pipeline produced is archived on Zenodo
([10.5281/zenodo.16881612](https://doi.org/10.5281/zenodo.16881612)).

## Why re-segment

The pipeline renders full A4 pages and then recovers line crops from the
rendered image by horizontal projection profile, rather than emitting crops
from the layout coordinates it already knows. This is deliberate: crops cut
from known coordinates have geometry no real document produces — exact
baselines, exact margins, no cropping jitter. Recovering them through the same
class of operation a segmenter performs at inference means the model trains on
the crop geometry it will actually be given.

## Install

```sh
git clone https://github.com/khaleeljageer/tesseract-gt-builder.git
cd tesseract-gt-builder
pip install -r requirements.txt
```

Training and inference additionally need Tesseract 4+ with the training tools,
plus a [tesstrain](https://github.com/tesseract-ocr/tesstrain) checkout and a
start model:

```sh
sudo apt install tesseract-ocr tesseract-ocr-tam libtesseract-dev
export TESSTRAIN_DIR=/path/to/tesstrain
export TESSDATA_DIR=/path/to/tessdata_best
```

## Generating a corpus

```sh
python normalize-gt.py     # raw_data/*.txt -> data/training-data.txt (12-word lines)
python generate-gt.py      # -> gt/*.tif + gt/*.gt.txt
```

For anything beyond a single default corpus — ablations, per-source control,
deduplication, deterministic font assignment — use the `experiments/` modules
instead, which parameterise the same pipeline:

```python
from experiments import corpus, render

pool  = corpus.build_pool()                                  # per-source line pools
lines = corpus.select(pool, sources=["wikinews-ta"], n_lines=5000, seed=0)
render.generate(lines, "gt/", font_names=None, assignment="round-robin", seed=0)
```

### Font assignment

`generate-gt.py` computes a round-robin font pairing and then discards it:
`create_a4_tiff_image` calls `random.choice` on the per-page window instead of
using the pairing it was handed, and the module-level `random` is never seeded.
Font usage therefore spans roughly 8% between the least- and most-used typeface,
and re-running the generator produces a different corpus.

`experiments/render.py` defaults to `assignment="round-robin"`, which is exactly
balanced and deterministic. `assignment="random"` reproduces the original
behaviour but takes an explicit seed. **Use round-robin for ablations** — under
random assignment two variants differ by more than the variable under test.

## Evaluating a model

```sh
# paired files
python tamil_ocr_eval.py --ground_truth gt.txt --prediction pred.txt

# tesstrain-style directories, with confidence intervals and confusions
python tamil_ocr_eval.py --gt_dir testset/gt --pred_dir testset/pred \
    --bootstrap 2000 --confusions 30 --json results/eval.json
```

Reports **grapheme CER**, **code-point CER** and **WER**, all as true Levenshtein
distance with unit cost for substitution, deletion and insertion. Grapheme CER
treats `கி` as one symbol rather than two code points, which is the more faithful
measure for Tamil; code-point CER is reported alongside it for comparability.

### Two things that will change your numbers

**Zero-width characters.** Tesseract emits ZWNJ liberally around Tamil
conjuncts. These are invisible, carry no recognition information, and are absent
from hand-produced ground truth — but scored literally they can dominate the
error rate. On one sample here they move grapheme CER from 20.76% to 4.83%. The
evaluator strips U+200B/200C/200D/FEFF and collapses whitespace by default,
prints the active normalisation profile on every run, and exposes
`--keep-zero-width` / `--keep-whitespace` to disable it. **Any Tamil OCR result
reported without stating its normalisation profile is not comparable to any
other.**

**NFC, not NFKC.** NFKC applies compatibility folding that rewrites Tamil
numerals and some punctuation, silently changing the denominator.

> The previous evaluator, `cer_wer_tamil.py`, was removed in favour of this one.
> It accumulated `SequenceMatcher` opcode spans on the reference side only, so
> **inserted characters cost nothing**, and it used NFKC. Any figure produced
> with it should be recomputed.

## Ablation studies

Three grids, each varying one thing and holding the rest fixed:

| Grid | Varies | Held fixed |
|---|---|---|
| `fonts` | 5 / 10 / 27 typefaces | 50,000 lines |
| `size` | 10k / 50k / 105,738 lines | 27 typefaces |
| `domain` | literary / newsprint / mixed | 24,000 lines |

```sh
python experiments/run_ablation.py all --list          # show the grid, run nothing
python experiments/run_ablation.py fonts --test-dir testset
python experiments/aggregate.py                        # -> LaTeX tables + figures
```

Every variant is scored against the same held-out set of **real** document
images, which you must build yourself — a model evaluated on synthetic lines
drawn from this same generative process reports an optimistic figure that does
not transfer. Variants with an existing `result.json` are skipped, so an
interrupted sweep resumes.

See `experiments/README.md` for the full workflow and cost estimates
(~2.5 GB of rendered crops per 100k lines).

## Analysis

```sh
python experiments/corpus_stats.py   # syllabary coverage, frequency distribution
python experiments/font_audit.py     # glyph coverage, shaping, font metadata
python find_cfr.py                   # character/word frequency + plot
```

`corpus_stats.py` reports coverage of the 247 traditional Tamil syllabary units.
The current corpus covers 227 across 13.1M graphemes — but 37 units occur fewer
than 27 times, so under round-robin assignment they cannot appear in every
typeface at any corpus size. Models trained on this data should be assumed weak
on those units.

`font_audit.py` verifies that Pillow has Raqm/HarfBuzz support (without it Tamil
is drawn unshaped — no conjunct formation, no vowel-sign reordering, every line
wrong while its transcription stays right) and that every typeface covers every
codepoint the corpus uses.

## Layout

```
normalize-gt.py     merge raw_data/ -> data/training-data.txt
generate-gt.py      render + segment -> gt/
tamil_ocr_eval.py   grapheme/code-point CER, WER, confusions, bootstrap CIs
config.py           page geometry and rendering constants
find_cfr.py         character and word frequency analysis
json2text.py        JSON -> plain text helper
verify.py           sample integrity check (adapt to your layout)

experiments/
  corpus.py         provenance-preserving corpus construction
  render.py         parameterised rendering and segmentation
  train.py          tesstrain / tesseract wrappers
  runner.py         variant orchestration, journal, skip-completed
  run_ablation.py   the three ablation grids
  aggregate.py      results -> LaTeX tables and figures
  corpus_stats.py   syllabary coverage statistics
  font_audit.py     typeface coverage and shaping audit

fonts/              27 Unicode Tamil typefaces
raw_data/           source texts (third-party; see licences below)
model/              trained .traineddata checkpoints
```

## Source texts

`raw_data/` contains third-party material redistributed under its own terms,
not under this repository's licence:

| Source | Licence |
|---|---|
| Tamil Wikisource | CC BY-SA |
| [`aitamilnadu/tamil_stories`](https://huggingface.co/datasets/aitamilnadu/tamil_stories) | Apache-2.0 |
| Theekkathir (theekkathir.in) | CC BY-SA 4.0 |
| Tamil Wikinews | CC BY-SA |
| Maattru | CC BY-SA |

The generated corpus is released under CC BY-SA 4.0.

## Licence

Code is licensed GPL-3.0 — see [LICENSE](LICENSE). Typefaces in `fonts/` and
texts in `raw_data/` are third-party works under their own terms.

## Citation

**Dataset**
```bibtex
@dataset{tamilocr_dataset_2025,
  author    = {Syedkhaleel Jageer},
  title     = {Synthetic OCR Dataset: 105,738 Tamil Text Lines Rendered in 27
               Diverse Fonts with Corresponding Ground Truth Annotations},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.16881612},
  url       = {https://doi.org/10.5281/zenodo.16881612}
}
```

**Code**
```bibtex
@misc{jageer2025tesseractGTBuilder,
  author       = {Syedkhaleel Jageer},
  title        = {{Tesseract-GT-Builder: Tools to generate ground-truth data for
                  Tesseract OCR (Tamil)}},
  howpublished = {\url{https://github.com/khaleeljageer/tesseract-gt-builder}},
  year         = {2025}
}
```
