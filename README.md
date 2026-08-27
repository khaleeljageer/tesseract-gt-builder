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

## Fine-tuning

```bash
python3 prepare_lstmf.py --gt-dir gt --jobs 10          # LANG_TYPE=Indic by default

cd $TESSTRAIN_DIR && make training \
  MODEL_NAME=tam_v3 START_MODEL=tam LANG_TYPE=Indic \
  TESSDATA=/usr/share/tesseract-ocr/5/tessdata \
  GROUND_TRUTH_DIR=$PWD/gt MAX_ITERATIONS=100000

python3 build_dawgs.py \
  --checkpoint $TESSTRAIN_DIR/data/tam_v3/checkpoints/tam_v3_4.959_1059_1400.checkpoint \
  --traineddata $TESSTRAIN_DIR/data/tam_v3/tam_v3.traineddata \
  --out model/tam_v3.traineddata
```

### Reproducing the defaults arm

The same corpus trained with `LANG_TYPE` unset, which is what separates
paragraph-aware layout from the training settings. Results in
`results/tam_unset.json`, predictions in `preds_unset/`, training log in
`results/training/tam_unset_training.log`.

```bash
# stage the same images with per-character boxes
python3 prepare_lstmf.py --gt-dir gt_unset --jobs 10 --lang-type ""

cd $TESSTRAIN_DIR && make training \
  MODEL_NAME=tam_unset START_MODEL=tam LANG_TYPE= \
  TESSDATA=/usr/share/tesseract-ocr/5/tessdata \
  GROUND_TRUTH_DIR=$PWD/gt_unset MAX_ITERATIONS=100000
```

On the 202-line reporting split this scores 12.98% grapheme CER against the
fixed-width corpus's 14.46% under the same defaults, and against 7.36% for
the same corpus with the workflow corrected. Redesigning the corpus is worth
1.48 points; correcting the workflow is worth 5.62.

### tesstrain settings for a syllabic script

Three settings decide whether fine-tuning helps at all. Each is silent and
each is what tesstrain does if you say nothing.

**`LANG_TYPE=Indic` is not optional, and it must be on the `make` line.** It
selects two things. `generate_wordstr_box.py` in place of
`generate_line_box.py`, which boxes the whole line rather than splitting
syllables — `generate_line_box.py` groups a mark with its consonant only when
the mark's canonical combining class is non-zero, and every Tamil vowel sign
has class 0, so `லி` becomes `ல` + `ி`. And `--pass_through_recoder` in
`combine_lang_model`, which keeps the pre-trained code space instead of
rebuilding it. The second matters more. Without it the model gets
`null_char=106` where stock tam has `2`, so `--continue_from` resumes stock's
weights into a code space they were never trained for. The symptom is a run
that starts far worse than the model it continued from and never catches up,
however long you train it. With the flag, a couple of hundred iterations
already match the start model. Check with `combine_tessdata -l` before you
train for a day.

**`make training` exports the best *training* BCER, which is the wrong model.**
Training BCER falls monotonically for the whole run while real-page CER turns
around early and climbs from there. Sweep the checkpoints and choose on real pages, not on the training
curve; the difference between the exported checkpoint and the best one on
real documents was several CER points in our runs. Choose on a split you
then do not report, or you are fitting the test set.

**No wordlist file means no dictionary.** `combine_lang_model` reads
`data/$MODEL/$MODEL.wordlist`, `.punc` and `.numbers`; nothing creates them, so
it is handed three paths that do not exist and emits a traineddata with no
DAWGs at all — 3.1 MB against stock's 6.0 MB. `build_dawgs.py` builds the word
DAWG from `raw_data` and takes the punctuation and number patterns from stock.
Building the wordlist from this corpus beats grafting stock's, and beats the
union of the two, so match the vocabulary rather than maximising it.

### Corpus hygiene

`unicharset_extractor` rejects lines whose Tamil is ill-formed —
`Invalid start of grapheme sequence` — and they then reach training as label
noise. Almost all of it is one artifact: in Bamini, TAB and TSCII the glyph for
`ர` sits at the codepoint Unicode gives to `ா`, so converted archives spell
`ர ்` as `ா ்`. `corpus.repair_tamil` undoes that and three related patterns;
every pattern it matches is illegal Tamil, so it cannot touch well-formed text
(103 words changed in 2,418,180, the rest byte-identical). What it cannot mend
— a word whose base consonant is simply missing — `corpus.well_formed` drops.

`audit_gt.py` finds these in an already-rendered build. It does not repair
them, because the `.tif` was rendered from the broken text and depicts the
broken glyphs; a repaired `.gt.txt` would no longer describe its image. Repair
belongs in `corpus.py`, before rendering.

The published v3 corpus keeps its 123 affected lines (0.056%), listed in
`corpus_defects_v3.txt`, so that it remains exactly the corpus the published
model was trained on. Use `audit_gt.py --delete` if you would rather drop them
and retrain.

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
| Tamil Wikisource | CC BY-SA 4.0 |
| [`aitamilnadu/tamil_stories`](https://huggingface.co/datasets/aitamilnadu/tamil_stories) | Apache-2.0 |
| Theekkathir ([theekkathir.in](https://theekkathir.in)) | CC BY-SA 4.0 |
| Tamil Wikinews | CC BY-SA 4.0 |
| Maattru ([maattru.in](https://maattru.in)) | CC BY-SA 4.0 |

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
