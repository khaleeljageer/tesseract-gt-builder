# Ablation runners — Experiments 3–5

Scaffolding for the three ablations in the paper plan. Each grid varies one
thing and holds the rest fixed; the fixed quantities are what make the
comparisons mean anything.

| Grid | Varies | Held fixed | Question |
|---|---|---|---|
| `fonts` | 5 / 10 / 27 typefaces | 50,000 lines | Does typographic diversity drive the gain, or is it just data volume? |
| `size` | 10k / 50k / 105,738 lines | 27 typefaces | How much corpus does a reuser actually need? |
| `domain` | literary / newsprint / mixed | 24,000 lines | Does training register transfer across register? |

The domain cap is not arbitrary. The corpus holds 167,245 literary lines
against 24,488 newsprint lines; comparing them at natural size would measure
volume, not register. Both arms are capped to the smaller one.

## Prerequisites

Neither Tesseract nor the training tools are installed on the machine where
this was scaffolded, so nothing here has been run end-to-end past rendering.
The rendering path *is* verified (120 lines → 120 crops, balanced fonts, no
orphans).

```bash
sudo apt install tesseract-ocr tesseract-ocr-tam libtesseract-dev
git clone https://github.com/tesseract-ocr/tesstrain
git clone https://github.com/tesseract-ocr/tessdata_best

export TESSTRAIN_DIR=$PWD/tesstrain
export TESSDATA_DIR=$PWD/tessdata_best

pip install -r requirements.txt
```

The runner checks all of this up front and lists everything missing at once,
rather than dying six hours into a sweep.

## The test set comes first

Every number these grids produce is measured against `testset/`, which must
exist before any of this runs:

```
testset/
  images/  *.tif      one real document line per file
  gt/      *.gt.txt   matching transcriptions
```

See `paper/TESTSET.md`. The test set is fixed across all variants and is
never regenerated — that is the whole point of it.

## Running

```bash
# see the grid without running anything
python experiments/run_ablation.py all --list

# one grid
python experiments/run_ablation.py fonts --test-dir testset

# everything
python experiments/run_ablation.py all --test-dir testset

# collect into LaTeX tables + the scaling figure
python experiments/aggregate.py
```

Variants are skipped if `result.json` already exists, so an interrupted sweep
resumes rather than restarting. Use `--force` to recompute.

## Output layout

```
results/
  journal.jsonl                    one row per variant: hypothesis + outcome
  fonts/f05/
    corpus.txt                     the exact lines used
    corpus.stats.json              grapheme/word statistics
    gt.manifest.json               fonts, per-font line counts, render config
    train.log
    model/                         the .traineddata
    pred/                          predictions on the test set
    result.json                    metrics, CIs, confusions
    code_snapshot/                 the scripts as they were at run time
```

`aggregate.py` writes `paper/tables/*.tex` and `paper/figures/scaling.pdf`.
Swap the placeholder tables in `main.tex` for `\input{tables/ablation_fonts}`
and so on, and the manuscript picks up new numbers on the next compile.

## Two corrections carried in `render.py`

These are behaviour changes against `generate-gt.py`, and both matter for the
paper's Methods section.

**Font assignment was not what the code claimed.** `generate-gt.py` computes a
round-robin pairing in `main()` and then discards it — `create_a4_tiff_image`
calls `random.choice(fonts)` on the per-page window instead of using the
pairing it was handed. The module-level `random` is never seeded either. So
the released corpus was built with *unseeded random sampling with
replacement*, not balanced round-robin: font usage spans roughly 8% between
the least- and most-used typeface, and re-running the generator produces a
different corpus.

`render.py` defaults to `assignment="round-robin"`, which is exactly balanced
and deterministic. `assignment="random"` reproduces the original behaviour but
takes an explicit seed. **Ablations must use round-robin** — under random
assignment two variants differ by more than the variable under test.

This needs reflecting in the paper: either describe the corpus as randomly
assigned, or regenerate it with the deterministic path and describe that.

**Line-height probe.** `generate-gt.py` measures line height from the Latin
string `"Sample"`, which never exercises Tamil ascenders, descenders or the
pulli. `render.py` uses a Tamil probe so tall glyphs are not clipped.

`render.py` also sorts fonts before indexing (`os.listdir` order is
filesystem-dependent, so font indexing was not portable between machines) and
refuses to write a page whose projection-profile band count disagrees with its
line count, rather than silently pairing an image with the wrong transcription.

## Cost

Roughly 2.5 GB of rendered crops per 100k lines; crops are deleted after
training unless `--keep-images`. Nine variants means nine training runs — plan
the wall-clock accordingly and consider running the grids on separate days.

`fonts/f27` and `size/n050k` are the same configuration (50,000 lines, 27
fonts). Left in deliberately: training both gives a free seed-stability check.
Drop one if you would rather have the hours back.
