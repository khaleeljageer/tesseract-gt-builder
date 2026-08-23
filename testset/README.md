# Real-document test set

300 hand-transcribed lines of printed Tamil, drawn from 295 source pages.
This is the material the headline number is measured on. Nothing here comes
from the generation pipeline — see `paper/TESTSET.md` for why that matters.

## Layout

    images/        300 line crops, one per file (.tif)
    gt/            300 transcriptions — EMPTY, yours to fill in
    gt_prefill/    stock-`tam` output for the same 300 lines
    manifest.csv   provenance for every drawn line
    reserve.csv    300 replacements, in draw order
    pool/          all 5,227 candidate lines (not committed)

## How it was built

    for d in Books Newsprint Forms Signage Degraded; do
      python3 make_testset.py segment --pages <scans>/$d --out testset/pool --stratum $d
    done
    python3 make_testset.py sample --pool testset/pool --out testset --seed 0

Seed 0. Same seed and same pool reproduces the same 300 lines exactly.

## Allocation

| Stratum | Lines | Pages | Material |
|---|---|---|---|
| newsprint | 80 | 80 | Tamil newspaper column crops, photographed |
| books | 80 | 79 | Photographed book pages |
| forms | 60 | 59 | Government forms and minutes, born-digital PDF at 300 dpi |
| signage | 40 | 40 | Tamil Nadu government announcements booklet, 2011–12 |
| degraded | 40 | 37 | Aged booklet scans — show-through, low contrast |

## What the sampler rejected, and what it did not

Rejections are **mechanical only**: blank crops, crops too short or too tall
against their stratum's median, aspect ratios too square to be a line, and
crops carrying a fragment of a neighbouring line (page curl, which deskewing
does not correct), and crops with no text left once horizontal rules and
solid panels are discounted. Nothing was rejected for being hard to read.
Filtering on whether an engine can read a line would select for easy lines
and make the headline figure optimistic.

The floors are deliberately low. Type size varies within a document, and a
threshold set high enough to catch every half-clipped line would also discard
the legitimately small print in the forms. Reject what you find by eye during
transcription:

    python3 make_testset.py replace --out testset --lines <id> [<id> ...]

which takes the next unused entry from `reserve.csv` for the same stratum.
Choosing a replacement yourself would select for legibility.

Reversed lines — light text knocked out of a coloured panel — are kept and
flipped to dark-on-light at crop time. `manifest.csv` records which: 10 of
the 300 drawn lines. Report the proportion; do not hide it.

## Two cautions

**The signage material is born-digital.** 31 of its 40 pages are vector text
rasterised at 300 dpi, not captures, and stock `tam` already reads them
better than any other stratum. It is a real out-of-set typography test —
the face is `VANAVILAvvaiyar`, a legacy non-Unicode font absent from the 29
used for training — but it does not test capture degradation. Describe it as
a government publication in a legacy typeface, not as photographed signage.

**Do not extract ground truth from the source PDFs.** They are
legacy-encoded: `pdftotext` yields `jäœ ts®¢Á¤ Jiw`, not
`தமிழ் வளர்ச்சித் துறை`. Transcribe visually.

## On `gt_prefill/`

Stock `tam` output for each line, 283 of 300 non-empty. It is there to be
corrected against the image, not accepted. Copy a file into `gt/` only after
you have read it against its crop; anything you do not check is not data.

Prefilling with the model under evaluation would be circular. Stock `tam` is
the baseline, so an error you fail to catch flatters the baseline and
understates the fine-tuned model — the conservative direction. It still
anchors your reading, which is why `gt/` starts empty and this is opt-in.

## Source scans

`raw_scans/` in this repository — 300 page images across the five strata,
gitignored for size and because redistributing whole pages of published books
is a different question from quoting single lines of them for evaluation.
**Back them up outside git**; the pool cannot be rebuilt without them.

Rebuild the pool with:

    for d in Books Newsprint Forms Signage Degraded; do
      python3 make_testset.py segment --pages raw_scans/$d --out testset/pool --stratum $d
    done

`manifest.csv` has empty `device`, `dpi` and `notes` columns. Fill them in
per source document while you still remember how each was captured.

## Models

`~/tessdata/` is a user-local TESSDATA_PREFIX that mirrors the system one and
adds the model under evaluation. Pass `--tessdata $HOME/tessdata` to
`predict`, or export TESSDATA_PREFIX.

| `-l` name | Model |
|---|---|
| `tam` | stock Tesseract Tamil — the baseline |
| `tamsyn` | trained on the regenerated 198,503-line corpus — **the headline model** |
| `tam_new` | an earlier model trained on the superseded corpus; do not report |

The system `tam_new.traineddata` is byte-identical to
`model/2026-08-21/tam_new.v1-old-corpus.traineddata`. It is kept only so the
older model can still be scored; the headline number is `tamsyn`
(`model/2026-08-21/tam_new.traineddata`, md5 `19e26cd5…`, 6,036,818 bytes).
