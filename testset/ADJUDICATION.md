# Double-entry adjudication

Raw disagreement between the two passes, and what it was.

## Rule characters are transcribed

`forms_020_l022` accounted for 30 of the 44 line-level edits on its own. Both
passes read the Tamil identically; they differed on the signature rule that
follows it, a run of 29 printed hyphens. Pass 1 transcribed them; pass 2 read
them as typographic furniture and omitted them.

**Ruling: transcribe them.** This follows the principle already applied to
corpus hygiene — the label must describe its image. A reference that silently
drops marks the crop visibly contains no longer describes what the model is
shown. The ruling is against our own interest, which is the useful test of
it: both models emit a rule run on that line, ours a malformed one
(`--------------டடடடடடடடடடடடடடட`), so omitting the run from the reference
would have penalised stock and ours alike while flattering neither.

It is one line in 300. No other test line contains a run of four or more
repeated rule characters.

## Everything else

| | edits | of 1,858 graphemes |
|---|---:|---:|
| the rule run above | 29 | 1.56% |
| other punctuation and spacing | 9 | 0.48% |
| **Tamil graphemes** | **11** | **0.59%** |

The eleven Tamil disagreements are worth naming, because they are not random:

    ப் dropped      ட் → ட       ம dropped      க → க்
    வ dropped       லு → ளு      ணி → னி

Two of the seven are ல/ள and ண/ன — members of exactly the confusable
consonant families that §9 finds the *models* almost never confuse (2–4% of
their substitutions). The families are a human reading difficulty rather than
a machine one, at least at this print quality. Three more are the pulli,
which is the mark the models also lose most often.

## Sampling, and the correction

The first batch (`gt_pass2/`, 60 lines) was drawn with the same seed the
reporting/selection split uses, and the same per-stratum shuffle, so taking
the first 20% drew entirely from the first third the selection split takes:
60 of 60 lines in the selection split, none in the reporting split. The batch
is still a stratified uniform sample of all 300, so its estimate is unbiased
for the transcription process, but it could not be described as measured
where the headline numbers are.

`gt_pass2b/` corrects this: 40 lines, 20% of the reporting split, stratified,
under an independent seed, verified disjoint from the selection split.

## Result

| batch | lines | grapheme | word | exact |
|---|---:|---:|---:|---:|
| selection split | 60 | 0.75% | 3.63% | 48/60 |
| reporting split | 40 | 1.48% | 5.09% | 32/40 |
| **combined** | **100** | **1.04%** | **4.20%** | **80/100** |
| less one two-line crop | 98 | 0.77% | 3.94% | |

A third of the test set, two independently drawn batches, agreeing to three
hundredths of a point once one defective crop is set aside.

## Three crops span two printed lines

`books_064_l004`, `books_072_l006` (reporting split) and
`newsprint_050_l011` (selection split) each contain two printed lines rather
than one — a segmentation defect the sampling QC did not catch. One of them
is the whole difference between the two batches above.

They are disclosed rather than removed. Both systems receive identical input,
and excluding them moves stock and the released model almost equally (8.38%
to 7.32% and 7.36% to 6.24% grapheme CER), leaving the gap at -1.08 rather
than -1.01. Removing test items after seeing results is worth avoiding when
the result does not depend on it.
