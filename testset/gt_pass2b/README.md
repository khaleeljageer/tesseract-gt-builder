# Second transcription pass — reporting split

Transcribe each `.tif` in this directory into the `.gt.txt` beside it, then:

    python3 make_testset.py agreement --pass1 testset/gt --pass2 testset/gt_pass2

It prints the grapheme disagreement rate, which is the transcription error
floor: the level below which a difference between two models is
indistinguishable from noise in the ground truth.

## Rules

**Do not look at `testset/gt/`.** The whole value of this pass is that it is
independent. Reading the first pass first measures nothing.

**Do not pre-fill with the model under evaluation.** You would be scoring the
model against its own output wherever you failed to spot an error. Stock
`tam` is acceptable as a starting point; a different engine is better; from
the image alone is best.

Transcribe what is printed, not what should have been printed. If the source
has a typo, keep it. Same normalisation as the first pass: Unicode NFC, no
zero-width characters, single spaces.

## What this is and is not

This is intra-annotator agreement -- the same reader, twice, blind -- not
inter-annotator agreement. It bounds transcription noise, which is what the
paper needs, but it does not measure whether two readers would agree on a
convention. Report it as the former; claiming the latter would be wrong.
