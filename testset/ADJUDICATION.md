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

## Sampling

The first pass-2 batch (`gt_pass2/`, 60 lines) was drawn with the same seed
the reporting/selection split uses, and the same per-stratum shuffle, so
taking the first 20% drew entirely from the first third the selection split
takes: all 60 lines are in the selection split and none in the reporting
split. The batch is still a stratified uniform random sample of the 300, so
the floor it measures is an unbiased estimate of the transcription process,
which is what the paper needs — but it cannot be described as measured on
the lines the headline numbers come from.

`gt_pass2b/` corrects this: 40 lines, 20% of the reporting split, stratified,
under an independent seed, disjoint from the selection split.
