# Experiment Log — TamilSynth resource-and-pipeline paper

## Contribution (one sentence)

Fine-tuning Tesseract for a syllabic script is dominated by three silent
`tesstrain` defaults rather than by corpus design: with all three wrong the
fine-tuned model is 5.9 CER points *worse* than the stock model, and with all
three right the same corpus yields a 24% relative reduction in word error.

## The resource

| | |
|---|---|
| Released corpus | 218,125 rendered line images + exact transcriptions |
| Rebuild with hygiene filter | 218,082 lines (43 dropped, §hygiene) |
| Raw pool / after dedup | 225,450 / 218,366 (7,048 exact duplicates, 3.13%) |
| Typefaces | 29, round-robin, 7,521–7,522 lines each (0.01% spread) |
| Graphemes | 13,370,858; 317 distinct |
| Coverage | uyir 12/12, aytham 1/1, mey 18/18, uyirmey 196/216 |
| Sources | wikisource-ta 58.4%, stories 24.7%, theekkathir 10.9%, maattru 4.8%, wikinews-ta 1.3% |
| Pages skipped in rendering | 0 |

## Test set (Experiment 1)

300 hand-transcribed lines of real printed Tamil, drawn from 5,227 segmented
candidates over 259 captured pages. Strata: newsprint 80, books 80, forms 60,
signage 40, degraded 40. Sampled with a per-page round-robin so no page
dominates; a page-md5 guard prevents duplicate pages being double-weighted.
Split 1-in-3 stratified by source into a 98-line **selection split** (used to
choose checkpoints and DAWG variants) and a 202-line **reporting split**
(never inspected until the number is reported), seed 20260824.

Contamination check: 3 of 248 sampled word 4-grams appear in training.

## Experiment 2 — headline, held-out reporting split (202 lines)

| model | CER | ΔCER 95% CI | WER | ΔWER 95% CI |
|---|---:|---|---:|---|
| stock `tam` | 8.38% | — | 24.82% | — |
| ours, iter 1400, no dawg | 8.11% | [−1.31, +0.87] | 19.33% | [−8.09, −3.14] |
| ours, + stock dawg | 7.80% | [−1.59, +0.52] | 19.87% | [−7.08, −2.77] |
| **ours, + corpus dawg (release)** | **7.36%** | **[−2.05, +0.15]** | **18.79%** | **[−8.24, −3.82]** |
| ours, + union dawg | 7.79% | [−1.63, +0.54] | 19.42% | [−7.62, −3.18] |

Paired bootstrap over lines, 10,000 resamples. P(better than stock): CER
0.959, WER 1.000. **CER claim is "no worse than stock"; WER claim is a 24%
relative reduction.**

Files: `results/tamv3_dawg_*.json`, `results/stock.json`, `results/checkpoint_sweep.json`

## Experiment 3 — the three defaults

### 3a. LANG_TYPE=Indic (box generator + pass_through_recoder)

Full-test-set CER by iteration, no DAWG either side. Stock = 8.87%.

| iter | v2 (LANG_TYPE blank) | v3 (LANG_TYPE=Indic) |
|---:|---:|---:|
| 100 | 40.59% | — |
| 200 | 35.20% | 8.76% |
| 300 | 38.33% | — |
| 700 | — | **8.09%** |
| 1,400 | 21.08% | 8.40% |
| 2,700 | 15.39% | 8.48% |
| 5,300 | 12.35% | 8.65% |
| 18,400 | **12.01%** | — |
| 46,600/46,800 | 14.20% | 11.05% |
| 79,800 | 13.52% | — |

v2 never reaches stock at any point on its curve. Structural evidence:
v2 traineddata has `null_char=106`, stock and v3 both have `null_char=2`.
Without `--pass_through_recoder`, `--continue_from` resumes stock's weights
into a rebuilt code space.

Box-generator evidence: `generate_line_box.py` groups a mark with its
consonant only when canonical combining class ≠ 0. Every Tamil vowel sign is
class 0 (category Mc), so `லி` splits into `ல` + `ி`; only pulli U+0BCD
(class 9) stays attached. 78 box records for 57 grapheme clusters on a
sample line.

### 3b. Stopping rule

Training BCER falls monotonically to 0.797% at iteration 46,800 while real-page
CER turns around near 700. Exported model 12.23%; iteration 1,400 of the same
run 8.40%. Worst on the sparsest crop quartile: 19.19% → 25.05%.

Density quartiles (ref graphemes per unit aspect ratio), full test set:

| Q | stock | v2 | v3 @46,800 |
|---|---:|---:|---:|
| Q1 sparsest | 19.07% | 39.81% | 26.23% |
| Q2 | 3.81% | 6.21% | 4.84% |
| Q3 | 4.47% | 6.73% | 6.50% |
| Q4 densest | 11.18% | 15.10% | 15.51% |

### 3c. Missing dictionary

`combine_lang_model` reads `data/$MODEL/$MODEL.wordlist`, `.punc`, `.numbers`;
nothing creates them, so it emits no DAWGs. v3 was 3.1 MB against stock's
6.0 MB. Our unicharset is a strict superset of stock's with IDs preserved (99
units keep positions; 9 appended: – ‘ ’ ௌ — ஔ + # \). Rebuilding stock's
wordlist against our unicharset reproduces its DAWG byte-size exactly
(2,943,474), which verifies ID compatibility. Corpus wordlist 290,761 words
beats stock's 251,899 and beats their union.

## Experiment 4 — per-stratum (full test set, no dawg)

| stratum | n | stock | v2 | v3 @46,800 |
|---|---:|---:|---:|---:|
| books | 80 | 7.39% | 12.71% | 10.23% |
| degraded | 40 | 8.47% | 14.74% | 14.44% |
| forms | 60 | 15.95% | 24.66% | 17.69% |
| newsprint | 80 | 6.75% | 12.38% | 11.02% |
| signage | 40 | 5.13% | 7.85% | 7.12% |

## Experiment 5 — corpus hygiene

146 malformed words in 2,418,180 (0.006%) put a vowel sign or pulli where
Tamil permits no mark; 123 lines of 218,125 (0.056%) affected.
`unicharset_extractor` rejects them ("Invalid start of grapheme sequence").

Dominant cause: in Bamini/TAB/TSCII the glyph for `ர` sits at the codepoint
Unicode assigns to `ா`, so converted archives spell `ர ்` as `ா ்`
(`தலைவா்` for `தலைவர்`, `தோ்தல்` for `தேர்தல்`). 75 of 155 instances.

`repair_tamil` fixes 103 words and leaves 2,418,034 byte-identical — every
pattern it matches is illegal Tamil, so it cannot touch well-formed text.
43 words are unrecoverable (base consonant absent) and their lines are dropped.

Per-source malformed word rate: theekkathir 0.017%, wikinews 0.009%,
wikisource 0.006%, stories 0.003%, maattru 0.001%.

## Experiment 6 — error analysis

Top grapheme substitutions for the release model are available in
`results/tamv3_dawg_corpus.json` (`aggregate.substitutions`). The earlier
draft asserted ர/ற and ள/ல/ழ confusion; measure it rather than assert it.

## Normalisation (protocol, not an experiment)

Zero-width stripping moves grapheme CER from 20.76% to 4.83% on a development
sample. PSM 13 vs PSM 7: PSM 7 returned nothing for 30 of 300 crops (10.8% of
reference characters); stock CER moved 16.32% → 8.87% on the fix.

## Failed / abandoned

- v1, v2 corpora: fixed-width 12-word chunking. Every line fills its measure,
  so the model never saw a short final line; on the sparsest crop quartile it
  scored 39.8% against stock's 19.1%. Motivated the paragraph-aware layout.
- Attributing v2→v3 to corpus design. The v2 sweep shows v2's failure is a
  representation collapse, not a data problem. **Not separately measurable.**

## Open questions

- Does the paragraph-aware corpus contribute anything on top of the Indic
  settings? Needs a v3-corpus / blank-LANG_TYPE run (one ~8h job). Not run.
- Ablations on font count and corpus scale (planned Experiments 3–5 in the
  original outline) are not run.
