# Latest model: tam_hyb (2026-11-23)

Use: `tesseract line.png out -l tam_hyb --psm 13` (copy the file into your tessdata directory).

Tesseract 5.5.0 `tam` fine-tuned with the corrected settings (LANG_TYPE=Indic,
pass-through recoder) on the 218,125 TamilSynth lines plus the 445 real lines in
`hybrid/` (seed 1, iteration 5,700, chosen on the 98-line selection split only),
packaged with the corpus dictionary by `build_dawgs.py`. Same file as
`model/2026-11-23/tam_hyb.traineddata`.

Held-out real lines (grapheme CER / WER, %; `tamil_ocr_eval.py`):

| Model | all 300 | reporting 202 | reporting 202, text only |
|---|---|---|---|
| stock `tam` | 8.87 / 24.57 | 8.38 / 24.82 | 5.73 / 15.70 |
| `tam_v3` (paper's released model) | 7.77 / 19.57 | 7.36 / 18.79 | 6.04 / 13.97 |
| `tam_hyb` | 7.25 / 18.48 | 7.01 / 17.81 | 6.03 / 14.16 |

Against stock on the reporting split: CER -1.37 (better in 9,850 of 10,000 line
resamples), WER -7.01 (10,000). Against `tam_v3`: CER -0.36 (8,856), WER -0.99
(8,087), not resolved, and level once punctuation is ignored. Better on forms
and newsprint, worse on signage. The real lines are in-domain for the test set
(same kinds of sources, same scanner), so this is an in-domain gain.
Predictions: `results/revision/hybrid_preds/tam_hyb_s1_i5700_dict/`.

The paper's released model remains `tam_v3` (`model/2026-08-24/`).
