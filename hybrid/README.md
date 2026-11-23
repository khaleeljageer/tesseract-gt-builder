# Hybrid real-plus-synthetic training lines (IJDAR revision 3, R2.7)

448 real printed-Tamil line images with hand-corrected transcriptions, used
only for TRAINING in the hybrid run. None comes from a test-set page, and none of its sources (four books, two journals,
three newspaper issues) was a test-set source (author's confirmation, 2026-11-23); no line
shares a run of four words with any test line (one common three-word phrase,
"இந்த ஊதிய உயர்வு", occurs in different sentences).

    raw/<source>/      page images, renamed (not published: full copyrighted pages,
                       as for the test set; raw_pages.csv keeps name, md5, issue)
    pool/              every segmented candidate line (not published)
    set/images, gt     the 448 drawn lines and their transcriptions
    set/gt_prefill     stock-tam pre-fill (zero-width chars removed), for comparison
    set/manifest.csv   provenance per line; reserve.csv replacements

Sources: 4 books (52 lines each), 2 journals (40 each), 3 newspaper issues
(news1 Tamil Hindu 2026-10-16: 50; news2 Theekkathir 2026-06-08: 55;
news3 Theekkathir 2026-06-10: 55). Short newspaper ids because
make_testset.py truncates line ids to 28 characters.

Draw: make_testset.py sample, seed 0, round-robin over pages, ~13 lines/page
(books), 10 or 8 (journals), 11-13 per clip (news). One crop replaced by
reserve (REJECTED.md). Pre-filled with stock tam, every line read and
corrected by the author against its image (one pass, no adjudication).

Conventions (same as the test set): type what is printed; superscript
footnote markers as plain digits; one space between words. Mechanical
normalisation after transcription: curly double quotes “ ” -> straight ",
matching the test set (which has only straight quotes). No other edits.

## Training and results (2026-11-23)

445 of the 448 lines were used: three contain Latin letters outside the
corpus unicharset and were excluded (book2_p03_l023, news2_c01_l012,
news2_c04_l005). Runs: `experiments/run_hybrid.sh`, `experiments/hybrid_queue.sh`
(Tesseract 5.5.0, desktop).

- Hybrid: each corrected synthetic-only run (tam_x9 = seed 0 rerun, tam_v3_s1,
  tam_v3_s2) repeated with its own list.train, unicharset and make call, the
  445 lines inserted 25 times each at random positions (5.4% of the stream).
- Real only: stock tam on the 445 lines alone, 10,000 iterations, list
  shuffled with tesstrain's shuffle.py and the seed (seeds 0, 1, 2).

Selected on sel98, reported on rep202 (`experiments/hybrid_bootstrap.py` ->
`results/revision/hybrid_bootstrap.json`; predictions in
`results/revision/hybrid_preds/`), pooled over three seeds, CER / WER:
synthetic only 8.39 / 19.42; hybrid 7.48 / 18.50; real only 6.84 / 18.85;
released model 7.36 / 18.79; stock 8.38 / 24.82. Online Resource Table F16.
