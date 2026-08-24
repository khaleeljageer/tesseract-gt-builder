# Source texts

These five files are third-party content, redistributed under their own
terms. The GPL-3.0 licence at the root of this repository covers the
pipeline code; it does not relicense anything in this directory.

| File | Source | Licence | Retrieved |
|---|---|---|---|
| `wikisource-ta.txt` | Tamil Wikisource — https://ta.wikisource.org | CC BY-SA 4.0 | ~2019 |
| `stories.txt` | `aitamilnadu/tamil_stories` — https://huggingface.co/datasets/aitamilnadu/tamil_stories | Apache-2.0 | ~2019 |
| `theekkathir_content_tamil_only.txt` | Theekkathir — https://theekkathir.in | CC BY-SA 4.0 | ~2019 |
| `maattru.txt` | Maattru — https://maattru.com | CC BY-SA 4.0 | ~2019 |
| `wikinews-ta.txt` | Tamil Wikinews — https://ta.wikinews.org | CC BY-SA 4.0 | ~2019 |

Retrieved in approximately 2019 and not re-fetched since; the exact dates
were not logged, and the year is recorded rather than a date invented for it.

Wikisource and Wikinews are continuously edited, so fetching them today will
not reproduce these files. That does not break reproducibility of the corpus:
the stored texts in this directory are the pipeline's input and are released
with it, so `regenerate_corpus.py --seed 0` reproduces the corpus from them
exactly. What a year-level record does not support is re-deriving these files
from the live sites; for the two Wikimedia sources, the dump archive at
https://dumps.wikimedia.org/ keeps dated snapshots that come close.

Attribution for the CC BY-SA sources is the project name and URL above; the
Apache-2.0 collection's NOTICE and licence text are preserved alongside it.
All five are share-alike or permissive and mutually compatible, so the
derived corpus is released under CC BY-SA 4.0.

## Text hygiene

The sources carry a small number of malformed Tamil sequences inherited from
pre-Unicode encodings — 146 words in 2,418,180. `experiments/corpus.py`
repairs 103 of them (`repair_tamil`) and drops the lines containing the
other 43 (`well_formed`). See the README's "Corpus hygiene" section.
