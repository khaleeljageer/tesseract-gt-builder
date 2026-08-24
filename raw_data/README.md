# Source texts

These five files are third-party content, redistributed under their own
terms. The GPL-3.0 licence at the root of this repository covers the
pipeline code; it does not relicense anything in this directory.

| File | Source | Licence | Retrieved |
|---|---|---|---|
| `wikisource-ta.txt` | Tamil Wikisource — https://ta.wikisource.org | CC BY-SA 4.0 | *(fill in)* |
| `stories.txt` | `aitamilnadu/tamil_stories` — https://huggingface.co/datasets/aitamilnadu/tamil_stories | Apache-2.0 | *(fill in)* |
| `theekkathir_content_tamil_only.txt` | Theekkathir — https://theekkathir.in | CC BY-SA 4.0 | *(fill in)* |
| `maattru.txt` | Maattru — https://maattru.com | CC BY-SA 4.0 | *(fill in)* |
| `wikinews-ta.txt` | Tamil Wikinews — https://ta.wikinews.org | CC BY-SA 4.0 | *(fill in)* |

Wikisource and Wikinews are continuously edited, so the retrieval date is
part of the provenance: without it the input to `regenerate_corpus.py` cannot
be reconstructed. Fill the column before publishing a release.

Attribution for the CC BY-SA sources is the project name and URL above; the
Apache-2.0 collection's NOTICE and licence text are preserved alongside it.
All five are share-alike or permissive and mutually compatible, so the
derived corpus is released under CC BY-SA 4.0.

## Text hygiene

The sources carry a small number of malformed Tamil sequences inherited from
pre-Unicode encodings — 146 words in 2,418,180. `experiments/corpus.py`
repairs 103 of them (`repair_tamil`) and drops the lines containing the
other 43 (`well_formed`). See the README's "Corpus hygiene" section.
