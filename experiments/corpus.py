"""Provenance-preserving corpus construction for the ablation studies.

normalize-gt.py concatenates all five sources into one stream, chunks it, and
deletes the intermediate. That is fine for producing a single training corpus
but destroys two things the ablations need:

  * per-source provenance, without which the domain experiment cannot be run;
  * a stable line ordering, without which two variants are not comparable.

This module rebuilds the line pool per source, deterministically, and never
deletes an input.

Note also that normalize-gt.py's docstring says 7 words per line while the
code uses 12. The corpus on Zenodo was built at 12, so WORDS_PER_LINE is 12
here and the docstring discrepancy should be fixed in the paper's Methods
section rather than silently reconciled.
"""

import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

WORDS_PER_LINE = 12

# A typeset paragraph ends with a line that is not full. Chunking one global
# word stream into equal 12-word pieces produces a corpus in which every line
# fills its measure, and a model trained on it has never seen a heading, a
# form label, or the last line of a paragraph. Evaluated on real documents it
# invents text to fill the empty part of the crop: on the quarter of the test
# set whose lines occupy least of their crop it scored 39.5% CER against
# stock Tesseract's 19.3%, while on well-filled lines the two were within a
# point of each other.
#
# So lay text out the way a typesetter does. Sentences are grouped into
# paragraphs, each paragraph is broken into lines of at most WORDS_PER_LINE,
# and the remainder becomes a short final line. Paragraph length is sampled,
# which makes the length of that final line uniform over 1..WORDS_PER_LINE
# rather than fixed.
SENTENCE_END = "\u0BD0.!?\u2026"      # Tamil texts punctuate with . ! ? …
PARAGRAPH_SENTENCES = (2, 8)        # inclusive range, sampled per paragraph

# Register grouping used by the domain ablation (Experiment 5). Maattru is
# contemporary journalism -- politics, society, science, cinema -- so it sits
# with the newspaper sources rather than with Wikisource's classical texts
# and the fiction collection. It joined the newsprint arm when its corpus
# grew from 614 to 9,781 lines; below that it could not shift either arm.
REGISTERS = {
    "literary": ["wikisource-ta", "stories"],
    "newsprint": ["theekkathir_content_tamil_only", "wikinews-ta", "maattru"],
}


def normalize_line(text: str) -> str:
    """NFC and whitespace collapse. Matches the evaluator's profile."""
    return " ".join(unicodedata.normalize("NFC", text).split())


# Five of the six malformed-sequence classes below are artifacts of the
# 8-bit Tamil encodings (Bamini, TAB, TSCII) the source sites were typeset
# in before they moved to Unicode. The dominant one is r-with-pulli: in those
# fonts the glyph for the consonant ர carries the same codepoint as the vowel
# sign ா, so a naive converter emits "ா ்" wherever the text said "ர ்".
# It reads correctly to a human -- the glyphs are nearly identical -- which is
# why it survives proofreading, and it is why theekkathir has the highest rate
# of the five sources (0.017% of words against wikisource's 0.006%).
#
# Every one of these patterns is a mark in a position where Tamil permits no
# mark: a vowel sign or pulli that follows another vowel sign or pulli rather
# than a consonant. Well-formed text therefore cannot match any of them, and
# repair_tamil is a no-op on it -- verified over all 2,418,180 words of
# raw_data, where it repairs 103 words in 2,418,180 and leaves every other word
# byte-identical.
#
# Ordering matters: RA_PULLI must run before the doubled-mark collapse, so
# that "வாா்டு" (வ ா ா ்) becomes "வார்டு" and not "வா்டு".
TAMIL_REPAIRS = (
    ("\u0BBE\u0BCD", "\u0BB0\u0BCD"),   # ா ்  -> ர ்   legacy RA
    ("\u0BC1\u0BBE", "\u0BC2"),          # ு ா  -> ூ     legacy UU
    ("\u0BBE\u0BBE", "\u0BBE"),          # ா ா  -> ா     doubled sign
    ("\u0BCD\u0BCD", "\u0BCD"),          # ் ்  -> ்     doubled pulli
)

TAMIL_CONSONANTS = frozenset(chr(c) for c in range(0x0B95, 0x0BBA))
TAMIL_MARKS = frozenset(chr(c) for c in range(0x0BBE, 0x0BCE)) | {"\u0BD7"}


def repair_tamil(text: str) -> str:
    """Undo the legacy-encoding artifacts listed in TAMIL_REPAIRS.

    Works on NFD, because ொ ோ ௌ compose a vowel sign out of two marks and
    the artifact hides inside them: "தோ்தல்" is த + ே + ா + ் + ..., whose
    "ா ்" is invisible in NFC but is the same defect as in "தலைவா்".
    """
    s = unicodedata.normalize("NFD", text)
    for bad, good in TAMIL_REPAIRS:
        while bad in s:
            s = s.replace(bad, good)
    # A pulli after a vowel sign is never legal and, unlike the patterns
    # above, carries no recoverable intent -- it is a stray mark, as in
    # "வேண்டு்ம்" for "வேண்டும்". Drop it.
    out = []
    for ch in s:
        if ch == "\u0BCD" and out and out[-1] in TAMIL_MARKS:
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def well_formed(text: str) -> bool:
    """False if a Tamil mark still lacks a consonant to attach to.

    What repair_tamil cannot fix is a word whose base consonant is simply
    absent -- "்ளனர்", "ேலைநிறுத்தம்", "ாநாட்டின்" -- which is what a scraper
    produces when it breaks a word across a page boundary mid-grapheme, and
    the odd transposition ("விளைாயடுகிறாய்" for "விளையாடுகிறாய்"). Neither
    can be recovered without the missing character, so the line is dropped:
    43 words in 2,418,180.
    """
    prev = ""
    for ch in unicodedata.normalize("NFC", text):
        if ch in TAMIL_MARKS and prev not in TAMIL_CONSONANTS:
            return False
        prev = ch
    return True


def sentences(words):
    """Split a word stream at sentence-final punctuation."""
    out, current = [], []
    for word in words:
        current.append(word)
        if word and word[-1] in SENTENCE_END:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def paragraph_lines(words, words_per_line=WORDS_PER_LINE, rng=None,
                    para_range=PARAGRAPH_SENTENCES):
    """Lay a word stream out as paragraphs of full lines plus a short one.

    The short final line is the point: it is what a model needs to have seen
    to recognise a heading or the end of a paragraph without hallucinating
    across the blank remainder of the crop.
    """
    rng = rng or random.Random(0)
    lines = []
    pool = sentences(words)
    i = 0
    while i < len(pool):
        take = rng.randint(*para_range)
        para = [w for sentence in pool[i:i + take] for w in sentence]
        i += take
        if not para:
            continue
        for j in range(0, len(para), words_per_line):
            chunk = para[j:j + words_per_line]
            if chunk:
                lines.append(" ".join(chunk))
    return lines


def build_pool(raw_dir="raw_data", words_per_line=WORDS_PER_LINE, seed=0,
               paragraphs=True, repair=True):
    """Chunk every source into lines, keeping provenance.

    Returns {source_stem: [line, ...]}. Sources are read in sorted order and
    chunked independently, so a given source always yields the same lines in
    the same order regardless of what else is present. Each source gets its
    own seeded generator, for the same reason.

    With paragraphs=False the old fixed-width chunking is used, which is what
    produced the v1 and v2 corpora; it is kept so those remain reproducible.
    repair=False likewise reproduces v1-v3, which were built before
    repair_tamil existed.
    """
    pool = {}
    for path in sorted(Path(raw_dir).glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        if repair:
            words = [repair_tamil(w) for w in words]
        if paragraphs:
            rng = random.Random(f"{seed}:{path.stem}")
            lines = [normalize_line(x)
                     for x in paragraph_lines(words, words_per_line, rng)]
        else:
            lines = [
                normalize_line(" ".join(words[i:i + words_per_line]))
                for i in range(0, len(words), words_per_line)
            ]
            # Drop any trailing short chunk so every line has equal word count.
            if lines and len(lines[-1].split()) < words_per_line:
                lines.pop()
        pool[path.stem] = lines
    return pool


def deduplicate(lines, seen=None):
    """Exact line dedup, order-preserving.

    Repeated lines inflate corpus size without adding typographic or
    linguistic information, and across a corpus assembled from overlapping
    web sources there are real duplicates. Report the removal count in the
    paper's corpus statistics.
    """
    seen = seen if seen is not None else set()
    out = []
    for line in lines:
        h = hashlib.md5(line.encode("utf-8")).digest()
        if h in seen:
            continue
        seen.add(h)
        out.append(line)
    return out


def has_tamil(line):
    """True if the line contains at least one Tamil character.

    Fixed-width chunking occasionally emits a line made entirely of
    standalone punctuation, when the source text contains a run of such
    tokens (4 lines in 192,347). These are useless as OCR training data, and
    they also break page segmentation: with no full-height glyphs, the
    hyphens, colons and baseline marks land in separate horizontal bands, so
    the projection profile reports more bands than lines and the whole page
    is rejected. One such line therefore costs 50.
    """
    return any("஀" <= c <= "௿" for c in line)


def select(pool, sources=None, n_lines=None, seed=0, dedup=True,
           require_tamil=True, require_well_formed=True):
    """Build one variant's line list.

    sources  restrict to these source stems (None = all)
    n_lines  cap the result at this many lines (None = no cap)
    seed     controls the shuffle, so variants are reproducible

    Lines are pooled across the selected sources, deduplicated, shuffled with
    a fixed seed, then truncated. Shuffling before truncation matters: the
    sources are concatenated in size order, so taking a prefix without
    shuffling would sample almost entirely from Wikisource.
    """
    stems = sources if sources is not None else sorted(pool)
    missing = [s for s in stems if s not in pool]
    if missing:
        raise KeyError(f"sources not found in raw_data: {missing}")

    lines = []
    seen = set()
    for stem in stems:
        chunk = pool[stem]
        lines.extend(deduplicate(chunk, seen) if dedup else chunk)

    if require_tamil:
        lines = [ln for ln in lines if has_tamil(ln)]

    # Whatever repair_tamil could not mend has to go, because the renderer
    # will draw it and the ground truth will then describe an image of a
    # broken glyph cluster. unicharset_extractor rejects these outright
    # ("Invalid start of grapheme sequence"), so in v3 they contributed
    # nothing to the unicharset and stayed in the training set as noise:
    # 123 lines of 218,125.
    if require_well_formed:
        lines = [ln for ln in lines if well_formed(ln)]

    random.Random(seed).shuffle(lines)

    if n_lines is not None:
        if n_lines > len(lines):
            raise ValueError(
                f"requested {n_lines:,} lines but only {len(lines):,} available "
                f"from {stems}. Reduce n_lines or widen the source set."
            )
        lines = lines[:n_lines]
    return lines


def stats(lines):
    """Corpus statistics for the paper's Corpus Construction section."""
    # tamil_ocr_eval.graphemes() prefers open-tamil and falls back to an
    # equivalent combining-mark rule, so statistics are available whether or
    # not open-tamil is installed.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tamil_ocr_eval import graphemes as _graphemes
    graph = [g for ln in lines for g in _graphemes(ln)]

    words = [w for ln in lines for w in ln.split()]
    return {
        "lines": len(lines),
        "words": len(words),
        "unique_words": len(set(words)),
        "codepoints": sum(len(ln) for ln in lines),
        "unique_graphemes": len(set(graph)) if graph is not None else None,
        "top_graphemes": Counter(graph).most_common(20) if graph is not None else None,
    }


def write_corpus(lines, path):
    """Write one line per row, plus a sidecar with the statistics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sidecar = path.with_suffix(".stats.json")
    sidecar.write_text(
        json.dumps(stats(lines), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    pool = build_pool()
    print(f"{'source':<36} {'lines':>9}  {'after dedup':>11}")
    total = 0
    for stem, lines in sorted(pool.items(), key=lambda kv: -len(kv[1])):
        total += len(lines)
        print(f"{stem:<36} {len(lines):>9,}  {len(deduplicate(lines)):>11,}")
    print(f"{'TOTAL':<36} {total:>9,}")
    for name, stems in REGISTERS.items():
        n = sum(len(pool[s]) for s in stems if s in pool)
        print(f"  register {name:<10} {n:>9,} lines")
