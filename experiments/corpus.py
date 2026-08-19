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

# Register grouping used by the domain ablation (Experiment 5). "maattru" is
# excluded from both arms: at 614 lines it cannot meaningfully shift either,
# and it is not cleanly one register or the other.
REGISTERS = {
    "literary": ["wikisource-ta", "stories"],
    "newsprint": ["theekkathir_content_tamil_only", "wikinews-ta"],
}


def normalize_line(text: str) -> str:
    """NFC and whitespace collapse. Matches the evaluator's profile."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def build_pool(raw_dir="raw_data", words_per_line=WORDS_PER_LINE):
    """Chunk every source into fixed-width lines, keeping provenance.

    Returns {source_stem: [line, ...]}. Sources are read in sorted order and
    chunked independently, so a given source always yields the same lines in
    the same order regardless of what else is present.
    """
    pool = {}
    for path in sorted(Path(raw_dir).glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
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
           require_tamil=True):
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
