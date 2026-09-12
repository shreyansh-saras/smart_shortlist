"""
Improved section chunking for the Smart Shortlisting Engine.

Key improvements over the naive version:
1. Header detection uses actual PDF font metadata (size / boldness) as the
   primary signal, not `line.startswith("skills")`. This is what fixes the
   core bug: a body sentence like "Skills-based hiring was a focus of this
   internship" or "Experience with distributed systems" would previously
   match as a header via startswith() and silently reset current_sec,
   corrupting every line that follows until the next false match.
2. Lines are sorted into reading order by (page, y-position, x-position)
   instead of relying on PyMuPDF's raw block order, which does not reliably
   match visual reading order for two-column resume layouts.
3. Fuzzy matching (difflib) catches typoed headers ("Eduction", "Skils",
   "Experiance") — this is exactly the messy-formatting bonus the problem
   statement calls out.
4. Header-like lines that don't map to the known taxonomy are tracked
   separately instead of being silently absorbed into whatever section was
   last active — useful both for debugging and for surfacing nonstandard
   resume structures.
5. Bullet/numbering prefixes are stripped before comparing header text.
6. NEW — chunk_sections_batch() parses multiple resumes in PARALLEL using
   a process pool. PDF parsing (PyMuPDF's page.get_text("dict") call) is
   CPU-bound, independent per resume, and releases the GIL at the C level,
   so it's a good multiprocessing candidate — unlike the embedding step in
   semantic_matcher.py, which should stay as ONE big batched call rather
   than being split across processes. For a run of ~18 resumes, this
   overlaps parsing work across CPU cores instead of doing it one file at
   a time.

Requires PyMuPDF: pip install pymupdf --break-system-packages
"""

import os
import re
import difflib
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


HEADER_SYNONYMS = {
    "experience": ["work history", "employment", "professional experience", "internships", "work experience", "experience"],
    "projects": ["personal projects", "academic projects", "key projects", "notable projects", "projects"],
    "skills": ["technical skills", "skills & tools", "technologies", "competencies", "core skills", "tech stack", "skills"],
    "education": ["academic background", "education & qualifications", "qualifications", "academics", "education"],
}

# Longest-synonym-first so "technical skills" is checked before the bare
# "skills" fallback — prevents short synonyms from winning collisions.
_ALL_SYNONYMS = sorted(
    ((syn, standard) for standard, syns in HEADER_SYNONYMS.items() for syn in syns),
    key=lambda pair: -len(pair[0])
)

_BULLET_PREFIX_RE = re.compile(r"^[\u2022\-\*\d\.\)\s]+")


def _clean_line(line: str) -> str:
    return _BULLET_PREFIX_RE.sub("", line).strip()


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]


def _looks_like_header(
    line: str,
    font_size: Optional[float] = None,
    is_bold: Optional[bool] = None,
    body_font_size: Optional[float] = None,
) -> bool:
    """
    Font size/boldness relative to the document's median body font is the
    strongest signal (headers are visually distinct in virtually every
    resume template). Falls back to casing heuristics only when font
    metadata isn't available.
    """
    words = line.split()
    if not (0 < len(words) <= 5):
        return False
    if line.endswith((".", ",", ";")):
        return False

    if font_size is not None and body_font_size:
        if font_size > body_font_size * 1.05:
            return True
    if is_bold:
        return True
    if font_size is not None:
        # Font metadata was available but didn't indicate a header —
        # don't fall through to casing heuristics, which are unreliable
        # once we already have better signal.
        return False

    # No font metadata available at all (e.g. plain-text input) — fall
    # back to casing as a weaker signal.
    return line.isupper() or line.istitle()


def _match_header(line: str, fuzzy_threshold: float = 0.82) -> Optional[str]:
    cleaned = _clean_line(line).lower()
    if not cleaned:
        return None

    for syn, standard in _ALL_SYNONYMS:
        if cleaned == syn:
            return standard

    best_ratio, best_standard = 0.0, None
    for syn, standard in _ALL_SYNONYMS:
        ratio = difflib.SequenceMatcher(None, cleaned, syn).ratio()
        if ratio > best_ratio:
            best_ratio, best_standard = ratio, standard
    return best_standard if best_ratio >= fuzzy_threshold else None


def extract_lines_with_layout(pdf_source) -> List[dict]:
    """
    Extract line-level text with font metadata, sorted into reading order.
    Rounding y0 to the nearest 10 units before sorting groups lines that
    sit on the same visual row (handles minor baseline jitter within a
    line) while still sorting top-to-bottom, then left-to-right — this is
    what makes simple two-column layouts parse in the right order.
    """
    if fitz is None:
        raise ImportError("PyMuPDF is required: pip install pymupdf --break-system-packages")

    doc = fitz.open(stream=pdf_source, filetype="pdf") if isinstance(pdf_source, bytes) else fitz.open(pdf_source)

    lines_info = []
    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                font_size = max(s["size"] for s in spans)
                is_bold = any("bold" in s["font"].lower() for s in spans)
                x0, y0 = line["bbox"][0], line["bbox"][1]
                lines_info.append({
                    "page": page_num, "text": text,
                    "font_size": font_size, "is_bold": is_bold,
                    "x0": x0, "y0": y0,
                })

    lines_info.sort(key=lambda l: (l["page"], round(l["y0"] / 10), l["x0"]))
    return lines_info


def chunk_sections(pdf_source) -> Dict[str, str]:
    """
    Returns a dict mapping standard section names -> concatenated text.
    Includes "_unclassified_headers" if header-like lines were found that
    didn't map to the known taxonomy (useful signal for messy/nonstandard
    resumes rather than a silent misclassification).
    """
    lines_info = extract_lines_with_layout(pdf_source)
    if not lines_info:
        return {}

    body_font_size = _median([l["font_size"] for l in lines_info])

    buckets: Dict[str, List[str]] = {k: [] for k in HEADER_SYNONYMS}
    buckets["general"] = []
    unclassified_headers: List[str] = []

    current_sec = "general"
    for line in lines_info:
        text = line["text"]
        if _looks_like_header(text, line["font_size"], line["is_bold"], body_font_size):
            standard = _match_header(text)
            if standard:
                # Recognised section header: switch to it
                current_sec = standard
            else:
                # Bold sub-header within a section (e.g. a project title like
                # "CivicConnect - Local Issue Reporting Platform"). Keep
                # current_sec active so the bullets that follow stay in the
                # correct bucket.  Only log for debugging.
                unclassified_headers.append(text)
                # Do NOT reset current_sec here — intentional
            continue
        buckets[current_sec].append(text)

    result = {k: " ".join(v) for k, v in buckets.items() if v}
    if unclassified_headers:
        result["_unclassified_headers"] = unclassified_headers
    return result



def extract_text_from_pdf(pdf_source) -> str:
    """Extracts reading-order plain text from PDF source."""
    lines_info = extract_lines_with_layout(pdf_source)
    return "\n".join(l["text"] for l in lines_info)


def parse_resume_content(pdf_source) -> Tuple[str, Dict[str, str]]:
    """Single-pass extraction of both raw text and classified sections."""
    lines_info = extract_lines_with_layout(pdf_source)
    if not lines_info:
        return "", {}

    raw_text = "\n".join(l["text"] for l in lines_info)
    body_font_size = _median([l["font_size"] for l in lines_info])

    buckets: Dict[str, List[str]] = {k: [] for k in HEADER_SYNONYMS}
    buckets["general"] = []
    unclassified_headers: List[str] = []

    current_sec = "general"
    for line in lines_info:
        text = line["text"]
        if _looks_like_header(text, line["font_size"], line["is_bold"], body_font_size):
            standard = _match_header(text)
            if standard:
                current_sec = standard
            else:
                unclassified_headers.append(text)
                # Keep current_sec active so section bullets stay in the current bucket
            continue
        buckets[current_sec].append(text)

    sections = {k: " ".join(v) for k, v in buckets.items() if v}
    if unclassified_headers:
        sections["_unclassified_headers"] = unclassified_headers
    return raw_text, sections


# ---------------------------------------------------------------------------
# Parallel batch parsing
# ---------------------------------------------------------------------------
#
# chunk_sections() itself stays synchronous and unchanged above — this is a
# separate entry point for the common case of parsing an entire resume pool
# at once. Each resume's PDF parsing is independent of every other resume's,
# so we hand them to a process pool instead of parsing one file at a time.
#
# Must be a MODULE-LEVEL function (not a closure or method) so it can be
# pickled and sent to worker processes.

def _chunk_single_for_pool(item: Tuple[str, Union[bytes, str]]) -> Tuple[str, Dict[str, object]]:
    name, pdf_source = item
    try:
        return name, chunk_sections(pdf_source)
    except Exception as e:
        # Surface the failure per-resume instead of crashing the whole
        # batch — one malformed PDF shouldn't take down the other 17.
        return name, {"_error": str(e)}


def chunk_sections_batch(
    pdf_sources: Dict[str, Union[bytes, str]],
    max_workers: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Parse multiple resumes' sections in parallel.

    Args:
        pdf_sources: {candidate_name: pdf_bytes_or_path}. Passing file
                     paths (str) rather than bytes avoids pickling large
                     byte blobs across the process boundary when the PDFs
                     already live on disk.
        max_workers: defaults to os.cpu_count() if not given. For a small
                     batch (e.g. under ~4 resumes) the process-startup
                     overhead can outweigh the parallelism gain — in that
                     case chunk_sections() in a plain loop is fine too.

    Returns:
        {candidate_name: chunk_sections() result}. A resume whose parsing
        raised an exception gets {"_error": "<message>"} instead of
        crashing the batch, so callers can flag it and continue with the
        rest of the pool.
    """
    if not pdf_sources:
        return {}

    workers = max_workers or os.cpu_count() or 1
    # No point spinning up more workers than resumes to parse.
    workers = min(workers, len(pdf_sources))

    results: Dict[str, Dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for name, sections in executor.map(_chunk_single_for_pool, pdf_sources.items()):
            results[name] = sections
    return results


# ---------------------------------------------------------------------------
# Pure-logic self-test (no PDF/fitz dependency) — validates header matching,
# fuzzy typo tolerance, and reading-order sort using mocked line data.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Header matching should not fire on body sentences containing skill words
    assert _match_header("Skills-based hiring was a focus") is None
    assert _match_header("Experience with distributed systems") is None

    # 2. Exact and fuzzy header matches
    assert _match_header("Technical Skills") == "skills"
    assert _match_header("Eduction") == "education"      # typo tolerance
    assert _match_header("• Work Experience") == "experience"

    # 3. Reading-order sort for a mocked two-column layout
    mock_lines = [
        {"page": 0, "text": "Right col line 1", "font_size": 10, "is_bold": False, "x0": 300, "y0": 100},
        {"page": 0, "text": "Left col line 1", "font_size": 10, "is_bold": False, "x0": 50, "y0": 100},
        {"page": 0, "text": "Left col line 2", "font_size": 10, "is_bold": False, "x0": 50, "y0": 130},
    ]
    mock_lines.sort(key=lambda l: (l["page"], round(l["y0"] / 10), l["x0"]))
    assert [l["text"] for l in mock_lines] == ["Left col line 1", "Right col line 1", "Left col line 2"]

    # 4. Font-size-based header detection (body ~10pt, header ~14pt bold)
    assert _looks_like_header("Projects", font_size=14, is_bold=True, body_font_size=10) is True
    assert _looks_like_header("Built a REST API in college", font_size=10, is_bold=False, body_font_size=10) is False

    # 5. Parallel batch parsing: patch chunk_sections (module-level, so the
    #    fork-based worker processes inherit the patched version) with a
    #    fake that doesn't need a real PDF or fitz, then confirm every
    #    resume gets parsed and results are keyed by the right name.
    import sys as _sys

    def _fake_chunk_sections(pdf_source):
        # pdf_source here will just be a plain string standing in for a path
        return {"skills": f"parsed:{pdf_source}"}

    _this_module = _sys.modules[__name__]
    _original_chunk_sections = _this_module.chunk_sections
    _this_module.chunk_sections = _fake_chunk_sections
    try:
        sources = {f"candidate_{i}": f"resume_{i}.pdf" for i in range(5)}
        batch_result = chunk_sections_batch(sources, max_workers=3)
        assert set(batch_result.keys()) == set(sources.keys())
        for name, path in sources.items():
            assert batch_result[name] == {"skills": f"parsed:{path}"}, (
                f"Unexpected result for {name}: {batch_result[name]}"
            )

        # 6. A single failing resume shouldn't take down the batch.
        def _fake_chunk_sections_with_failure(pdf_source):
            if pdf_source == "resume_2.pdf":
                raise ValueError("corrupted PDF")
            return {"skills": f"parsed:{pdf_source}"}

        _this_module.chunk_sections = _fake_chunk_sections_with_failure
        batch_result2 = chunk_sections_batch(sources, max_workers=3)
        assert "_error" in batch_result2["candidate_2"]
        assert batch_result2["candidate_0"] == {"skills": "parsed:resume_0.pdf"}
    finally:
        _this_module.chunk_sections = _original_chunk_sections

    print("All pure-logic self-tests passed.")