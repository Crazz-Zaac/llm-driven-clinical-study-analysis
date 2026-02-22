"""
Text preprocessing pipeline for clinical research articles.

Cleans raw extracted text and prepares it for LLM-based structured extraction.

Steps:
  1. Remove boilerplate (headers, footers, form feeds)
  2. Remove references, figure/table captions
  3. Normalise whitespace and non-UTF characters
  4. Segment text into sections (Abstract, Methods, Results, etc.)
  5. Chunk long sections for context-limited LLMs
  6. (Optional) Biomedical entity linking via scispaCy
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import spacy
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Default chunk size in characters (~4 k chars ≈ ~1 k tokens).
# GPT-4-8k can handle ~6 k tokens of input comfortably; 4 k chars/chunk is safe.
DEFAULT_CHUNK_SIZE = 4000
DEFAULT_CHUNK_OVERLAP = 200

# Canonical section order – keys are normalised to lower-case for matching.
SECTION_HEADINGS = [
    "abstract",
    "introduction",
    "background",
    "methods",
    "materials and methods",
    "study design",
    "data source",
    "study population",
    "statistical analysis",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "acknowledgements",
    "acknowledgments",
    "references",
    "appendix",
    "supplementary",
]

# Regex that matches common section headings (numbered or plain).
_HEADING_RE = re.compile(
    r"^(?:\d+\.?\s*)?("
    + "|".join(re.escape(h) for h in SECTION_HEADINGS)
    + r")\b",
    re.IGNORECASE | re.MULTILINE,
)

# ---------------------------------------------------------------------------
# 1. Boilerplate / artefact removal
# ---------------------------------------------------------------------------

def _remove_boilerplate(text: str) -> str:
    """Strip recurring journal artefacts produced by PDF extraction."""
    # Form feeds
    text = text.replace("\f", "\n")
    # Repeated banners like "ARTICLE IN PRESS", "ACCEPTED MANUSCRIPT"
    text = re.sub(
        r"(?:ARTICLE\s+IN\s+PRESS|ACCEPTED\s+MANUSCRIPT)\s*\n?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Page numbers at start/end of lines
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)
    return text


# ---------------------------------------------------------------------------
# 2. Remove references, figure/table captions
# ---------------------------------------------------------------------------

def _remove_references_section(text: str) -> str:
    """Cut everything from the References heading onward."""
    match = re.search(
        r"(?m)^(?:\d+\.?\s*)?(?:references?|bibliography|works cited)\s*$",
        text,
        re.IGNORECASE,
    )
    if match:
        text = text[: match.start()]
    return text


def _remove_figure_table_captions(text: str) -> str:
    """Remove figure/table captions and their descriptions, plus abbreviation blocks."""
    # Single-line captions: "Figure 1.", "Table 2:", "Supplementary Table 1 ..."
    text = re.sub(
        r"(?m)^(?:Supplementary\s+)?(?:Figure|Fig\.?|Table)\s*\d+.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Multi-line captions that start with "Figure N." or "Table N." and continue
    # with indented / run-on description lines until a blank line
    text = re.sub(
        r"(?mi)^(?:Supplementary\s+)?(?:Figure|Fig\.?|Table)\s*\d+[\.\:\s].*?(?:\n(?!\n).*?)*(?=\n\n|\Z)",
        "",
        text,
    )
    # Abbreviation blocks often at the end: "Abbreviations:" or "Abbreviations\n"
    text = re.sub(
        r"(?mi)^Abbreviations?\s*[\:\n].*",
        "",
        text,
        flags=re.DOTALL,
    )
    return text


def _remove_inline_citations(text: str) -> str:
    """Remove bracketed numeric citations like (1), (2, 3), (14–18)."""
    text = re.sub(r"\([\d,;\s–\-]+\)", "", text)
    return text


# ---------------------------------------------------------------------------
# 3. Normalise whitespace & encoding
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """Normalise unicode, collapse whitespace, strip non-printable chars."""
    # Unicode NFC normalisation
    text = unicodedata.normalize("NFC", text)
    # Replace common unicode dashes / quotes with ASCII equivalents
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Remove control characters (keep newlines and tabs)
    text = re.sub(r"[^\S\n\t]+", " ", text)  # collapse horizontal whitespace
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


# ---------------------------------------------------------------------------
# 4. Section segmentation
# ---------------------------------------------------------------------------

def segment_sections(text: str) -> dict[str, str]:
    """
    Split text into named sections based on headings.

    Returns a dict mapping section name → section body text.
    If no headings are detected the entire text is returned under "full_text".
    """
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        return {"full_text": text.strip()}

    sections: dict[str, str] = {}

    # Text before the first heading
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections["preamble"] = preamble

    for i, m in enumerate(matches):
        name = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections[name] = body

    return sections


# ---------------------------------------------------------------------------
# 5. Chunking for LLMs
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping chunks that respect sentence boundaries.

    Each chunk is at most *max_chars* characters.  Overlap ensures context
    continuity between consecutive chunks.
    """
    if len(text) <= max_chars:
        return [text]

    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            # Keep last few sentences for overlap
            overlap_text = " ".join(current)
            overlap_sents: list[str] = []
            ol = 0
            for s in reversed(current):
                ol += len(s) + 1
                overlap_sents.insert(0, s)
                if ol >= overlap:
                    break
            current = overlap_sents
            current_len = sum(len(s) + 1 for s in current)
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_sections(
    sections: dict[str, str],
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, str]]:
    """
    Chunk every section independently, returning a flat list of dicts:
        [{"section": <name>, "chunk_index": <i>, "text": <chunk>}, ...]
    """
    result: list[dict[str, str]] = []
    for name, body in sections.items():
        for i, chunk in enumerate(chunk_text(body, max_chars, overlap)):
            result.append({"section": name, "chunk_index": i, "text": chunk})
    return result


# ---------------------------------------------------------------------------
# 6. (Optional) Biomedical entity linking via scispaCy
# ---------------------------------------------------------------------------

_SCI_NLP: Optional[spacy.language.Language] = None


def _load_scispacy():
    """Lazy-load the scispaCy biomedical model (en_core_sci_lg)."""
    global _SCI_NLP
    if _SCI_NLP is not None:
        return _SCI_NLP
    try:
        _SCI_NLP = spacy.load("en_core_sci_lg")
        logger.info("Loaded scispaCy model en_core_sci_lg")
    except OSError:
        logger.warning(
            "scispaCy model 'en_core_sci_lg' not installed — "
            "entity linking will be skipped. Install with:\n"
            "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
            "releases/v0.5.5/en_core_sci_lg-0.5.5.tar.gz"
        )
        _SCI_NLP = None
    return _SCI_NLP


def extract_biomedical_entities(text: str) -> list[dict[str, str]]:
    """
    Detect biomedical entities in *text* using scispaCy.

    Returns a list of dicts: [{"text": ..., "label": ..., "start": ..., "end": ...}]
    Returns an empty list if the model is not installed.
    """
    nlp = _load_scispacy()
    if nlp is None:
        return []

    doc = nlp(text[:100_000])  # cap input to avoid OOM on very long texts
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]


# ---------------------------------------------------------------------------
# Public API – full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    raw_text: str,
    *,
    remove_references: bool = True,
    remove_captions: bool = True,
    remove_citations: bool = True,
    chunk: bool = True,
    max_chunk_chars: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    extract_entities: bool = False,
) -> dict:
    """
    Run the full preprocessing pipeline on *raw_text*.

    Returns a dict with keys:
        cleaned_text    – cleaned full text (str)
        sections        – dict[section_name, section_text]
        chunks          – list of chunk dicts (if chunk=True)
        entities        – list of entity dicts (if extract_entities=True)
    """
    text = raw_text

    # Step 1 – boilerplate
    text = _remove_boilerplate(text)

    # Step 2 – references, captions, citations
    if remove_references:
        text = _remove_references_section(text)
    if remove_captions:
        text = _remove_figure_table_captions(text)
    if remove_citations:
        text = _remove_inline_citations(text)

    # Step 3 – normalise
    text = _normalise_text(text)

    # Step 4 – section segmentation
    sections = segment_sections(text)

    # Step 5 – chunking
    chunks = []
    if chunk:
        chunks = chunk_sections(
            sections, max_chars=max_chunk_chars, overlap=chunk_overlap
        )

    # Step 6 – entity linking (optional)
    entities = []
    if extract_entities:
        entities = extract_biomedical_entities(text)

    return {
        "cleaned_text": text,
        "sections": sections,
        "chunks": chunks,
        "entities": entities,
    }
