from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict

import fitz  # PyMuPDF
from PIL import Image


@dataclass
class TextChunk:
    file_path: str
    page_number: int
    text: str
    kind: str  # "book" or "paper"
    subject: Optional[str] = None


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[\t\x0b\x0c]+", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: List[str] = []
    buffer: List[str] = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) >= chunk_size:
            # Hard wrap long paragraphs
            for i in range(0, len(paragraph), chunk_size - overlap):
                chunks.append(paragraph[i : i + chunk_size])
        else:
            buffer.append(paragraph)
            merged = "\n\n".join(buffer)
            if len(merged) >= chunk_size:
                chunks.append(merged[:chunk_size])
                # keep tail for overlap
                tail_start = max(0, len(merged) - overlap)
                buffer = [merged[tail_start:]]

    if buffer:
        chunks.append("\n\n".join(buffer))

    return [c.strip() for c in chunks if c.strip()]


def extract_text_chunks_from_pdf(pdf_path: str | Path, kind: str, subject: Optional[str] = None,
                                 chunk_size: int = 800, overlap: int = 100) -> List[TextChunk]:
    pdf_path = str(pdf_path)
    chunks: List[TextChunk] = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_index in range(len(doc)):
                page = doc[page_index]
                text = page.get_text("text") or ""
                text = normalize_whitespace(text)
                for chunk_text in split_text_into_chunks(text, chunk_size, overlap):
                    chunks.append(TextChunk(
                        file_path=pdf_path,
                        page_number=page_index + 1,
                        text=chunk_text,
                        kind=kind,
                        subject=subject
                    ))
    except Exception as exc:
        # Return what we have; caller can log
        pass
    return chunks


REFERENCE_HEADINGS = [
    r"^references\b",
    r"^bibliography\b",
    r"^works cited\b",
]

STOP_HEADINGS = [
    r"^appendix\b",
    r"^acknowledg(e)?ments?\b",
    r"^about the author\b",
]


def extract_references_from_pdf(pdf_path: str | Path) -> List[str]:
    references: List[str] = []
    try:
        with fitz.open(str(pdf_path)) as doc:
            full_pages_text: List[str] = []
            for page_index in range(len(doc)):
                text = doc[page_index].get_text("text") or ""
                full_pages_text.append(text)
            full_text = "\n\n".join(full_pages_text)
            lines = [normalize_whitespace(l) for l in full_text.split("\n")]

            # Find the start of references section by heading
            start_idx = None
            for idx, line in enumerate(lines):
                line_lower = line.lower().strip()
                if any(re.match(p, line_lower) for p in REFERENCE_HEADINGS):
                    start_idx = idx + 1
                    break

            if start_idx is None:
                return []

            # Collect lines until next heading-like stop
            for i in range(start_idx, len(lines)):
                line_lower = lines[i].lower().strip()
                if not lines[i].strip():
                    continue
                if any(re.match(p, line_lower) for p in STOP_HEADINGS):
                    break
                references.append(lines[i].strip())

            # Prune very short lines
            references = [r for r in references if len(r) > 5]
    except Exception:
        return []

    return references


def pdf_first_page_to_pil_image(pdf_path: str | Path, zoom: float = 2.0) -> Optional[Image.Image]:
    try:
        with fitz.open(str(pdf_path)) as doc:
            if len(doc) == 0:
                return None
            page = doc[0]
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            mode = "RGB" if pix.n < 4 else "RGBA"
            image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            if mode == "RGBA":
                image = image.convert("RGB")
            return image
    except Exception:
        return None


def image_file_to_pil(uploaded_bytes: bytes) -> Optional[Image.Image]:
    try:
        return Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    except Exception:
        return None