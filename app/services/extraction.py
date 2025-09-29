import io
import os
from typing import Optional

from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _extract_from_pdf(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as buffer:
        return pdf_extract_text(buffer) or ""


def _extract_from_docx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as buffer:
        document = Document(buffer)
        paragraphs = [p.text for p in document.paragraphs if p.text]
        return "\n".join(paragraphs)


def _extract_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Fallback to latin-1 to avoid errors for simple text
        return file_bytes.decode("latin-1")


def extract_text_from_file(file_bytes: bytes, filename: Optional[str]) -> str:
    if not filename:
        raise ValueError("Filename is required to determine file type")

    _, ext = os.path.splitext(filename.lower())
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    if ext == ".pdf":
        return _extract_from_pdf(file_bytes)
    if ext == ".docx":
        return _extract_from_docx(file_bytes)
    if ext == ".txt":
        return _extract_from_txt(file_bytes)

    # Defensive default
    raise ValueError(f"Unsupported file type: {ext}")

