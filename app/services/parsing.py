import io
import re
from pathlib import Path

from pypdf import PdfReader


class ParseError(Exception):
    """An error whose message is safe to show to the user."""


def clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", "")                  # NUL characters break Postgres inserts
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)     # join words split by a line-end hyphen
    text = re.sub(r"[ \t]+", " ", text)              # collapse extra spaces
    return text


def _extract_pdf(raw: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise ParseError("The PDF is password-protected")
        pages = [page.extract_text() or "" for page in reader.pages]
    except ParseError:
        raise
    except Exception:
        raise ParseError("Could not read the PDF; the file may be corrupted")

    text = clean_pdf_text("\n\n".join(pages))
    if not text.strip():
        raise ParseError("The PDF has no text layer (it may be a scan)")
    return text


def extract_text(filename: str, raw: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(raw)
    try:
        return raw.decode("utf-8-sig")  # utf-8-sig strips a BOM if present
    except UnicodeDecodeError:
        raise ParseError("The file must be UTF-8 encoded")