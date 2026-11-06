import io
import re
from pathlib import Path

from pypdf import PdfReader


class ParseError(Exception):
    """Ошибка, текст которой можно показать пользователю."""


def clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", "")                      # NUL-символы ломают запись в Postgres
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)         # склеиваем слова, разорванные переносом
    text = re.sub(r"[ \t]+", " ", text)                  # лишние пробелы
    return text


def _extract_pdf(raw: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise ParseError("PDF защищён паролем")
        pages = [page.extract_text() or "" for page in reader.pages]
    except ParseError:
        raise
    except Exception:
        raise ParseError("Не удалось прочитать PDF, файл повреждён?")

    text = clean_pdf_text("\n\n".join(pages))
    if not text.strip():
        raise ParseError("В PDF нет текстового слоя (возможно, это скан)")
    return text


def extract_text(filename: str, raw: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(raw)
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ParseError("Файл должен быть в кодировке UTF-8")