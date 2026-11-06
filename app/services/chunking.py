import re

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _split_long(paragraph: str, limit: int) -> list[str]:
    """Режет длинный абзац по предложениям, а слишком длинные предложения по словам."""
    parts: list[str] = []
    for sentence in _SENTENCE_END.split(paragraph):
        sentence = sentence.strip()
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            if cut <= 0:  # в отрезке нет пробелов: режем жёстко
                cut = limit
            parts.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            parts.append(sentence)
    return parts


def chunk_text(text: str, max_chars: int = 500, overlap: int = 80) -> list[str]:
    """Режет текст на куски не длиннее max_chars.

    Границы проходят по абзацам и предложениям, а не посреди слова.
    Каждый следующий кусок начинается с хвоста предыдущего (overlap),
    чтобы мысль на стыке не потерялась.
    """
    if overlap < 0 or overlap >= max_chars // 2:
        raise ValueError("overlap должен быть в диапазоне [0, max_chars // 2)")

    # место под перекрытие и пробел резервируем заранее,
    # чтобы итоговый кусок гарантированно влез в max_chars
    limit = max_chars - overlap - 1 if overlap else max_chars

    # 1. абзацы, внутри которых лишние пробелы и переводы строк схлопнуты
    paragraphs = [
        re.sub(r"\s+", " ", p).strip()
        for p in _PARAGRAPH_BREAK.split(text.replace("\r\n", "\n"))
    ]
    pieces: list[str] = []
    for p in paragraphs:
        if not p:
            continue
        pieces.extend([p] if len(p) <= limit else _split_long(p, limit))

    # 2. упаковываем мелкие части в куски, пока влезают
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= limit:
            current += " " + piece
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)

    # 3. перекрытие: к каждому куску спереди добавляем хвост предыдущего
    if overlap and len(chunks) > 1:
        result = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            tail = prev[-overlap:]
            if len(prev) > overlap and prev[-overlap - 1] != " ":
                # хвост начинается посреди слова: отбрасываем обрывок
                space = tail.find(" ")
                tail = tail[space + 1:] if space != -1 else ""
            result.append((tail + " " + cur).strip())
        chunks = result

    return chunks