import re

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _split_long(paragraph: str, limit: int) -> list[str]:
    """Split a long paragraph by sentences, and overlong sentences by words."""
    parts: list[str] = []
    for sentence in _SENTENCE_END.split(paragraph):
        sentence = sentence.strip()
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            if cut <= 0:  # no spaces in the slice: hard cut
                cut = limit
            parts.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            parts.append(sentence)
    return parts


def chunk_text(text: str, max_chars: int = 500, overlap: int = 80) -> list[str]:
    """Split text into chunks of at most max_chars characters.

    Boundaries follow paragraphs and sentences rather than cutting mid-word.
    Each chunk starts with the tail of the previous one (overlap), so an idea
    that straddles a boundary is not lost.
    """
    if overlap < 0 or overlap >= max_chars // 2:
        raise ValueError("overlap must be in the range [0, max_chars // 2)")

    # Reserve room for the overlap and a space so the final chunk
    # is guaranteed to fit into max_chars
    limit = max_chars - overlap - 1 if overlap else max_chars

    # 1. Paragraphs, with whitespace and line breaks collapsed inside each one
    paragraphs = [
        re.sub(r"\s+", " ", p).strip()
        for p in _PARAGRAPH_BREAK.split(text.replace("\r\n", "\n"))
    ]
    pieces: list[str] = []
    for p in paragraphs:
        if not p:
            continue
        pieces.extend([p] if len(p) <= limit else _split_long(p, limit))

    # 2. Pack small pieces into chunks while they fit
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

    # 3. Overlap: prepend the tail of the previous chunk to each chunk
    if overlap and len(chunks) > 1:
        result = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            tail = prev[-overlap:]
            if len(prev) > overlap and prev[-overlap - 1] != " ":
                # The tail starts mid-word: drop the partial word
                space = tail.find(" ")
                tail = tail[space + 1:] if space != -1 else ""
            result.append((tail + " " + cur).strip())
        chunks = result

    return chunks