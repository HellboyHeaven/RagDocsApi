import pytest

from app.services.chunking import chunk_text


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("Короткий текст.") == ["Короткий текст."]


def test_whitespace_is_normalized():
    assert chunk_text("  Привет   \n\n\n  мир  ") == ["Привет мир"]


def test_small_paragraphs_are_packed_together():
    assert chunk_text("Первый абзац.\n\nВторой абзац.") == ["Первый абзац. Второй абзац."]


def test_large_paragraphs_stay_separate():
    p1 = "а" * 59 + "."
    p2 = "б" * 59 + "."
    assert chunk_text(p1 + "\n\n" + p2, max_chars=100, overlap=0) == [p1, p2]


def test_split_happens_on_sentence_boundaries():
    sentence = ("слово " * 9).strip() + "."  # 54 символа
    text = " ".join([sentence] * 5)
    chunks = chunk_text(text, max_chars=100, overlap=0)
    assert len(chunks) == 5
    assert all(c.endswith(".") for c in chunks)


def test_no_chunk_exceeds_max_chars():
    words = [f"w{i:03d}" for i in range(300)]
    for c in chunk_text(" ".join(words), max_chars=100, overlap=20):
        assert len(c) <= 100


def test_long_word_without_spaces_is_hard_split():
    chunks = chunk_text("a" * 2000, max_chars=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_overlap_repeats_end_of_previous_chunk():
    words = [f"w{i:03d}" for i in range(100)]
    chunks = chunk_text(" ".join(words), max_chars=100, overlap=20)
    assert len(chunks) > 1
    for prev, cur in zip(chunks, chunks[1:]):
        assert prev.split()[-1] in cur.split()


def test_no_words_are_lost():
    words = [f"w{i:03d}" for i in range(100)]
    chunks = chunk_text(" ".join(words), max_chars=100, overlap=20)
    assert set(words) <= set(" ".join(chunks).split())


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("текст", max_chars=100, overlap=60)