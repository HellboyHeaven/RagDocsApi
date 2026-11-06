import pytest

from app.services.parsing import ParseError, clean_pdf_text, extract_text


def test_hyphenated_words_are_joined():
    assert clean_pdf_text("это важный доку-\nмент") == "это важный документ"


def test_regular_line_breaks_are_kept():
    assert clean_pdf_text("первая строка\nвторая строка") == "первая строка\nвторая строка"


def test_extra_spaces_are_collapsed():
    assert clean_pdf_text("слово    слово\t\tслово") == "слово слово слово"


def test_nul_characters_are_removed():
    assert clean_pdf_text("при\x00вет") == "привет"


def test_text_file_is_decoded():
    assert extract_text("a.txt", "Привет".encode("utf-8")) == "Привет"


def test_bom_is_stripped():
    assert extract_text("a.txt", b"\xef\xbb\xbfhello") == "hello"


def test_wrong_encoding_raises_parse_error():
    with pytest.raises(ParseError):
        extract_text("a.txt", "Привет".encode("cp1251"))


def test_broken_pdf_raises_parse_error():
    with pytest.raises(ParseError):
        extract_text("a.pdf", b"this is not a pdf")