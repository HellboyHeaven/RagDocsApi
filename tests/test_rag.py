from app.services.rag import SYSTEM_PROMPT, build_rag_messages, build_sources


def make_hit(i: int, text: str = "текст") -> dict:
    return {
        "document_id": f"00000000-0000-0000-0000-00000000000{i}",
        "filename": f"doc{i}.txt",
        "position": i,
        "content": text,
        "distance": 0.1 * i,
    }


def test_messages_structure():
    msgs = build_rag_messages("Вопрос?", [make_hit(1)])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT


def test_fragments_are_numbered_from_one():
    msgs = build_rag_messages("Вопрос?", [make_hit(1, "первый"), make_hit(2, "второй")])
    user = msgs[1]["content"]
    assert "[1] (source: doc1.txt)\nпервый" in user
    assert "[2] (source: doc2.txt)\nвторой" in user


def test_question_is_in_user_message():
    msgs = build_rag_messages("Что такое SSE?", [make_hit(1)])
    assert msgs[1]["content"].endswith("Question: Что такое SSE?")


def test_sources_numbers_match_fragments():
    sources = build_sources([make_hit(1), make_hit(2)])
    assert [s["n"] for s in sources] == [1, 2]
    assert sources[0]["filename"] == "doc1.txt"


def test_snippet_is_truncated():
    sources = build_sources([make_hit(1, "а" * 1000)])
    assert len(sources[0]["snippet"]) == 200


def test_sources_are_json_serializable():
    import json
    json.dumps(build_sources([make_hit(1)]))