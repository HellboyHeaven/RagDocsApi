import pytest

from app.services.rag import NO_ANSWER
from tests.conftest import make_hit, parse_sse


@pytest.fixture(autouse=True)
def fixed_threshold(monkeypatch):
    # Independent of whatever MAX_DISTANCE is set in the local .env
    monkeypatch.setattr("app.routers.ask.settings.max_distance", 0.5)


@pytest.fixture
def llm_calls(monkeypatch):
    """Replaces the LLM with a fake and records every prompt it receives."""
    calls = []

    async def fake_stream_chat(messages):
        calls.append(messages)
        yield {"type": "token", "text": "The answer "}
        yield {"type": "token", "text": "is 42 [1]."}
        yield {"type": "usage", "in": 10, "out": 5}

    monkeypatch.setattr("app.routers.ask.stream_chat", fake_stream_chat)
    return calls


def patch_search(monkeypatch, hits):
    async def fake_search(session, vector, k):
        return hits

    monkeypatch.setattr("app.routers.ask.crud.search_chunks", fake_search)


async def ask(client, question: str = "What is the answer?", **extra):
    return await client.post("/ask", json={"question": question, **extra})


async def test_answer_streams_sources_first(client, fake_embed, llm_calls, monkeypatch):
    patch_search(monkeypatch, [make_hit(1, 0.2, "The answer is 42.")])

    r = await ask(client)

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[0]["type"] == "sources"
    assert events[0]["sources"][0]["filename"] == "doc1.txt"
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert answer == "The answer is 42 [1]."
    assert events[-1] == {"type": "done"}


async def test_prompt_contains_fragments_and_question(client, fake_embed, llm_calls, monkeypatch):
    patch_search(monkeypatch, [make_hit(1, 0.2, "The answer is 42.")])

    await ask(client, "What is it?")

    user_message = llm_calls[0][1]["content"]
    assert "[1] (source: doc1.txt)\nThe answer is 42." in user_message
    assert user_message.endswith("Question: What is it?")


async def test_far_chunks_skip_the_llm(client, fake_embed, llm_calls, monkeypatch):
    patch_search(monkeypatch, [make_hit(1, 0.9)])  # above the 0.5 threshold

    r = await ask(client)

    events = parse_sse(r.text)
    assert events[0] == {"type": "sources", "sources": []}
    assert events[1] == {"type": "token", "text": NO_ANSWER}
    assert events[-1] == {"type": "done"}
    assert llm_calls == []  # no model call, no tokens spent


async def test_only_close_chunks_reach_the_prompt(client, fake_embed, llm_calls, monkeypatch):
    patch_search(monkeypatch, [make_hit(1, 0.2, "NEAR CONTENT"), make_hit(2, 0.9, "FARAWAY CONTENT")])

    r = await ask(client)

    sources = parse_sse(r.text)[0]["sources"]
    assert [s["n"] for s in sources] == [1]
    assert "NEAR CONTENT" in llm_calls[0][1]["content"]
    assert "FARAWAY CONTENT" not in llm_calls[0][1]["content"]


async def test_llm_failure_is_reported_as_event(client, fake_embed, monkeypatch):
    patch_search(monkeypatch, [make_hit()])

    async def broken_stream(messages):
        raise RuntimeError("boom")
        yield  # makes this function an async generator

    monkeypatch.setattr("app.routers.ask.stream_chat", broken_stream)

    r = await ask(client)

    assert r.status_code == 200  # the status line is already sent once streaming starts
    assert parse_sse(r.text)[-1]["type"] == "error"


@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "x" * 2001}, {"question": "ok", "k": 0}, {"question": "ok", "k": 21}],
)
async def test_invalid_requests_are_rejected(client, payload):
    r = await client.post("/ask", json=payload)
    assert r.status_code == 422