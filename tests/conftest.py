import json
import os
import uuid

# Must be set before the app is imported: Settings reads the environment at import time.
# Tests never touch a real LLM or a real database.
os.environ["LLM_BASE_URL"] = "http://llm.test/v1"
os.environ["LLM_API_KEY"] = "test-key"
os.environ["LLM_MODEL"] = "test-model"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/test"

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.db import get_session  # noqa: E402
from app.main import app  # noqa: E402


@pytest_asyncio.fixture
async def client():
    async def no_session():
        yield None  # CRUD functions are mocked in these tests, so no session is needed

    app.dependency_overrides[get_session] = no_session
    # ASGITransport does not run the lifespan, so the embedding model is not loaded
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def fake_embed(monkeypatch):
    """Replaces the embedding model with constant vectors."""

    async def _embed(texts):
        return [[0.0] * 384 for _ in texts]

    for module in ("documents", "ask", "search"):
        monkeypatch.setattr(f"app.routers.{module}.embed", _embed)


def make_hit(i: int = 1, distance: float = 0.2, content: str = "Some fragment text.") -> dict:
    return {
        "chunk_id": i,
        "document_id": uuid.UUID(int=i),
        "filename": f"doc{i}.txt",
        "position": i - 1,
        "content": content,
        "distance": distance,
    }


def parse_sse(text: str) -> list[dict]:
    return [json.loads(block[6:]) for block in text.split("\n\n") if block.startswith("data: ")]