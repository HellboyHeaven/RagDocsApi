import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

LONG_TEXT = ("Sentence number one. " * 60).encode()


async def upload(client, name: str, content: bytes, mime: str = "text/plain"):
    return await client.post("/documents", files={"file": (name, content, mime)})


async def test_rejects_unsupported_extension(client):
    r = await upload(client, "image.png", b"data", "image/png")
    assert r.status_code == 415


async def test_rejects_oversized_file(client, monkeypatch):
    # With a 0 MB limit any non-empty file is too large
    monkeypatch.setattr("app.routers.documents.settings.max_upload_mb", 0)
    r = await upload(client, "a.txt", b"x")
    assert r.status_code == 413


async def test_rejects_file_without_text(client):
    r = await upload(client, "a.txt", b"   \n\n  ")
    assert r.status_code == 422


async def test_rejects_non_utf8_text(client):
    r = await upload(client, "a.txt", "Привет".encode("cp1251"))
    assert r.status_code == 422


async def test_rejects_broken_pdf(client):
    r = await upload(client, "a.pdf", b"this is not a pdf", "application/pdf")
    assert r.status_code == 422


async def test_upload_saves_chunks_with_vectors(client, fake_embed, monkeypatch):
    # Small chunks so the test does not depend on the local .env settings
    monkeypatch.setattr("app.routers.documents.settings.chunk_size", 200)
    monkeypatch.setattr("app.routers.documents.settings.chunk_overlap", 20)

    saved = {}

    async def fake_create_document(session, filename, chunks, vectors):
        saved.update(filename=filename, chunks=chunks, vectors=vectors)
        return SimpleNamespace(
            id=uuid.uuid4(), filename=filename, created_at=datetime.now(timezone.utc)
        )

    monkeypatch.setattr("app.routers.documents.crud.create_document", fake_create_document)

    r = await upload(client, "notes.txt", LONG_TEXT)

    assert r.status_code == 201
    body = r.json()
    assert body["filename"] == "notes.txt"
    assert len(saved["chunks"]) > 1
    assert body["chunks_count"] == len(saved["chunks"])
    assert len(saved["vectors"]) == len(saved["chunks"])  # one vector per chunk


async def test_list_documents(client, monkeypatch):
    async def fake_list(session):
        return [{
            "id": uuid.uuid4(),
            "filename": "a.txt",
            "created_at": datetime.now(timezone.utc),
            "chunks_count": 3,
        }]

    monkeypatch.setattr("app.routers.documents.crud.list_documents", fake_list)
    r = await client.get("/documents")
    assert r.status_code == 200
    assert r.json()[0]["chunks_count"] == 3


async def test_delete_unknown_document_returns_404(client, monkeypatch):
    async def fake_delete(session, doc_id):
        return False

    monkeypatch.setattr("app.routers.documents.crud.delete_document", fake_delete)
    r = await client.delete(f"/documents/{uuid.uuid4()}")
    assert r.status_code == 404


async def test_delete_document(client, monkeypatch):
    async def fake_delete(session, doc_id):
        return True

    monkeypatch.setattr("app.routers.documents.crud.delete_document", fake_delete)
    r = await client.delete(f"/documents/{uuid.uuid4()}")
    assert r.status_code == 204