import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document


async def create_document(
    session: AsyncSession,
    filename: str,
    chunks: list[str],
    vectors: list[list[float]],
) -> Document:
    doc = Document(filename=filename)
    session.add(doc)
    await session.flush()  # получаем doc.id до коммита

    session.add_all(
        Chunk(document_id=doc.id, position=i, content=text, embedding=vec)
        for i, (text, vec) in enumerate(zip(chunks, vectors))
    )
    await session.commit()  # документ и все куски сохраняются одной транзакцией
    return doc


async def list_documents(session: AsyncSession) -> list[dict]:
    result = await session.execute(
        select(Document, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
    )
    return [
        {"id": d.id, "filename": d.filename, "created_at": d.created_at, "chunks_count": n}
        for d, n in result.all()
    ]


async def delete_document(session: AsyncSession, doc_id: uuid.UUID) -> bool:
    doc = await session.get(Document, doc_id)
    if doc is None:
        return False
    await session.delete(doc)
    await session.commit()
    return True