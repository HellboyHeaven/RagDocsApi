import asyncio

from app.services.parsing import ParseError, extract_text
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.config import settings
from app.db import get_session
from app.schemas import DocumentOut
from app.services.chunking import chunk_text
from app.services.embeddings import embed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
EMBED_BATCH = 64  # сколько кусков считаем за один вызов модели


@router.post("", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile,
    session: AsyncSession = Depends(get_session),
):
    filename = file.filename or "untitled"
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Поддерживаются только файлы .txt и .md")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    raw = await file.read(max_bytes + 1)  # читаем на байт больше лимита
    if len(raw) > max_bytes:
        raise HTTPException(413, f"Файл больше {settings.max_upload_mb} МБ")

    try:
        # разбор PDF нагружает процессор, поэтому уводим его с event loop
        text = await asyncio.to_thread(extract_text, filename, raw)
    except ParseError as e:
        raise HTTPException(422, str(e))

    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(422, "В файле нет текста")

    vectors: list[list[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        vectors.extend(await embed(chunks[i : i + EMBED_BATCH]))

    doc = await crud.create_document(session, filename, chunks, vectors)
    logger.info("Загружен документ %s: %d кусков", filename, len(chunks))
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        created_at=doc.created_at,
        chunks_count=len(chunks),
    )


@router.get("", response_model=list[DocumentOut])
async def list_documents(session: AsyncSession = Depends(get_session)):
    return await crud.list_documents(session)


@router.delete("/{doc_id}", status_code=204)
async def delete_document(
    doc_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    if not await crud.delete_document(session, doc_id):
        raise HTTPException(404, "Документ не найден")