import asyncio
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
from app.services.parsing import ParseError, extract_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
EMBED_BATCH = 64  # how many chunks to embed per model call


@router.post("", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile,
    session: AsyncSession = Depends(get_session),
):
    filename = file.filename or "untitled"
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Only .txt, .md and .pdf files are supported")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    raw = await file.read(max_bytes + 1)  # read one byte past the limit to detect overflow
    if len(raw) > max_bytes:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB")

    try:
        # Parsing is CPU-bound, so keep it off the event loop
        text = await asyncio.to_thread(extract_text, filename, raw)
    except ParseError as e:
        raise HTTPException(422, str(e))

    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(422, "The file contains no text")

    vectors: list[list[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        vectors.extend(await embed(chunks[i : i + EMBED_BATCH]))

    doc = await crud.create_document(session, filename, chunks, vectors)
    logger.info("Uploaded document %s: %d chunks", filename, len(chunks))
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
        raise HTTPException(404, "Document not found")