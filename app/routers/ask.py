import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.config import settings
from app.db import get_session
from app.schemas import AskRequest
from app.services.embeddings import embed
from app.services.llm import friendly_error, stream_chat
from app.services.rag import NO_ANSWER, build_rag_messages, build_sources

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ask", tags=["ask"])


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("")
async def ask(body: AskRequest, session: AsyncSession = Depends(get_session)):
    # Retrieval happens BEFORE the stream starts, so errors here
    # are returned as a normal HTTP status
    [query_vector] = await embed([body.question])
    found = await crud.search_chunks(session, query_vector, body.k or settings.top_k)
    hits = [h for h in found if h["distance"] <= settings.max_distance]

    logger.info(
        "ask: found %d, kept after threshold %d, distances %s",
        len(found), len(hits), [round(h["distance"], 2) for h in found],
    )

    async def event_stream():
        yield sse({"type": "sources", "sources": build_sources(hits)})

        if not hits:  # nothing relevant: skip the model call and save tokens
            yield sse({"type": "token", "text": NO_ANSWER})
            yield sse({"type": "done"})
            return

        try:
            async for ev in stream_chat(build_rag_messages(body.question, hits)):
                yield sse(ev)
            yield sse({"type": "done"})
        except Exception as e:
            logger.exception("Error while generating the answer")
            yield sse({"type": "error", "message": friendly_error(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")