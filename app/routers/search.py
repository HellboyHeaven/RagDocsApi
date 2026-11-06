from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.db import get_session
from app.schemas import SearchHit
from app.services.embeddings import embed

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=list[SearchHit])
async def search(
    q: str = Query(min_length=1, max_length=1000),
    k: int = Query(5, ge=1, le=20),
    session: AsyncSession = Depends(get_session),
):
    [query_vector] = await embed([q])
    return await crud.search_chunks(session, query_vector, k)