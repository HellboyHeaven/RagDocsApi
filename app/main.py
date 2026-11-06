import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import documents
from app.services.embeddings import get_model
from app.routers import documents, search


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # загружаем модель эмбеддингов заранее, чтобы первый запрос не ждал её загрузки
    await asyncio.to_thread(get_model)
    yield


app = FastAPI(title="RagDocsApi", lifespan=lifespan)
app.include_router(documents.router)
app.include_router(search.router)

@app.get("/health")
async def health():
    return {"status": "ok"}

