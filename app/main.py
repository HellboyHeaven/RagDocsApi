import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import ask, documents, search
from app.services.embeddings import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model at startup so the first request doesn't wait for it
    await asyncio.to_thread(get_model)
    yield


app = FastAPI(title="RagDocsApi", lifespan=lifespan)

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(ask.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


# Mount LAST, after all routers: it catches everything that wasn't matched earlier
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="static",
)