import asyncio
from functools import lru_cache

from fastembed import TextEmbedding

from app.config import settings


@lru_cache
def get_model() -> TextEmbedding:
    # модель загружается один раз при первом вызове и остаётся в памяти
    return TextEmbedding(model_name=settings.embedding_model, cache_dir=".models")


def embed_sync(texts: list[str]) -> list[list[float]]:
    return [vec.tolist() for vec in get_model().embed(texts)]


async def embed(texts: list[str]) -> list[list[float]]:
    # эмбеддинг считает CPU и блокирует поток, поэтому уводим его с event loop
    return await asyncio.to_thread(embed_sync, texts)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)