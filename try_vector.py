import asyncio

from sqlalchemy import select

from app.db import SessionLocal, engine
from app.models import Chunk, Document
from app.services.embeddings import embed

TEXTS = [
    "Кошка спит на диване",
    "Налоги нужно заплатить до первого июля",
    "Борщ варят из свёклы и капусты",
    "Банк одобрил заявку на кредит",
]
QUERIES = ["Котёнок дремлет", "Когда сдавать декларацию", "Рецепт супа"]


async def main():
    async with SessionLocal() as s:
        doc = Document(filename="demo.txt")
        s.add(doc)
        await s.flush()  # получаем doc.id, не завершая транзакцию

        vectors = await embed(TEXTS)
        for i, (text, vec) in enumerate(zip(TEXTS, vectors)):
            s.add(Chunk(document_id=doc.id, position=i, content=text, embedding=vec))
        await s.commit()

        for q in QUERIES:
            [qv] = await embed([q])
            dist = Chunk.embedding.cosine_distance(qv)
            rows = (
                await s.execute(
                    select(Chunk.content, dist.label("dist")).order_by(dist).limit(2)
                )
            ).all()
            print(f"\nЗапрос: {q!r}")
            for content, d in rows:
                print(f"  {d:.2f}  {content}")

        await s.delete(doc)  # чистим за собой, куски удалятся каскадом
        await s.commit()
    await engine.dispose()


asyncio.run(main())