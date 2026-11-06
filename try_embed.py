from app.services.embeddings import cosine, embed_sync

texts = [
    "Кошка спит на диване",
    "Котёнок дремлет на софе",
    "Налоги нужно заплатить до первого июля",
    "Платёж по налогу необходимо внести в начале лета",
    "Как приготовить борщ",
]

vectors = embed_sync(texts)
print(f"Размер вектора: {len(vectors[0])}\n")
print("Первые 5 чисел первого вектора:", [round(x, 3) for x in vectors[0][:5]], "\n")

pairs = [(0, 1), (2, 3), (0, 2), (0, 4), (2, 4)]
for i, j in pairs:
    print(f"{cosine(vectors[i], vectors[j]):.2f}  {texts[i]!r} <-> {texts[j]!r}")