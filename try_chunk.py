import sys

from app.services.chunking import chunk_text

text = open(sys.argv[1], encoding="utf-8").read()
chunks = chunk_text(text)
avg = sum(map(len, chunks)) // max(len(chunks), 1)
print(f"Кусков: {len(chunks)}, средняя длина: {avg} симв.\n")
for i, c in enumerate(chunks[:5]):
    print(f"--- кусок {i} ({len(c)} симв.) ---\n{c}\n")