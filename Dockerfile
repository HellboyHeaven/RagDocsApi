FROM docker.io/library/python:3.12-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image: the container then starts fast and works offline.
# This layer sits before the app code, so code changes don't trigger a re-download.
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "import os; from fastembed import TextEmbedding; TextEmbedding(model_name=os.environ['EMBEDDING_MODEL'], cache_dir='.models')"

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .

EXPOSE 8000

# Apply migrations, then start the server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]