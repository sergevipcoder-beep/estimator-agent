"""Индексация корпуса в Qdrant.

Задание предлагает поднять Qdrant в Docker. Мы используем официальный
local mode того же клиента: QdrantClient(path=...) — тот же самый API,
но без сервера и без Docker. Чтобы переехать на серверный Qdrant,
меняется ровно одна строка подключения, остальной код не трогается.

Запуск (2-3 запроса к API на весь корпус):
    python -m rag.index
"""

import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

from rag.chunking import load_chunks
from rag.embeddings import MODEL as EMBED_MODEL
from rag.embeddings import embed_texts

STORAGE_PATH = str((Path(__file__).parent.parent / "qdrant_data").resolve())
COLLECTION = "projects"


def open_client() -> QdrantClient:
    # Одна строка, отделяющая нас от Docker:
    # для сервера было бы QdrantClient(url="http://localhost:6333")
    return QdrantClient(path=STORAGE_PATH)


def build_index() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    chunks = load_chunks()
    print(f"Чанков к индексации: {len(chunks)}")
    print(f"Модель эмбеддингов: {EMBED_MODEL}")

    vectors = embed_texts([c.text for c in chunks])
    # Размерность заранее неизвестна: в каталоге OpenRouter её нет,
    # а Qdrant требует размер вектора при создании коллекции.
    size = len(vectors[0])
    print(f"Размерность вектора: {size}")

    client = open_client()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
    )

    client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(
                id=number,
                vector=vector,
                # payload — это метаданные. Благодаря source в ответе
                # можно показать, из какого документа взят факт.
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.index,
                },
            )
            for number, (chunk, vector) in enumerate(zip(chunks, vectors))
        ],
    )
    count = client.count(COLLECTION).count
    client.close()
    print(f"Загружено точек в коллекцию {COLLECTION!r}: {count}")
    print(f"Хранилище: {STORAGE_PATH}")


if __name__ == "__main__":
    build_index()
