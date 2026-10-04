"""Эмбеддинги через OpenRouter.

Эмбеддинг — это превращение текста в массив чисел так, что близкие по смыслу
тексты дают близкие массивы. Благодаря этому поиск идёт по смыслу, а не по
совпадению слов.

Важно для бесплатного тарифа: поле input принимает СПИСОК текстов, то есть
весь корпус уходит за 1-2 запроса вместо одного запроса на чанк.

Проверка руками (1 запрос):
    python -m rag.embeddings
"""

import math
import os
import sys

import openai
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("EMBED_MODEL", "nvidia/nemotron-3-embed-1b:free")
BATCH_SIZE = 32


def _client() -> openai.OpenAI:
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=0,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Считает эмбеддинги батчами. Один запрос на каждые BATCH_SIZE текстов."""
    client = _client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(model=MODEL, input=batch)
        # порядок ответа не гарантирован — сортируем по index
        for item in sorted(response.data, key=lambda d: d.index):
            vectors.append(item.embedding)
    return vectors


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def main() -> None:
    """Проверка, что эмбеддинги действительно measure смысл, а не слова."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    phrases = [
        "интернет-магазин автозапчастей",
        "магазин по продаже деталей для автомобилей",
        "рецепт борща со сметаной",
    ]
    vectors = embed_texts(phrases)

    print(f"Модель: {MODEL}")
    print(f"Размерность вектора: {len(vectors[0])}\n")
    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            print(f"{cosine(vectors[i], vectors[j]):.3f}  {phrases[i]}  <->  {phrases[j]}")
    print(
        "\nПервые две фразы почти не имеют общих слов, но близки по смыслу —"
        "\nименно поэтому поиск по эмбеддингам находит то, что не находит поиск по строкам."
    )


if __name__ == "__main__":
    main()
