"""Поиск по корпусу + генерация ответа строго по найденному контексту.

Это и есть RAG: вопрос -> эмбеддинг вопроса -> ближайшие чанки в Qdrant ->
подставить их в промпт -> попросить модель ответить ТОЛЬКО по ним.

Берём top-5, а не top-1, потому что векторный поиск вероятностный: нужный
кусок не всегда первый, зато модель умеет выбрать подходящий из пяти.

Запуск (2 запроса: эмбеддинг вопроса + генерация):
    python -m rag.search "Что пошло не так в проектах с 1С?"
    python -m rag.search --test
"""

import os
import sys

import openai
from dotenv import load_dotenv

from rag.embeddings import embed_one
from rag.index import COLLECTION, open_client

load_dotenv()

CHAT_MODEL = os.getenv("MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
TOP_K = 5

# Главная защита от галлюцинаций — не код, а формулировка.
SYSTEM_PROMPT = """Ты отвечаешь на вопросы по архиву прошлых проектов студии.

Жёсткие правила:
- Отвечай ТОЛЬКО на основании приведённого ниже контекста.
- Если в контексте нет ответа, напиши ровно: «В архиве проектов нет информации
  по этому вопросу.» и не добавляй никаких догадок из общих знаний.
- Не придумывай цифры, названия клиентов и сроки, которых нет в контексте.
- В конце ответа перечисли файлы-источники, на которые опирался."""

TEST_QUESTIONS = [
    # ответ в корпусе есть
    "Из-за чего сорвали срок в проекте с интеграцией 1С и на сколько?",
    # ответа в корпусе нет — модель должна честно это признать
    "Делали ли мы мобильное приложение на Flutter для доставки еды и сколько оно стоило?",
]


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    vector = embed_one(question)
    client = open_client()
    try:
        found = client.query_points(COLLECTION, query=vector, limit=top_k).points
    finally:
        client.close()
    return [
        {
            "score": point.score,
            "source": point.payload["source"],
            "chunk_index": point.payload["chunk_index"],
            "text": point.payload["text"],
        }
        for point in found
    ]


def build_context(found: list[dict]) -> str:
    return "\n\n".join(
        f"[источник: {item['source']}, чанк {item['chunk_index']}]\n{item['text']}"
        for item in found
    )


def answer(question: str, log=print) -> str:
    found = retrieve(question)
    log(f"# Вопрос\n\n{question}\n")
    log(f"## Найдено чанков: {len(found)} (top-{TOP_K})")
    for item in found:
        log(f"  {item['score']:.3f}  {item['source']} (чанк {item['chunk_index']})")

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Контекст:\n\n{build_context(found)}\n\nВопрос: {question}",
            },
        ],
    )
    if not response.choices:
        log("\n! провайдер вернул пустой ответ")
        return "Провайдер вернул пустой ответ, попробуй ещё раз."

    text = response.choices[0].message.content or "(пусто)"
    log(f"\n## Ответ\n\n{text}")
    return text


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)

    is_test = args[0] == "--test"
    questions = TEST_QUESTIONS if is_test else [" ".join(args)]

    # Лог пишем сами, а не перенаправлением вывода: иначе PowerShell сохранит
    # его в кодировке консоли и кириллица превратится в кракозябры.
    from common import Log

    log = Log("rag-test-2.7" if is_test else "rag-query", timestamped=not is_test)
    try:
        for question in questions:
            answer(question, log)
            log("\n" + "=" * 70 + "\n")
    finally:
        log.save()


if __name__ == "__main__":
    main()
