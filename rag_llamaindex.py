"""То же самое, что rag/ — но на LlamaIndex. Часть Б Шага 2.

Сравнение с рукописной версией:

| Что делаем           | Руками (rag/)                  | Здесь                       |
|----------------------|--------------------------------|-----------------------------|
| чтение файлов        | rag/chunking.py, glob + read   | SimpleDirectoryReader       |
| резка на чанки       | split_text() с перехлёстом     | SentenceSplitter            |
| эмбеддинги батчами   | rag/embeddings.py              | OpenAILikeEmbedding         |
| векторное хранилище  | rag/index.py, Qdrant           | VectorStoreIndex (в памяти) |
| поиск + промпт       | rag/search.py                  | as_query_engine()           |

Про Qdrant в этой версии. Официальная интеграция
llama-index-vector-stores-qdrant объявляет requires_python ">=3.10,<3.14",
то есть на Python 3.14 она не устанавливается: pip откатывается на старые
версии зависимостей, а у того numpy нет готовых сборок под 3.14 и он
пытается компилироваться из исходников. Поэтому здесь используется
встроенное хранилище LlamaIndex. На механику RAG это не влияет —
меняется только место, где лежат векторы.

Запуск (2-3 запроса на индексацию + 2 на каждый вопрос):
    python rag_llamaindex.py --test
    python rag_llamaindex.py "Что пошло не так в проектах с 1С?"
"""

import os
import sys

import truststore
from dotenv import load_dotenv

# LlamaIndex тянет за собой свой httpx со вшитым списком корневых сертификатов.
# Если в системе есть свой сертификат в цепочке (антивирус, корпоративный прокси),
# запросы падают с CERTIFICATE_VERIFY_FAILED. Эта строка заставляет Python брать
# сертификаты из хранилища операционной системы. Проверка сертификатов остаётся
# включённой — отключать её нельзя.
truststore.inject_into_ssl()
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike

from rag.chunking import KNOWLEDGE_DIR
from rag.search import SYSTEM_PROMPT, TEST_QUESTIONS, TOP_K

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"
CHAT_MODEL = os.getenv("MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nemotron-3-embed-1b:free")


def configure() -> None:
    """Глобальные настройки LlamaIndex: какая модель отвечает, какая векторизует."""
    key = os.environ["OPENROUTER_API_KEY"]
    Settings.llm = OpenAILike(
        model=CHAT_MODEL,
        api_base=BASE_URL,
        api_key=key,
        is_chat_model=True,
        max_tokens=1024,
        max_retries=0,
        system_prompt=SYSTEM_PROMPT,
    )
    Settings.embed_model = OpenAILikeEmbedding(
        model_name=EMBED_MODEL,
        api_base=BASE_URL,
        api_key=key,
        embed_batch_size=32,
    )
    # Параметры чанкинга — те же, что в нашей версии, чтобы сравнение было честным.
    # Разница: LlamaIndex считает в токенах, мы считали в символах.
    Settings.node_parser = SentenceSplitter(chunk_size=400, chunk_overlap=60)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    questions = TEST_QUESTIONS if args[0] == "--test" else [" ".join(args)]

    configure()

    # Вот здесь вся индексация: чтение, чанкинг, эмбеддинги, хранилище — 2 строки.
    documents = SimpleDirectoryReader(str(KNOWLEDGE_DIR)).load_data()
    index = VectorStoreIndex.from_documents(documents)
    print(f"Документов прочитано: {len(documents)}")
    print(f"Узлов в индексе: {len(index.docstore.docs)}")
    print(f"Модель ответов: {CHAT_MODEL}\nМодель эмбеддингов: {EMBED_MODEL}\n")

    engine = index.as_query_engine(similarity_top_k=TOP_K)

    for question in questions:
        print(f"# Вопрос\n\n{question}\n")
        response = engine.query(question)
        print("## Источники")
        for node in response.source_nodes:
            name = node.metadata.get("file_name", "?")
            print(f"  {node.score:.3f}  {name}")
        print(f"\n## Ответ\n\n{response}")
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
