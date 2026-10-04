"""Резка документов на чанки с перехлёстом.

Чанк — единица поиска. Эмбеддинг целого документа размывает смысл, поэтому
режем на куски. Перехлёст (overlap) нужен, чтобы факт, попавший на стык
двух чанков, не потерялся ни в одном из них.

Про размер. Задание просит 300-500 токенов. Токенизатор режет кириллицу
гораздо мельче латиницы: для русского текста один токен — это примерно
2-3 символа, для английского около 4. Поэтому 400 токенов по-русски — это
около 1000 символов, а не 1600. Считаем в символах и держим этот коэффициент.

Запуск для просмотра результата (API не трогается):
    python -m rag.chunking
    python -m rag.chunking 400 80
"""

import sys
from dataclasses import dataclass
from pathlib import Path

KNOWLEDGE_DIR = (Path(__file__).parent.parent / "data" / "knowledge").resolve()

TARGET_CHARS = 1000  # ~400 токенов для русского текста
OVERLAP_CHARS = 150

CHARS_PER_TOKEN_RU = 2.5  # грубая оценка для отчётности


@dataclass
class Chunk:
    text: str
    source: str  # имя файла — попадёт в метаданные и в ответ как ссылка
    index: int  # номер чанка внутри документа

    @property
    def approx_tokens(self) -> int:
        return int(len(self.text) / CHARS_PER_TOKEN_RU)


def split_text(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Режет текст по абзацам, добирая их до целевого размера.

    Резать по абзацам, а не по символам, важно: иначе чанк начинается
    с середины предложения и его эмбеддинг получается мусорным.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= target_chars or not current:
            current = candidate
            continue
        chunks.append(current)
        # новый чанк начинается с хвоста предыдущего — это и есть перехлёст
        tail = current[-overlap_chars:] if overlap_chars else ""
        current = f"{tail}\n\n{paragraph}" if tail else paragraph

    if current:
        chunks.append(current)
    return chunks


def load_chunks(
    target_chars: int = TARGET_CHARS, overlap_chars: int = OVERLAP_CHARS
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for number, piece in enumerate(split_text(text, target_chars, overlap_chars)):
            chunks.append(Chunk(text=piece, source=path.name, index=number))
    return chunks


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    target = int(sys.argv[1]) if len(sys.argv) > 1 else TARGET_CHARS
    overlap = int(sys.argv[2]) if len(sys.argv) > 2 else OVERLAP_CHARS

    chunks = load_chunks(target, overlap)
    documents = len({c.source for c in chunks})
    tokens = [c.approx_tokens for c in chunks]

    print(f"Параметры: target_chars={target}, overlap_chars={overlap}")
    print(f"Документов: {documents}, чанков: {len(chunks)}")
    print(
        f"Токенов в чанке (оценка): мин {min(tokens)}, "
        f"макс {max(tokens)}, в среднем {sum(tokens) // len(tokens)}"
    )
    multi = sorted({c.source for c in chunks if c.index > 0})
    print(f"Документов, разрезанных на несколько чанков: {len(multi)}")

    print("\n--- первый чанк ---")
    print(chunks[0].text[:400])
    if len(chunks) > 1:
        print("\n--- второй чанк (видно перехлёст с первым) ---")
        print(chunks[1].text[:400])


if __name__ == "__main__":
    main()
