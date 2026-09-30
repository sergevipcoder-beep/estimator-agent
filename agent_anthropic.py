"""Агент-сметчик: основной цикл tool-calling на голом Anthropic SDK.

Никаких agent-фреймворков: только официальный SDK и обычный while.

Запуск:
    python agent.py "Посчитай смету по брифу brief_shop.txt"
    python agent.py --test        # три заготовленных сценария
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

import tools

load_dotenv()

# Консоль Windows по умолчанию не в UTF-8 — без этого кириллица в print падает.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = 2048
MAX_STEPS = 12  # предохранитель от бесконечного цикла

SYSTEM_PROMPT = """Ты — помощник-сметчик в студии веб-разработки.
Студия делает сайты, интернет-магазины, веб-приложения, CRM, админ-панели и парсеры.

Правила работы:
- Ставки и содержание брифов ты не знаешь и не угадываешь — берёшь только из инструментов.
- Все арифметические расчёты делаешь инструментом calculate, а не в уме.
- Если инструмент вернул ошибку, прочитай текст ошибки и исправься: уточни аргументы
  или обойдись доступными данными. Если данных по какой-то роли нет — не выдумывай
  ставку, а честно укажи это в ответе отдельной строкой.
- Финальный ответ: короткая смета по ролям (часы x ставка = сумма), итог и, если есть,
  список того, что посчитать не удалось."""

LOG_DIR = Path(__file__).parent / "logs"

TEST_QUESTIONS = [
    # многошаговый сценарий: чтение файла -> несколько ставок -> расчёт
    "Прочитай бриф brief_shop.txt и посчитай смету по нему. "
    "Уложимся ли мы в бюджет клиента?",
    # проверка, что агент выбирает инструменты, а не вызывает все подряд
    "Посчитай 15% от 340, а потом скажи, тепло ли сейчас в Амстердаме.",
    # заведомо битые аргументы: агент должен получить ошибки и выкарабкаться
    "Узнай ставку роли 'котик' и посчитай выражение '10 / 0'. "
    "Потом всё равно скажи ставку бэкенда.",
]


class Log:
    """Пишет одновременно в консоль и в файл — файл потом несём преподавателю."""

    def __init__(self, path: Path):
        self.path = path
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def execute_tool(block, log: Log) -> dict:
    """Выполняет один запрошенный инструмент и всегда возвращает tool_result.

    Ключевая идея: упасть нельзя. Любая ошибка превращается в текст для модели,
    и модель получает шанс исправиться на следующем витке цикла.
    """
    log(f"  -> вызов `{block.name}` с аргументами {block.input}")
    func = tools.REGISTRY.get(block.name)
    try:
        if func is None:
            raise tools.ToolError(f"инструмента {block.name!r} не существует")
        output, is_error = func(**block.input), False
    except tools.ToolError as exc:
        output, is_error = f"Ошибка инструмента: {exc}", True
    except TypeError as exc:
        output, is_error = f"Ошибка инструмента: неверный набор аргументов — {exc}", True
    except Exception as exc:  # непредвиденное: тоже не роняем процесс
        output, is_error = f"Ошибка инструмента: {type(exc).__name__}: {exc}", True

    marker = "ОШИБКА" if is_error else "ok"
    preview = output if len(output) < 300 else output[:300] + " ...[обрезано]"
    log(f"  <- {marker}: {preview}")
    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": output,
        "is_error": is_error,
    }


def run_agent(question: str, log: Log) -> str:
    client = Anthropic()  # ключ берётся из переменной ANTHROPIC_API_KEY
    messages = [{"role": "user", "content": question}]

    log(f"# Вопрос\n\n{question}\n")
    log(f"Модель: {MODEL}\n")

    for step in range(1, MAX_STEPS + 1):
        log(f"## Шаг {step}")
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools.TOOL_SCHEMAS,
            messages=messages,
        )
        log(
            f"  stop_reason={response.stop_reason}, "
            f"токенов: вход {response.usage.input_tokens}, "
            f"выход {response.usage.output_tokens}"
        )

        # мысли модели вслух между вызовами инструментов
        for block in response.content:
            if block.type == "text" and block.text.strip():
                log(f"  модель рассуждает: {block.text.strip()}")

        if response.stop_reason != "tool_use":
            final = "\n".join(b.text for b in response.content if b.type == "text")
            log(f"\n# Финальный ответ\n\n{final}")
            return final

        # 1) кладём в историю ответ модели целиком, включая блоки tool_use
        messages.append({"role": "assistant", "content": response.content})

        # 2) выполняем ВСЕ запрошенные инструменты и собираем результаты
        results = [
            execute_tool(block, log)
            for block in response.content
            if block.type == "tool_use"
        ]

        # 3) отдаём результаты обратно как сообщение роли user
        messages.append({"role": "user", "content": results})

    log("\n# Прерывание: превышен лимит шагов")
    return "Не удалось закончить за отведённое число шагов."


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    questions = TEST_QUESTIONS if args[0] == "--test" else [" ".join(args)]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for index, question in enumerate(questions, start=1):
        suffix = f"-{index}" if len(questions) > 1 else ""
        log = Log(LOG_DIR / f"{stamp}{suffix}.md")
        try:
            run_agent(question, log)
        finally:
            log.save()
            print(f"\n[лог сохранён: {log.path}]\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
