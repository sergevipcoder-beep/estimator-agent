"""Агент-сметчик: цикл tool-calling на официальном Anthropic SDK.

Никаких agent-фреймворков: только SDK провайдера и обычный while.
Близнец agent_openrouter.py — цикл тот же, отличается только транспорт.

Ключа Anthropic пока нет, поэтому рабочая версия — agent_openrouter.py.
Этот файл готов и заработает без правок, как только ключ появится.

Запуск:
    python agent_anthropic.py --test 1
    python agent_anthropic.py "Посчитай смету по brief_shop.txt"
"""

import os
import sys

import anthropic
from dotenv import load_dotenv

import tools
from common import (
    MAX_STEPS,
    MAX_TOKENS,
    SYSTEM_PROMPT,
    Log,
    pick_question,
    run_tool,
)

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")


def run_agent(question: str, log: Log) -> str:
    client = anthropic.Anthropic()  # ключ из переменной ANTHROPIC_API_KEY
    messages = [{"role": "user", "content": question}]

    log(f"# Вопрос\n\n{question}\n")
    log(f"Модель: {MODEL} (напрямую через Anthropic)\n")

    for step in range(1, MAX_STEPS + 1):
        log(f"## Шаг {step}")
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,  # у Anthropic системный промпт — отдельный параметр,
                tools=tools.TOOL_SCHEMAS,  # а схемы берутся как есть, без переупаковки
                messages=messages,
            )
        except anthropic.RateLimitError:
            log("\n# Остановка: исчерпан лимит запросов (429)")
            return "Лимит запросов исчерпан, попробуй позже."
        except anthropic.APIStatusError as exc:
            log(f"\n# Остановка: API вернул {exc.status_code}")
            return f"Ошибка API: {exc.status_code}"

        log(
            f"  stop_reason={response.stop_reason}, "
            f"токенов: вход {response.usage.input_tokens}, "
            f"выход {response.usage.output_tokens}"
        )
        for block in response.content:
            if block.type == "text" and block.text.strip():
                log(f"  модель говорит: {block.text.strip()}")

        if response.stop_reason != "tool_use":
            final = "\n".join(b.text for b in response.content if b.type == "text")
            log(f"\n# Финальный ответ\n\n{final}")
            return final

        # 1) ответ модели целиком отправляем обратно в историю
        messages.append({"role": "assistant", "content": response.content})

        # 2) выполняем все запрошенные инструменты
        # 3) результаты уходят ОДНИМ сообщением с ролью user —
        #    в этом главное отличие от OpenAI-формата, где на каждый
        #    результат создаётся отдельное сообщение с ролью "tool"
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            # Аргументы у Anthropic приходят готовым словарём — разбирать JSON не нужно
            output, is_error = run_tool(block.name, block.input, log)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,  # у OpenAI такого поля нет вообще
                }
            )
        messages.append({"role": "user", "content": results})

    log(f"\n# Прерывание: превышен лимит {MAX_STEPS} шагов")
    return "Не удалось закончить за отведённое число шагов."


def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "Нет ключа Anthropic. Пока пользуйся agent_openrouter.py."
        )
    question, name = pick_question(sys.argv[1:])
    log = Log(f"anthropic-{name}")
    try:
        run_agent(question, log)
    finally:
        log.save()


if __name__ == "__main__":
    main()
