"""Агент-сметчик: цикл tool-calling через OpenRouter (OpenAI-совместимый API).

Никаких agent-фреймворков: только SDK провайдера и обычный while.
Близнец agent_anthropic.py — цикл тот же, отличается только транспорт.

Запуск:
    python agent_openrouter.py --test 1
    python agent_openrouter.py "Посчитай смету по brief_shop.txt"
"""

import json
import os
import sys

import openai
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

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = os.getenv("MODEL", "nvidia/nemotron-3-super-120b-a12b:free")


def openai_tool_schemas() -> list[dict]:
    """Переупаковывает схемы из tools.py в формат OpenAI.

    Содержимое схемы то же самое — меняется только обёртка:
    Anthropic ждёт input_schema на верхнем уровне,
    OpenAI — parameters внутри объекта function.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["input_schema"],
            },
        }
        for schema in tools.TOOL_SCHEMAS
    ]


def tool_result_message(tool_call_id: str, output: str, is_error: bool) -> dict:
    """Результат инструмента в формате OpenAI.

    Отличие от Anthropic: отдельная роль "tool" вместо блока tool_result,
    и НЕТ поля is_error — про ошибку модели приходится сообщать текстом.
    """
    content = f"ОШИБКА: {output}" if is_error else output
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def handle_tool_call(call, log: Log) -> dict:
    """Разбирает один запрос на вызов инструмента и выполняет его."""
    # Особенность OpenAI-формата: аргументы приходят СТРОКОЙ JSON, а не словарём.
    # Модель иногда присылает битый JSON — у Anthropic такого класса ошибок нет.
    try:
        args = json.loads(call.function.arguments or "{}")
        if not isinstance(args, dict):
            raise ValueError("ожидался объект JSON")
    except (json.JSONDecodeError, ValueError) as exc:
        log(f"  -> вызов `{call.function.name}`, аргументы не разобрались")
        log(f"  <- ОШИБКА: битый JSON от модели: {call.function.arguments!r}")
        return tool_result_message(
            call.id,
            f"аргументы не удалось разобрать как JSON ({exc}). "
            "Пришли аргументы ещё раз корректным JSON-объектом.",
            is_error=True,
        )

    output, is_error = run_tool(call.function.name, args, log)
    return tool_result_message(call.id, output, is_error)


def run_agent(question: str, log: Log) -> str:
    client = openai.OpenAI(
        base_url=BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=0,  # повторы отключены: бесплатный лимит 50 запросов в сутки
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    log(f"# Вопрос\n\n{question}\n")
    log(f"Модель: {MODEL} (через OpenRouter)\n")

    empty_responses = 0  # пустые ответы провайдера подряд: переживаем один

    for step in range(1, MAX_STEPS + 1):
        log(f"## Шаг {step}")
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                tools=openai_tool_schemas(),
                messages=messages,
            )
        except openai.RateLimitError:
            log("\n# Остановка: исчерпан лимит запросов (429)")
            log("Бесплатный тариф OpenRouter: 20 запросов в минуту, 50 в сутки.")
            return "Лимит запросов исчерпан, попробуй позже."
        except openai.APIStatusError as exc:
            log(f"\n# Остановка: API вернул {exc.status_code}")
            log(str(exc.message) if hasattr(exc, "message") else str(exc))
            return f"Ошибка API: {exc.status_code}"

        # Бесплатные провайдеры OpenRouter иногда возвращают ответ вообще без
        # choices (сбой upstream). Один такой проглатываем и повторяем шаг;
        # два подряд — выходим, чтобы не жечь суточный лимит.
        if not response.choices:
            empty_responses += 1
            if empty_responses >= 2:
                log("\n# Остановка: провайдер дважды вернул пустой ответ")
                return "Провайдер недоступен, попробуй позже или смени модель."
            log("  ! пустой ответ провайдера, повторяю этот же шаг")
            continue
        empty_responses = 0

        message = response.choices[0].message
        if response.usage:
            log(
                f"  finish_reason={response.choices[0].finish_reason}, "
                f"токенов: вход {response.usage.prompt_tokens}, "
                f"выход {response.usage.completion_tokens}"
            )

        if message.content and message.content.strip():
            log(f"  модель говорит: {message.content.strip()}")

        # Проверяем именно наличие tool_calls, а не finish_reason: некоторые
        # модели через OpenRouter отдают finish_reason="stop", но при этом
        # всё-таки просят вызвать инструмент.
        if not message.tool_calls:
            final = message.content or "(пустой ответ)"
            log(f"\n# Финальный ответ\n\n{final}")
            return final

        # 1) ответ модели целиком отправляем обратно в историю
        messages.append(message.model_dump(exclude_none=True))

        # 2) выполняем все запрошенные инструменты
        # 3) каждый результат — отдельное сообщение с ролью "tool"
        for call in message.tool_calls:
            messages.append(handle_tool_call(call, log))

    log(f"\n# Прерывание: превышен лимит {MAX_STEPS} шагов")
    return "Не удалось закончить за отведённое число шагов."


def main() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit(
            "Нет ключа. Скопируй .env.example в .env и впиши OPENROUTER_API_KEY."
        )
    question, name = pick_question(sys.argv[1:])
    log = Log(name)
    try:
        run_agent(question, log)
    finally:
        log.save()


if __name__ == "__main__":
    main()
