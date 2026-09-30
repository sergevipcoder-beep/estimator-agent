"""Проверка ключа OpenRouter и остатка суточного лимита.

Обращается к служебному эндпоинту /api/v1/key — он НЕ тратит квоту моделей,
поэтому запускать можно сколько угодно. Полезно перед прогоном агента:
бесплатный тариф даёт всего 50 запросов в сутки.

Используется только стандартная библиотека: пакет httpx в новых версиях
переименован в httpx2, и полагаться на его имя ненадёжно.

Запуск: python check_key.py
"""

import json
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

key = os.getenv("OPENROUTER_API_KEY")
if not key:
    raise SystemExit("Нет ключа. Скопируй .env.example в .env и впиши OPENROUTER_API_KEY.")

request = urllib.request.Request(
    "https://openrouter.ai/api/v1/key",
    headers={"Authorization": f"Bearer {key}"},
)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)["data"]
except urllib.error.HTTPError as exc:
    if exc.code == 401:
        raise SystemExit("Ключ не принят (401). Проверь, что скопирован целиком.") from exc
    raise SystemExit(f"OpenRouter вернул {exc.code}: {exc.read().decode(errors='replace')}") from exc

print(f"Ключ рабочий: {data.get('label') or 'без имени'}")
print(f"Потрачено кредитов всего: {data.get('usage')}")
print(f"Бесплатный тариф: {'да' if data.get('is_free_tier') else 'нет'}")

daily = data.get("free_model_daily_requests")
if daily:
    print(
        f"Запросы к бесплатным моделям за сегодня: "
        f"{daily['used']} из {daily['limit']}, осталось {daily['remaining']}"
    )
    if daily["remaining"] < 15:
        print("\nВНИМАНИЕ: на главный сценарий нужно ~12 запросов. Может не хватить.")
