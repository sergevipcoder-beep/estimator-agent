"""Инструменты агента.

Каждый инструмент — это две вещи:
  1) обычная Python-функция, которая что-то делает;
  2) JSON-схема, по которой модель понимает, когда и как её вызвать.

Модель никогда не выполняет код. Она лишь возвращает имя инструмента и
аргументы; выполняет их основной цикл в agent.py.
"""

import ast
import json
import operator
from pathlib import Path

DATA_DIR = (Path(__file__).parent / "data").resolve()


class ToolError(Exception):
    """Предсказуемая ошибка инструмента: её текст уйдёт модели как tool_result."""


# --------------------------------------------------------------------------
# 1. read_brief — чтение локального текстового файла
# --------------------------------------------------------------------------

def read_brief(filename: str) -> str:
    target = (DATA_DIR / filename).resolve()
    if not target.is_relative_to(DATA_DIR):
        raise ToolError("читать можно только файлы внутри папки data/")
    if not target.is_file():
        available = ", ".join(p.name for p in sorted(DATA_DIR.glob("*.txt"))) or "нет файлов"
        raise ToolError(f"файл {filename!r} не найден. Доступные файлы: {available}")
    return target.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 2. get_price — цена позиции из прайс-листа (фикс за тип проекта + доплаты)
# --------------------------------------------------------------------------

def _load_pricelist() -> tuple[dict, dict]:
    data = json.loads((DATA_DIR / "pricelist.json").read_text(encoding="utf-8"))
    return data["base"], data["addons"]


def get_price(item: str) -> str:
    base, addons = _load_pricelist()
    key = item.strip().lower()
    for section, entries in (("базовый проект", base), ("доплата", addons)):
        if key in entries:
            entry = entries[key]
            return (
                f"{key} — {entry['price_rub']} руб ({section}: {entry['note']})"
            )
    raise ToolError(
        f"позиции {item!r} нет в прайс-листе. "
        f"Типы проектов: {', '.join(base)}. "
        f"Доплаты: {', '.join(addons)}"
    )


# --------------------------------------------------------------------------
# 3. calculate — безопасный калькулятор на ast, без eval
# --------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ToolError(
        "в выражении разрешены только числа и операторы + - * / // % ** и скобки"
    )


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"не удалось разобрать выражение {expression!r}: {exc.msg}") from exc
    try:
        result = _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise ToolError("деление на ноль") from exc
    except OverflowError as exc:
        raise ToolError("слишком большое число") from exc
    return f"{expression} = {result:,.2f}".replace(",", " ")


# --------------------------------------------------------------------------
# 4. get_weather — заглушка. Нужна, чтобы проверить, что агент ВЫБИРАЕТ
#    инструменты, а не вызывает все подряд.
# --------------------------------------------------------------------------

_WEATHER = {
    "москва": "+4 °C, пасмурно, ветер 5 м/с",
    "екатеринбург": "-6 °C, снег, ветер 7 м/с",
    "амстердам": "+9 °C, дождь, ветер 6 м/с",
    "сочи": "+14 °C, ясно, ветер 2 м/с",
}


def get_weather(city: str) -> str:
    key = city.strip().lower()
    if key not in _WEATHER:
        raise ToolError(
            f"нет данных о погоде для {city!r}. Известные города: {', '.join(_WEATHER)}"
        )
    return f"Погода в {city}: {_WEATHER[key]}"


# --------------------------------------------------------------------------
# Схемы для модели. Текст в description — это промпт, а не документация:
# именно по нему модель решает, вызывать инструмент или нет.
# --------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "read_brief",
        "description": (
            "Прочитать текстовый файл из папки data проекта — например бриф клиента. "
            "Вызывай, когда нужно узнать содержание брифа, ТЗ или заметки по проекту."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Имя файла внутри папки data, например 'brief_shop.txt'",
                }
            },
            "required": ["filename"],
        },
    },
    {
        "name": "get_price",
        "description": (
            "Получить цену одной позиции из прайс-листа студии: либо базовую цену типа "
            "проекта (лендинг, интернет-магазин, CRM, парсер и т.п.), либо стоимость "
            "доплаты за отдельный модуль (интеграция с 1С, личный кабинет, платёжный "
            "шлюз и т.п.). Одна позиция за один вызов. Никогда не придумывай цены сам — "
            "всегда спрашивай здесь."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item": {
                    "type": "string",
                    "description": (
                        "Название позиции, например 'интернет-магазин', "
                        "'интеграция с 1С', 'парсер'"
                    ),
                }
            },
            "required": ["item"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Посчитать арифметическое выражение. Используй для любых расчётов смет, "
            "сумм и процентов — не считай в уме."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Выражение, например '600000 + 250000 + 150000' или '1370000 / 3'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Узнать текущую погоду в городе. Только про погоду, для расчётов не нужен.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Название города"}
            },
            "required": ["city"],
        },
    },
]

REGISTRY = {
    "read_brief": read_brief,
    "get_price": get_price,
    "calculate": calculate,
    "get_weather": get_weather,
}
