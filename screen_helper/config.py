"""Настройки программы: значения по умолчанию и загрузка из config.json."""
from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Optional, Union

# Допустимые уровни рассуждения для моделей, которые принимают не true/false, а уровень
THINK_LEVELS = ("low", "medium", "high")

DEFAULT_PROMPT = (
    "На скриншоте вопрос (чаще всего с вариантами ответа). "
    "Прочитай вопрос и все варианты. "
    "Сначала одной строкой дай правильный ответ (буква или номер и текст варианта), "
    "затем 1-2 предложения объяснения. "
    "Если правильных вариантов несколько - перечисли все. "
    "Если вопрос открытый - дай краткий ответ. "
    "Если текст на скриншоте нечитаем или вопрос обрезан - так и скажи, ничего не выдумывай. "
    "Отвечай на языке вопроса."
)


class ConfigError(ValueError):
    """Ошибка в файле настроек (понятное сообщение для пользователя)."""


@dataclass(frozen=True)
class Settings:
    """Все настройки программы. Значения по умолчанию подходят для первого запуска."""

    ollama_url: str = "http://localhost:11434/api/chat"  # адрес Ollama
    model: str = "gemma4:12b"        # модель с поддержкой изображений (см. `ollama list`)
    keep_alive: str = "30m"          # сколько держать модель в памяти между запросами
    num_ctx: int = 16384             # размер контекста (с рассуждением лучше не меньше 16384)
    max_side: int = 1600             # макс. сторона скриншота в px: меньше = быстрее
    think: Union[bool, str] = True   # True/False или уровень "low"/"medium"/"high"
    request_timeout: int = 300       # сколько секунд ждать данные от Ollama
    hotkey_full: str = "ctrl+alt+s"  # снимок всего экрана
    hotkey_area: str = "ctrl+alt+a"  # снимок выделенной области
    hotkey_hide: str = "ctrl+alt+h"  # скрыть/показать окно с ответом
    hotkey_quit: str = "ctrl+alt+q"  # выйти из программы
    prompt: str = DEFAULT_PROMPT     # инструкция для модели


_INT_FIELDS = {"num_ctx", "max_side", "request_timeout"}


def _validate(key: str, value) -> None:
    """Проверяет тип и допустимость одного значения, иначе бросает ConfigError."""
    if key in _INT_FIELDS:
        # bool в Python - подкласс int, поэтому исключаем его явно
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"Поле «{key}» должно быть положительным целым числом")
    elif key == "think":
        if not (isinstance(value, bool) or value in THINK_LEVELS):
            raise ConfigError(
                f"Поле «think» должно быть true, false или одним из {list(THINK_LEVELS)}"
            )
    else:
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"Поле «{key}» должно быть непустой строкой")


def load_settings(path: Optional[Union[str, Path]] = None) -> Settings:
    """
    Читает настройки из JSON-файла и накладывает их на значения по умолчанию.

    path=None: берём config.json из текущей папки, а если его нет - настройки по умолчанию.
    Явно указанный файл обязан существовать.
    Ключи, начинающиеся с «_», игнорируются (удобно для комментариев в JSON).
    """
    if path is None:
        file = Path("config.json")
        if not file.exists():
            return Settings()
    else:
        file = Path(path)
        if not file.exists():
            raise ConfigError(f"Файл настроек не найден: {file}")

    try:
        # utf-8-sig: файлы из Блокнота/PowerShell на Windows часто начинаются с BOM
        data = json.loads(file.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"Файл {file} - некорректный JSON: {e}") from e

    if not isinstance(data, dict):
        raise ConfigError(f"Файл {file} должен содержать JSON-объект {{...}}")

    data = {k: v for k, v in data.items() if not k.startswith("_")}
    known = {f.name for f in fields(Settings)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"Неизвестные поля в {file}: {', '.join(unknown)}. Допустимые: {', '.join(sorted(known))}"
        )
    for key, value in data.items():
        _validate(key, value)
    return replace(Settings(), **data)
