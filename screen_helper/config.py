"""Настройки программы: значения по умолчанию и загрузка из config.gemma.json."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Optional, Union

# Допустимые уровни рассуждения для моделей, которые принимают не true/false, а уровень
THINK_LEVELS = ("low", "medium", "high")

# С каким сервером разговариваем:
#   ollama - Ollama (/api/chat);
#   openai - любой OpenAI-совместимый сервер, например llama-server из llama.cpp (/v1/chat/completions)
BACKENDS = ("ollama", "openai")

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

    backend: str = "ollama"          # "ollama" или "openai" (llama-server и другие OpenAI-совместимые)
    ollama_url: str = "http://localhost:11434/api/chat"  # адрес Ollama (backend = "ollama")
    openai_url: str = "http://localhost:8080/v1/chat/completions"  # адрес llama-server (backend = "openai")
    model: str = "gemma4:12b"        # модель с поддержкой изображений (для llama-server - любое имя)
    keep_alive: str = "30m"          # сколько держать модель в памяти (только Ollama)
    num_ctx: int = 16384             # размер контекста (только Ollama; у llama-server задаётся флагом -c)
    max_side: int = 1600             # макс. сторона скриншота в px: меньше = быстрее
    think: Union[bool, str] = True   # True/False или уровень "low"/"medium"/"high"
    reasoning_effort: Optional[str] = None  # уровень рассуждения для backend = "openai", напр. "medium"
    request_timeout: int = 300       # сколько секунд ждать данные от сервера
    max_tokens: Optional[int] = None     # потолок длины ответа вместе с рассуждением (None = не задавать)
    temperature: Optional[float] = None  # параметры сэмплинга; None = не передавать, решает сервер
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    extra_body: dict = field(default_factory=dict)  # любые доп. поля, добавляемые в тело запроса как есть
    hotkey_full: str = "ctrl+alt+s"  # снимок всего экрана
    hotkey_area: str = "ctrl+alt+a"  # снимок выделенной области
    hotkey_hide: str = "ctrl+alt+h"  # скрыть/показать окно с ответом
    hotkey_quit: str = "ctrl+alt+q"  # выйти из программы
    prompt: str = DEFAULT_PROMPT     # инструкция для модели

    @property
    def api_url(self) -> str:
        """Адрес сервера для выбранного бэкенда."""
        return self.openai_url if self.backend == "openai" else self.ollama_url


_INT_FIELDS = {"num_ctx", "max_side", "request_timeout"}
_OPTIONAL_INT_FIELDS = {"max_tokens", "top_k"}              # целое > 0 или null
_OPTIONAL_FLOAT_FIELDS = {"temperature", "top_p", "min_p"}  # число или null
_OPTIONAL_STR_FIELDS = {"reasoning_effort"}                 # непустая строка или null


def _is_number(value) -> bool:
    """Число (int/float), но не bool: в Python True - это тоже int."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate(key: str, value) -> None:
    """Проверяет тип и допустимость одного значения, иначе бросает ConfigError."""
    if key in _INT_FIELDS:
        # bool в Python - подкласс int, поэтому исключаем его явно
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"Поле «{key}» должно быть положительным целым числом")
    elif key in _OPTIONAL_INT_FIELDS:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ConfigError(f"Поле «{key}» должно быть положительным целым числом или null")
    elif key in _OPTIONAL_FLOAT_FIELDS:
        if value is not None and not _is_number(value):
            raise ConfigError(f"Поле «{key}» должно быть числом или null")
        if key == "temperature" and value is not None and value < 0:
            raise ConfigError("Поле «temperature» не может быть отрицательным")
        if key in ("top_p", "min_p") and value is not None and not 0 <= value <= 1:
            raise ConfigError(f"Поле «{key}» должно быть от 0 до 1")
    elif key in _OPTIONAL_STR_FIELDS:
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConfigError(f"Поле «{key}» должно быть непустой строкой или null")
    elif key == "backend":
        if value not in BACKENDS:
            raise ConfigError(f"Поле «backend» должно быть одним из {list(BACKENDS)}")
    elif key == "extra_body":
        if not isinstance(value, dict):
            raise ConfigError("Поле «extra_body» должно быть JSON-объектом {...}")
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

    path=None: берём config.gemma.json из текущей папки, а если его нет - настройки по умолчанию.
    Явно указанный файл обязан существовать.
    Ключи, начинающиеся с «_», игнорируются (удобно для комментариев в JSON).
    """
    if path is None:
        file = Path("config.gemma.json")
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