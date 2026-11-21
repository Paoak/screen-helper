"""Точка входа: python -m screen_helper [--config путь/к/config.json]."""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .config import ConfigError, load_settings


def main(argv: Optional[List[str]] = None) -> int:
    """Читает настройки и запускает приложение. Возвращает код выхода."""
    parser = argparse.ArgumentParser(
        prog="screen_helper",
        description="Локальный ассистент: скриншот по хоткею -> ответ модели из Ollama.",
    )
    parser.add_argument("--config", help="путь к config.json (по умолчанию - config.json в текущей папке)")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.config)
    except ConfigError as e:
        print(f"Ошибка настроек: {e}", file=sys.stderr)
        return 2

    # GUI-модули импортируем здесь: так --help и проверка настроек работают без tkinter/keyboard
    from .app import App
    App(settings).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
