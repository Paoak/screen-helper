"""Тесты загрузки и проверки настроек."""
import json
from pathlib import Path

import pytest

from screen_helper.config import DEFAULT_PROMPT, ConfigError, Settings, load_settings

ROOT = Path(__file__).resolve().parent.parent


def write(tmp_path, data, encoding="utf-8"):
    """Записывает JSON-файл настроек и возвращает путь."""
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding=encoding)
    return p


def test_defaults():
    s = Settings()
    assert s.model == "gemma4:12b"
    assert s.think is True
    assert s.num_ctx == 16384
    assert s.prompt == DEFAULT_PROMPT


def test_no_config_json_in_cwd_gives_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_settings() == Settings()


def test_config_json_in_cwd_is_picked_up(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path, {"model": "qwen3-vl:8b"})
    assert load_settings().model == "qwen3-vl:8b"


def test_explicit_missing_file_is_error(tmp_path):
    with pytest.raises(ConfigError, match="не найден"):
        load_settings(tmp_path / "nope.json")


def test_override_keeps_other_defaults(tmp_path):
    s = load_settings(write(tmp_path, {"think": False, "max_side": 800}))
    assert s.think is False and s.max_side == 800
    assert s.model == Settings().model


def test_utf8_bom_is_accepted(tmp_path):
    # Блокнот и PowerShell на Windows любят дописывать BOM
    p = write(tmp_path, {"prompt": "Ответь коротко"}, encoding="utf-8-sig")
    assert load_settings(p).prompt == "Ответь коротко"


def test_invalid_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="некорректный JSON"):
        load_settings(p)


def test_json_must_be_object(tmp_path):
    with pytest.raises(ConfigError, match="JSON-объект"):
        load_settings(write(tmp_path, ["a", "b"]))


def test_unknown_key_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="modle"):
        load_settings(write(tmp_path, {"modle": "x"}))


def test_underscore_keys_are_comments(tmp_path):
    s = load_settings(write(tmp_path, {"_comment": "что угодно", "model": "m"}))
    assert s.model == "m"


@pytest.mark.parametrize("key,value", [
    ("num_ctx", 0), ("num_ctx", -5), ("num_ctx", "8192"), ("num_ctx", True),
    ("max_side", 1.5), ("request_timeout", None),
    ("model", ""), ("model", "   "), ("model", 5), ("prompt", ["a"]),
    ("think", "yes"), ("think", 1), ("think", None),
])
def test_bad_values_are_rejected(tmp_path, key, value):
    with pytest.raises(ConfigError):
        load_settings(write(tmp_path, {key: value}))


@pytest.mark.parametrize("value", [True, False, "low", "medium", "high"])
def test_valid_think_values(tmp_path, value):
    assert load_settings(write(tmp_path, {"think": value})).think == value


@pytest.mark.parametrize("path", sorted(ROOT.glob("examples/*.json")) + [ROOT / "config.example.json"])
def test_shipped_example_configs_are_valid(path):
    """Примеры из репозитория не должны устаревать вместе с кодом."""
    assert isinstance(load_settings(path), Settings)
