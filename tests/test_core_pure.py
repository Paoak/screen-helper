"""Тесты чистой логики: картинка, payload, разбор потока, ошибки, рамка выделения, состояние окна."""
import base64
import io
import json

import pytest
import requests
from PIL import Image

from screen_helper.config import Settings
from screen_helper.core import (
    EMPTY_ANSWER_MSG,
    AnswerView,
    HotkeyGate,
    OllamaError,
    build_payload,
    friendly_error,
    iter_events,
    prepare_image_b64,
    selection_to_bbox,
)


def decode(b64: str) -> Image.Image:
    """Обратно превращает base64 в картинку (чтобы проверять результат prepare_image_b64)."""
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def line(**kw) -> bytes:
    """Одна строка NDJSON-потока Ollama."""
    return json.dumps(kw, ensure_ascii=False).encode()


# ------------------------------------------------------------------ prepare_image_b64

def test_big_image_is_downscaled_keeping_aspect_ratio():
    img = Image.new("RGB", (4000, 2000), "white")
    out = decode(prepare_image_b64(img, 1000))
    assert out.format == "JPEG"
    assert out.size == (1000, 500)


def test_original_image_is_not_modified():
    img = Image.new("RGB", (4000, 2000), "white")
    prepare_image_b64(img, 1000)
    assert img.size == (4000, 2000)


def test_small_image_is_not_upscaled():
    out = decode(prepare_image_b64(Image.new("RGB", (100, 50)), 1600))
    assert out.size == (100, 50)


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_other_color_modes_are_converted(mode):
    # ImageGrab может вернуть не RGB; JPEG прозрачность не поддерживает
    out = decode(prepare_image_b64(Image.new(mode, (64, 64)), 1600))
    assert out.mode == "RGB"


# ------------------------------------------------------------------ build_payload

def test_payload_shape():
    s = Settings(model="m:1b", num_ctx=4096, keep_alive="5m", prompt="Вопрос?")
    p = build_payload("QUJD", s, True)
    assert p["model"] == "m:1b"
    assert p["stream"] is True
    assert p["think"] is True
    assert p["keep_alive"] == "5m"
    assert p["options"] == {"num_ctx": 4096}
    assert p["messages"] == [{"role": "user", "content": "Вопрос?", "images": ["QUJD"]}]


@pytest.mark.parametrize("think", [True, False, "high"])
def test_payload_passes_think_through(think):
    assert build_payload("x", Settings(), think)["think"] == think


# ------------------------------------------------------------------ iter_events

def test_thinking_is_counted_cumulatively_and_content_streams():
    lines = [
        line(message={"thinking": "abc"}),
        line(message={"thinking": "de"}),
        line(message={"content": "Отв"}),
        line(message={"content": "ет"}, done=True),
    ]
    assert list(iter_events(lines)) == [
        ("think", 3), ("think", 5), ("chunk", "Отв"), ("chunk", "ет"),
    ]


def test_empty_and_garbage_lines_are_skipped():
    lines = [b"", None, b"   ", b"not json", b"[1, 2]", line(message={"content": "ok"})]
    assert list(iter_events(lines)) == [("chunk", "ok")]


def test_null_message_and_null_fields_do_not_crash():
    lines = [line(message=None), line(message={"content": None, "thinking": None})]
    assert list(iter_events(lines)) == []


def test_stops_at_done():
    lines = [line(message={"content": "a"}, done=True), line(message={"content": "после done"})]
    assert list(iter_events(lines)) == [("chunk", "a")]


def test_error_line_raises():
    with pytest.raises(OllamaError, match="boom"):
        list(iter_events([line(error="boom")]))


def test_str_lines_are_accepted_too():
    assert list(iter_events(['{"message": {"content": "x"}}'])) == [("chunk", "x")]


# ------------------------------------------------------------------ friendly_error

def test_connection_error_hint():
    msg = friendly_error(requests.ConnectionError("refused"), Settings())
    assert "Не удаётся подключиться" in msg and "localhost:11434" in msg


def test_connect_timeout_counts_as_connection_problem():
    msg = friendly_error(requests.ConnectTimeout("t"), Settings())
    assert "Не удаётся подключиться" in msg


def test_read_timeout_hint():
    msg = friendly_error(requests.ReadTimeout("t"), Settings(request_timeout=42))
    assert "42" in msg and "не отвечает" in msg


def test_model_not_found_suggests_pull():
    msg = friendly_error(OllamaError("model 'x' not found", 404), Settings(model="x:1b"))
    assert "ollama pull x:1b" in msg


def test_out_of_memory_hint():
    msg = friendly_error(OllamaError("requires more system memory", 500), Settings())
    assert "num_ctx" in msg


def test_other_ollama_error_is_passed_through():
    assert "invalid image" in friendly_error(OllamaError("invalid image", 400), Settings())


def test_unknown_exception_falls_back_to_text_or_class_name():
    assert friendly_error(ValueError("плохо"), Settings()) == "плохо"
    assert friendly_error(KeyError(), Settings()) == "KeyError"


# ------------------------------------------------------------------ selection_to_bbox

SIZE = (1920, 1080)


def test_normal_selection():
    assert selection_to_bbox(10, 20, 110, 220, 1, 1, SIZE) == (10, 20, 110, 220)


def test_reversed_drag_gives_same_box():
    assert selection_to_bbox(110, 220, 10, 20, 1, 1, SIZE) == (10, 20, 110, 220)


def test_dpi_scaling_is_applied():
    # экран 1920x1080 в координатах tkinter, снимок 2880x1620 (масштаб 150%)
    assert selection_to_bbox(100, 100, 300, 200, 1.5, 1.5, (2880, 1620)) == (150, 150, 450, 300)


def test_tiny_selection_is_a_misclick():
    assert selection_to_bbox(5, 5, 10, 10, 1, 1, SIZE) is None
    assert selection_to_bbox(50, 50, 50, 300, 1, 1, SIZE) is None  # нулевая ширина


def test_minimum_size_boundary():
    assert selection_to_bbox(0, 0, 10, 100, 1, 1, SIZE) is None      # ровно 10 - ещё мало
    assert selection_to_bbox(0, 0, 11, 100, 1, 1, SIZE) == (0, 0, 11, 100)


def test_selection_outside_screen_is_clamped():
    assert selection_to_bbox(-50, -50, 5000, 5000, 1, 1, SIZE) == (0, 0, 1920, 1080)


# ------------------------------------------------------------------ AnswerView

def test_new_request_shows_thinking_and_bumps_id():
    v = AnswerView()
    rid, action = v.new_request()
    assert rid == 1 and action == ("set", "Думаю…")
    assert v.new_request()[0] == 2


def test_think_progress_then_first_chunk_replaces_it():
    v = AnswerView()
    rid, _ = v.new_request()
    assert v.handle(rid, "think", 120) == ("set", "Думаю… (120 симв. рассуждения)")
    assert v.handle(rid, "chunk", "Ответ") == ("set", "Ответ")
    assert v.handle(rid, "chunk", " B") == ("append", " B")


def test_late_think_events_do_not_overwrite_the_answer():
    v = AnswerView()
    rid, _ = v.new_request()
    v.handle(rid, "chunk", "A")
    assert v.handle(rid, "think", 999) is None
    assert v.handle(rid, "notice", "что-то") is None


def test_stale_request_events_are_ignored():
    v = AnswerView()
    old, _ = v.new_request()
    v.new_request()
    for kind in ("think", "chunk", "error", "done", "notice"):
        assert v.handle(old, kind, "x") is None


def test_error_before_answer_replaces_text():
    v = AnswerView()
    rid, _ = v.new_request()
    assert v.handle(rid, "error", "нет связи") == ("set", "Ошибка: нет связи")


def test_error_after_partial_answer_keeps_the_answer():
    v = AnswerView()
    rid, _ = v.new_request()
    v.handle(rid, "chunk", "Частичный ответ")
    action = v.handle(rid, "error", "обрыв")
    assert action[0] == "append" and "обрыв" in action[1]


def test_done_without_any_chunk_explains_what_happened():
    v = AnswerView()
    rid, _ = v.new_request()
    assert v.handle(rid, "done", None) == ("set", EMPTY_ANSWER_MSG)


def test_done_after_answer_changes_nothing():
    v = AnswerView()
    rid, _ = v.new_request()
    v.handle(rid, "chunk", "ok")
    assert v.handle(rid, "done", None) is None


def test_notice_is_shown_before_answer():
    v = AnswerView()
    rid, _ = v.new_request()
    assert v.handle(rid, "notice", "без think") == ("set", "без think")


def test_unknown_event_kind_is_ignored():
    v = AnswerView()
    rid, _ = v.new_request()
    assert v.handle(rid, "weird", 1) is None


# ------------------------------------------------------------------ HotkeyGate

def test_key_repeat_is_debounced():
    g = HotkeyGate(min_interval=1.0)
    assert g.accept(10.0) is True
    assert g.accept(10.03) is False      # автоповтор зажатой клавиши
    assert g.accept(10.9) is False
    assert g.accept(11.1) is True        # прошла секунда - новое нажатие


def test_events_queued_during_capture_are_stale():
    g = HotkeyGate()
    queued_during_capture = 5.0
    g.mark_done(7.0)                     # захват завершился в 7.0
    assert g.is_stale(queued_during_capture) is True
    assert g.is_stale(7.0) is True       # граница считается устаревшим
    assert g.is_stale(8.0) is False      # новое нажатие после захвата - обычное


def test_first_event_is_never_stale():
    assert HotkeyGate().is_stale(0.0) is False
