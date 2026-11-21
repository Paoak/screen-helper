"""Тесты ask_model: сетевой вызов подменён, Ollama и GUI не нужны."""
import base64
import json
import queue
from dataclasses import replace

import pytest
import requests
from PIL import Image

from screen_helper.config import Settings
from screen_helper.core import EMPTY_ANSWER_MSG, THINK_UNSUPPORTED_MSG, AnswerView, ask_model


class FakeResponse:
    """Минимальная замена requests.Response для потокового запроса."""

    def __init__(self, lines=(), status=200, body=""):
        self.status_code = status
        self._lines = list(lines)
        self.text = body
        self.closed = False

    def json(self):
        return json.loads(self.text)

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


class FakePost:
    """Подмена requests.post: отдаёт заготовленные ответы по очереди и запоминает вызовы."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, json=None, stream=None, timeout=None):
        self.calls.append({"url": url, "json": json, "stream": stream, "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def line(**kw):
    return json.dumps(kw, ensure_ascii=False).encode()


def error_response(status, message):
    return FakeResponse(status=status, body=json.dumps({"error": message}))


def run(post, settings=None, is_current=lambda: True):
    """Запускает ask_model с фейковым post и возвращает все события из очереди."""
    q = queue.Queue()
    img = Image.new("RGB", (300, 200), "white")
    ask_model(img, 1, q, is_current, settings or Settings(), post=post)
    events = []
    while not q.empty():
        events.append(q.get())
    return events


def kinds(events):
    return [e[1] for e in events]


# ------------------------------------------------------------------ happy path

def test_successful_stream_emits_think_chunks_and_done():
    post = FakePost(FakeResponse([
        line(message={"thinking": "ab"}),
        line(message={"thinking": "cd"}),
        line(message={"content": "Отв"}),
        line(message={"content": "ет"}, done=True),
    ]))
    assert run(post) == [
        (1, "think", 2), (1, "think", 4), (1, "chunk", "Отв"), (1, "chunk", "ет"), (1, "done", None),
    ]


def test_request_is_built_from_settings():
    s = Settings(model="m:1b", ollama_url="http://h:1/api/chat", request_timeout=7, think="low")
    post = FakePost(FakeResponse([line(done=True)]))
    run(post, s)
    call = post.calls[0]
    assert call["url"] == "http://h:1/api/chat"
    assert call["timeout"] == 7 and call["stream"] is True
    assert call["json"]["model"] == "m:1b"
    assert call["json"]["think"] == "low"
    image = call["json"]["messages"][0]["images"][0]
    assert base64.b64decode(image)[:2] == b"\xff\xd8"      # JPEG


def test_think_false_is_sent_as_false():
    post = FakePost(FakeResponse([line(done=True)]))
    run(post, replace(Settings(), think=False))
    assert post.calls[0]["json"]["think"] is False


def test_stream_without_any_content_still_finishes_with_done():
    events = run(FakePost(FakeResponse([line(message={"thinking": "x"}, done=True)])))
    assert kinds(events) == ["think", "done"]


def test_connection_is_closed_after_normal_finish():
    resp = FakeResponse([line(message={"content": "a"}, done=True)])
    run(FakePost(resp))
    assert resp.closed


# ------------------------------------------------------------------ отмена устаревшего запроса

def test_stale_request_stops_streaming_and_sends_no_done():
    resp = FakeResponse([line(message={"content": "1"}), line(message={"content": "2"}),
                         line(message={"content": "3"}, done=True)])
    answers = iter([True, False])                 # после первого куска пришёл новый запрос
    events = run(FakePost(resp), is_current=lambda: next(answers, False))
    assert events == [(1, "chunk", "1")]
    assert resp.closed                            # соединение закрыто -> Ollama прекращает генерацию


# ------------------------------------------------------------------ ошибки

def test_model_not_found_gives_pull_hint():
    post = FakePost(error_response(404, "model 'gemma4:12b' not found"))
    events = run(post)
    assert kinds(events) == ["error"]
    assert "ollama pull gemma4:12b" in events[0][2]


def test_out_of_memory_error():
    events = run(FakePost(error_response(500, "model requires more system memory")))
    assert "Не хватает памяти" in events[0][2]


def test_connection_refused():
    events = run(FakePost(requests.ConnectionError("refused")))
    assert kinds(events) == ["error"]
    assert "Запущена ли Ollama" in events[0][2]


def test_read_timeout():
    events = run(FakePost(requests.ReadTimeout("slow")))
    assert "не отвечает" in events[0][2]


def test_error_in_the_middle_of_stream_keeps_earlier_chunks_and_skips_done():
    resp = FakeResponse([line(message={"content": "часть"}), line(error="boom")])
    events = run(FakePost(resp))
    assert kinds(events) == ["chunk", "error"]
    assert "boom" in events[1][2]


def test_unrelated_400_is_not_retried():
    post = FakePost(error_response(400, "invalid image data"))
    events = run(post)
    assert len(post.calls) == 1
    assert "invalid image data" in events[0][2]


def test_error_body_that_is_not_json_is_still_reported():
    events = run(FakePost(FakeResponse(status=502, body="Bad Gateway")))
    assert "Bad Gateway" in events[0][2]


def test_broken_image_is_reported_not_raised():
    q = queue.Queue()
    ask_model(object(), 1, q, lambda: True, Settings(), post=FakePost())   # не картинка
    assert q.get()[1] == "error"


# ------------------------------------------------------------------ модель без поддержки think

def test_unsupported_thinking_falls_back_to_plain_request_once():
    post = FakePost(
        error_response(400, '"gemma3:12b" does not support thinking'),
        FakeResponse([line(message={"content": "Ответ"}, done=True)]),
    )
    events = run(post)
    assert kinds(events) == ["notice", "chunk", "done"]
    assert events[0][2] == THINK_UNSUPPORTED_MSG
    assert [c["json"]["think"] for c in post.calls] == [True, False]


def test_no_fallback_when_think_is_already_off():
    post = FakePost(error_response(400, "does not support thinking"))
    events = run(post, replace(Settings(), think=False))
    assert len(post.calls) == 1
    assert kinds(events) == ["error"]


def test_fallback_request_failing_is_reported():
    post = FakePost(error_response(400, "does not support thinking"),
                    error_response(500, "boom"))
    events = run(post)
    assert kinds(events) == ["notice", "error"]
    assert "boom" in events[1][2]


# ------------------------------------------------------------------ вместе с AnswerView

def feed(view, events):
    """Прогоняет события через AnswerView и собирает итоговый текст окна."""
    text = ""
    for rid, kind, val in events:
        action = view.handle(rid, kind, val)
        if action is None:
            continue
        text = action[1] if action[0] == "set" else text + action[1]
    return text


def test_end_to_end_answer_text():
    view = AnswerView()
    view.new_request()
    post = FakePost(FakeResponse([
        line(message={"thinking": "рассуждаю"}),
        line(message={"content": "B) "}),
        line(message={"content": "Ответ"}, done=True),
    ]))
    assert feed(view, run(post)) == "B) Ответ"


def test_end_to_end_empty_answer_is_explained():
    view = AnswerView()
    view.new_request()
    post = FakePost(FakeResponse([line(message={"thinking": "долго думаю"}, done=True)]))
    assert feed(view, run(post)) == EMPTY_ANSWER_MSG


def test_end_to_end_fallback_notice_is_replaced_by_answer():
    view = AnswerView()
    view.new_request()
    post = FakePost(error_response(400, "does not support thinking"),
                    FakeResponse([line(message={"content": "Ответ"}, done=True)]))
    assert feed(view, run(post)) == "Ответ"
