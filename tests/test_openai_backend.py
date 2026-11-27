"""Тесты бэкенда «openai» (llama-server): запрос, разбор SSE, ошибки, выбор адреса. Сеть подменена."""
import json
import queue
from dataclasses import replace

import pytest
import requests
from PIL import Image

from screen_helper.config import ConfigError, Settings, load_settings
from screen_helper.core import (
    EMPTY_ANSWER_MSG,
    EMPTY_ANSWER_MSG_OPENAI,
    AnswerView,
    OllamaError,
    ask_model,
    build_openai_payload,
    build_payload,
    empty_answer_message,
    friendly_error,
    iter_openai_events,
)

OPENAI = Settings(backend="openai", model="bonsai")


class FakeResponse:
    """Минимальная замена requests.Response для потокового запроса."""

    def __init__(self, lines=(), status=200, body=""):
        self.status_code = status
        self._lines = list(lines)
        self.text = body

    def json(self):
        return json.loads(self.text)

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class FakePost:
    """Подмена requests.post: отдаёт заготовленные ответы по очереди и запоминает вызовы."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, json=None, stream=None, timeout=None):
        self.calls.append({"url": url, "json": json, "stream": stream, "timeout": timeout})
        return self.responses.pop(0)


def kinds(events):
    return [e[1] for e in events]


def sse(**kw) -> bytes:
    """Одна строка SSE: «data: {...}»."""
    return b"data: " + json.dumps(kw, ensure_ascii=False).encode()


def delta(content=None, reasoning=None, finish=None) -> bytes:
    """Кусок потока llama-server с дельтой ответа или рассуждения."""
    d = {}
    if content is not None:
        d["content"] = content
    if reasoning is not None:
        d["reasoning_content"] = reasoning
    return sse(choices=[{"index": 0, "delta": d, "finish_reason": finish}])


def run(post, settings=OPENAI, is_current=lambda: True):
    q = queue.Queue()
    ask_model(Image.new("RGB", (300, 200), "white"), 1, q, is_current, settings, post=post)
    events = []
    while not q.empty():
        events.append(q.get())
    return events


# ------------------------------------------------------------------ запрос

def test_payload_has_text_and_image_parts():
    p = build_openai_payload("QUJD", OPENAI, True)
    assert p["stream"] is True and p["model"] == "bonsai"
    parts = p["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": OPENAI.prompt}
    assert parts[1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}


def test_payload_has_no_optional_fields_by_default():
    p = build_openai_payload("x", OPENAI, True)
    for key in ("reasoning_effort", "chat_template_kwargs", "temperature", "top_p", "top_k",
                "min_p", "max_tokens", "thinking_budget_tokens", "keep_alive", "options"):
        assert key not in p


def test_reasoning_effort_is_sent_both_ways():
    p = build_openai_payload("x", replace(OPENAI, reasoning_effort="medium"), True)
    assert p["reasoning_effort"] == "medium"
    assert p["chat_template_kwargs"] == {"reasoning_effort": "medium"}


def test_think_level_is_used_when_reasoning_effort_not_set():
    p = build_openai_payload("x", OPENAI, "high")
    assert p["reasoning_effort"] == "high"


def test_explicit_reasoning_effort_beats_think_level():
    p = build_openai_payload("x", replace(OPENAI, reasoning_effort="xhigh"), "low")
    assert p["reasoning_effort"] == "xhigh"


def test_think_false_disables_thinking_and_drops_effort():
    p = build_openai_payload("x", replace(OPENAI, reasoning_effort="medium"), False)
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["thinking_budget_tokens"] == 0
    assert "reasoning_effort" not in p


def test_sampling_and_max_tokens_are_sent_only_when_set():
    s = replace(OPENAI, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, max_tokens=8192)
    p = build_openai_payload("x", s, True)
    assert (p["temperature"], p["top_p"], p["top_k"], p["min_p"], p["max_tokens"]) == (1.0, 0.95, 20, 0.0, 8192)


def test_zero_sampling_values_are_not_dropped():
    """0 - допустимое значение (min_p=0), его нельзя путать с «не задано»."""
    p = build_openai_payload("x", replace(OPENAI, min_p=0.0, temperature=0.0), True)
    assert p["min_p"] == 0.0 and p["temperature"] == 0.0


def test_extra_body_overrides_and_merges_nested():
    s = replace(OPENAI, reasoning_effort="medium",
                extra_body={"cache_prompt": False, "chat_template_kwargs": {"foo": 1}, "temperature": 0.3})
    p = build_openai_payload("x", s, True)
    assert p["cache_prompt"] is False and p["temperature"] == 0.3
    assert p["chat_template_kwargs"] == {"reasoning_effort": "medium", "foo": 1}


def test_ollama_payload_gets_sampling_in_options():
    s = replace(Settings(), temperature=1.0, top_k=20, max_tokens=4096)
    opts = build_payload("x", s, True)["options"]
    assert opts == {"num_ctx": 16384, "temperature": 1.0, "top_k": 20, "num_predict": 4096}


def test_ollama_payload_unchanged_by_default():
    assert build_payload("x", Settings(), True)["options"] == {"num_ctx": 16384}


# ------------------------------------------------------------------ разбор потока

def test_reasoning_is_counted_and_content_streams():
    lines = [delta(reasoning="abc"), delta(reasoning="de"), delta(content="О"), delta(content="К"),
             delta(finish="stop"), b"data: [DONE]"]
    assert list(iter_openai_events(lines)) == [("think", 3), ("think", 5), ("chunk", "О"), ("chunk", "К")]


def test_alternative_reasoning_field_name_is_accepted():
    line = sse(choices=[{"delta": {"reasoning": "xyz"}}])
    assert list(iter_openai_events([line])) == [("think", 3)]


def test_noise_lines_are_skipped():
    lines = [b"", b": keep-alive", b"event: ping", b"data: not json", b"data: [1, 2]",
             sse(choices=[]), sse(usage={"total_tokens": 5}), delta(content="ok")]
    assert list(iter_openai_events(lines)) == [("chunk", "ok")]


def test_null_fields_do_not_crash():
    lines = [sse(choices=[{"delta": None}]), sse(choices=[{"delta": {"content": None, "reasoning_content": None}}]),
             sse(choices=[None]), sse(choices="oops")]
    assert list(iter_openai_events(lines)) == []


def test_stops_at_done_marker():
    lines = [delta(content="a"), b"data: [DONE]", delta(content="after")]
    assert list(iter_openai_events(lines)) == [("chunk", "a")]


def test_str_lines_and_data_without_space():
    lines = ['data:' + json.dumps({"choices": [{"delta": {"content": "x"}}]})]
    assert list(iter_openai_events(lines)) == [("chunk", "x")]


def test_non_stream_json_reply_is_understood():
    body = json.dumps({"choices": [{"message": {"content": "целиком"}}]}).encode()
    assert list(iter_openai_events([body])) == [("chunk", "целиком")]


def test_error_event_raises_with_message():
    with pytest.raises(OllamaError, match="boom"):
        list(iter_openai_events([sse(error={"message": "boom", "code": 500})]))


# ------------------------------------------------------------------ ask_model

def test_full_stream_goes_to_queue():
    post = FakePost(FakeResponse([delta(reasoning="hm"), delta(content="Ответ: B"), b"data: [DONE]"]))
    events = run(post)
    assert kinds(events) == ["think", "chunk", "done"]
    call = post.calls[0]
    assert call["url"] == OPENAI.openai_url and call["stream"] is True
    assert call["json"]["messages"][0]["content"][1]["type"] == "image_url"


def test_ollama_url_is_not_used_for_openai_backend():
    post = FakePost(FakeResponse([delta(content="x")]))
    run(post, replace(OPENAI, openai_url="http://127.0.0.1:9/v1/chat/completions"))
    assert post.calls[0]["url"] == "http://127.0.0.1:9/v1/chat/completions"


def test_http_error_object_is_unwrapped():
    body = json.dumps({"error": {"code": 400, "message": "image input is not supported - hint: mmproj", "type": "x"}})
    events = run(FakePost(FakeResponse(status=400, body=body)))
    assert kinds(events) == ["error"]
    assert "--mmproj" in events[0][2]


def test_think_400_is_not_retried_for_openai_backend():
    """Откат «без think» - особенность Ollama; у llama-server повтор не нужен."""
    body = json.dumps({"error": {"message": "think is wrong"}})
    post = FakePost(FakeResponse(status=400, body=body))
    events = run(post)
    assert len(post.calls) == 1 and kinds(events) == ["error"]


def test_stale_request_stops_openai_stream():
    post = FakePost(FakeResponse([delta(content="a"), delta(content="b")]))
    flags = iter([True, False])
    events = run(post, is_current=lambda: next(flags, False))
    assert kinds(events) == ["chunk"]


# ------------------------------------------------------------------ понятные ошибки

@pytest.mark.parametrize("exc, expected", [
    (requests.ConnectionError("x"), "llama-server"),
    (requests.ReadTimeout("x"), "reasoning_effort"),
    (OllamaError("Loading model", 503), "загружает модель"),
    (OllamaError("the request exceeds the available context size, try increasing it", 400), "-c"),
    (OllamaError("File Not Found", 404), "/v1/chat/completions"),
    (OllamaError("failed to allocate buffer", 500), "памяти"),
    (OllamaError("weird", 500), "weird"),
])
def test_openai_friendly_errors(exc, expected):
    assert expected in friendly_error(exc, OPENAI)


def test_ollama_errors_still_mention_ollama():
    assert "Ollama" in friendly_error(requests.ConnectionError("x"), Settings())


def test_empty_answer_message_depends_on_backend():
    assert empty_answer_message(Settings()) == EMPTY_ANSWER_MSG
    assert empty_answer_message(OPENAI) == EMPTY_ANSWER_MSG_OPENAI
    view = AnswerView(EMPTY_ANSWER_MSG_OPENAI)
    rid, _ = view.new_request()
    assert view.handle(rid, "done", None) == ("set", EMPTY_ANSWER_MSG_OPENAI)


# ------------------------------------------------------------------ настройки

def write(tmp_path, data):
    p = tmp_path / "config.gemma.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_api_url_follows_backend():
    assert Settings().api_url == Settings().ollama_url
    assert OPENAI.api_url == OPENAI.openai_url


def test_new_fields_load_from_file(tmp_path):
    s = load_settings(write(tmp_path, {
        "backend": "openai", "reasoning_effort": "medium", "temperature": 1, "top_p": 0.95,
        "top_k": 20, "min_p": 0, "max_tokens": 8192, "extra_body": {"cache_prompt": True}}))
    assert s.backend == "openai" and s.reasoning_effort == "medium"
    assert (s.temperature, s.top_p, s.top_k, s.min_p, s.max_tokens) == (1, 0.95, 20, 0, 8192)
    assert s.extra_body == {"cache_prompt": True}


def test_null_means_not_set(tmp_path):
    s = load_settings(write(tmp_path, {"reasoning_effort": None, "temperature": None, "max_tokens": None}))
    assert s.reasoning_effort is None and s.temperature is None and s.max_tokens is None


@pytest.mark.parametrize("key, value", [
    ("backend", "llama"), ("backend", 1),
    ("reasoning_effort", ""), ("reasoning_effort", 5), ("reasoning_effort", True),
    ("temperature", "hot"), ("temperature", True), ("temperature", -0.1),
    ("top_p", 1.5), ("top_p", -0.1), ("min_p", 2),
    ("top_k", 0), ("top_k", 2.5), ("top_k", True),
    ("max_tokens", 0), ("max_tokens", "many"),
    ("extra_body", [1]), ("extra_body", "x"),
])
def test_bad_new_values_are_rejected(tmp_path, key, value):
    with pytest.raises(ConfigError, match=key):
        load_settings(write(tmp_path, {key: value}))