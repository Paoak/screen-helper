"""
Логика без GUI и хоткеев: подготовка картинки, запрос к серверу модели (Ollama или
OpenAI-совместимый llama-server), разбор потока ответа, состояние окна ответа и защита
от дублей хоткея. Всё это покрыто тестами (tests/).
"""
from __future__ import annotations

import base64
import io
import json
import traceback
from typing import Callable, Iterable, Iterator, Optional, Tuple, Union

import requests
from PIL import Image

from .config import Settings

Think = Union[bool, str]
Event = Tuple[str, object]          # ("think", N) / ("chunk", "текст")
Action = Tuple[str, str]            # ("set", "текст") / ("append", "текст")

EMPTY_ANSWER_MSG = (
    "Модель закончила, но не вернула ответ. Возможно, ей не хватило контекста на рассуждение: "
    "увеличьте num_ctx или выключите think в настройках."
)
EMPTY_ANSWER_MSG_OPENAI = (
    "Модель закончила, но не вернула ответ. Скорее всего, рассуждение съело весь лимит: задайте "
    "reasoning_effort \"medium\", увеличьте max_tokens или контекст сервера (-c), либо выключите think."
)
THINK_UNSUPPORTED_MSG = "Эта модель не поддерживает режим рассуждения - отвечаю без него…"


def empty_answer_message(settings: Settings) -> str:
    """Что сказать, если модель закончила без ответа (советы зависят от бэкенда)."""
    return EMPTY_ANSWER_MSG_OPENAI if settings.backend == "openai" else EMPTY_ANSWER_MSG


class OllamaError(Exception):
    """Ошибка, которую вернул сам сервер модели - Ollama или llama-server (статус и/или текст)."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------- картинка и запрос

def prepare_image_b64(img: Image.Image, max_side: int) -> str:
    """Уменьшает копию снимка до max_side по большей стороне и кодирует в base64 (JPEG)."""
    img = img.convert("RGB")        # convert() возвращает копию, оригинал не трогаем
    img.thumbnail((max_side, max_side))  # только уменьшает, маленькие картинки не растягивает
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _sampling(settings: Settings) -> dict:
    """Параметры сэмплинга, которые пользователь задал явно (остальные решает сервер)."""
    pairs = (("temperature", settings.temperature), ("top_p", settings.top_p),
             ("top_k", settings.top_k), ("min_p", settings.min_p))
    return {name: value for name, value in pairs if value is not None}


def _merge(payload: dict, extra: dict) -> dict:
    """Добавляет в запрос поля из extra_body; вложенные объекты (chat_template_kwargs) сливает."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return payload


def build_payload(image_b64: str, settings: Settings, think: Think) -> dict:
    """Собирает тело запроса к Ollama (/api/chat)."""
    options = {"num_ctx": settings.num_ctx, **_sampling(settings)}
    if settings.max_tokens is not None:
        options["num_predict"] = settings.max_tokens      # так в Ollama называется потолок длины
    payload = {
        "model": settings.model,
        "stream": True,
        "think": think,
        "keep_alive": settings.keep_alive,
        "options": options,
        "messages": [{"role": "user", "content": settings.prompt, "images": [image_b64]}],
    }
    return _merge(payload, settings.extra_body)


def build_openai_payload(image_b64: str, settings: Settings, think: Think) -> dict:
    """
    Собирает тело запроса к OpenAI-совместимому серверу (llama-server: /v1/chat/completions).
    Контекст (-c) и выгрузка слоёв на GPU задаются при запуске сервера, а не здесь.
    Рассуждение:
      think=False             - просим не рассуждать;
      reasoning_effort / think="medium" - уровень рассуждения (передаём двумя способами:
                                обычным полем и через chat_template_kwargs - так его видит шаблон модели).
    """
    payload = {
        "model": settings.model,          # llama-server с одной моделью имя не проверяет
        "stream": True,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": settings.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
        **_sampling(settings),
    }
    if settings.max_tokens is not None:
        payload["max_tokens"] = settings.max_tokens

    template_kwargs: dict = {}
    effort = settings.reasoning_effort or (think if isinstance(think, str) else None)
    if think is False:
        template_kwargs["enable_thinking"] = False
        payload["thinking_budget_tokens"] = 0    # на новых сборках llama.cpp: 0 = без рассуждения
    elif effort:
        payload["reasoning_effort"] = effort
        template_kwargs["reasoning_effort"] = effort
    if template_kwargs:
        payload["chat_template_kwargs"] = template_kwargs
    return _merge(payload, settings.extra_body)


def iter_events(lines: Iterable[Union[bytes, str]]) -> Iterator[Event]:
    """
    Разбирает построчный JSON-поток Ollama и выдаёт события:
      ("think", N)     - модель рассуждает, всего N символов рассуждения (сам текст не показываем);
      ("chunk", текст) - очередной кусок итогового ответа.
    Пустые и битые строки пропускаются, строка {"error": ...} бросает OllamaError.
    """
    think_chars = 0
    for raw in lines:
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if "error" in data:
            raise OllamaError(str(data["error"]))
        msg = data.get("message") or {}
        thinking = msg.get("thinking") or ""
        if thinking:
            think_chars += len(thinking)
            yield ("think", think_chars)
        content = msg.get("content") or ""
        if content:
            yield ("chunk", content)
        if data.get("done"):
            break


def _error_message(err) -> str:
    """Текст ошибки: у Ollama это строка, у llama-server - объект {"message": ..., "code": ...}."""
    if isinstance(err, dict) and err.get("message"):
        return str(err["message"])
    return str(err)


def iter_openai_events(lines: Iterable[Union[bytes, str]]) -> Iterator[Event]:
    """
    Разбирает поток SSE от OpenAI-совместимого сервера и выдаёт те же события, что iter_events:
      ("think", N)     - идёт рассуждение (поле reasoning_content), всего N символов;
      ("chunk", текст) - очередной кусок итогового ответа (поле content).
    Строки вида «data: {...}»; «data: [DONE]» - конец. Комментарии, event: и битый JSON пропускаются,
    «data: {"error": ...}» бросает OllamaError. Если сервер проигнорировал stream и прислал один
    обычный JSON-объект, он тоже разбирается.
    """
    think_chars = 0
    for raw in lines:
        if not raw:
            continue
        text = (raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw).strip()
        if text.startswith("data:"):
            text = text[5:].strip()
        elif not text.startswith("{"):
            continue                       # ": keep-alive", "event: ..." и прочее
        if text == "[DONE]":
            break
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("error"):
            raise OllamaError(_error_message(data["error"]))
        choices = data.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if not isinstance(choice, dict):
            continue                       # например, последний кусок со статистикой usage
        delta = choice.get("delta") or choice.get("message") or {}
        if not isinstance(delta, dict):
            continue
        thinking = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(thinking, str) and thinking:
            think_chars += len(thinking)
            yield ("think", think_chars)
        content = delta.get("content")
        if isinstance(content, str) and content:
            yield ("chunk", content)


def _error_text(resp) -> str:
    """Достаёт текст ошибки из ответа сервера (Ollama: {"error": "..."}, llama-server: {"error": {...}})."""
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("error"):
            return _error_message(body["error"])
    except Exception:
        pass
    return resp.text or f"HTTP {resp.status_code}"


def _is_think_unsupported(err: OllamaError) -> bool:
    """Модель не умеет рассуждать: Ollama отвечает 400 с упоминанием thinking."""
    return err.status == 400 and "think" in str(err).lower()


def _openai_error(exc: OllamaError, settings: Settings) -> str:
    """Подсказки для типичных ошибок llama-server."""
    text, low = str(exc), str(exc).lower()
    if "mmproj" in low or "image input is not supported" in low:
        return ("Сервер запущен без модуля зрения, поэтому картинки не принимает. "
                "Запустите llama-server с флагом --mmproj <файл проектора> (или скриптом PrismML, "
                "который подключает его сам).")
    if exc.status == 503 or "loading model" in low:
        return "Сервер ещё загружает модель. Подождите немного и повторите."
    if "context size" in low or "exceed_context" in low:
        return ("Скриншот и рассуждение не поместились в контекст сервера. Запустите llama-server "
                "с большим -c (например -c 16384) или уменьшите max_side в настройках.")
    if exc.status == 404:
        return (f"Сервер ответил 404 по адресу {settings.openai_url}. "
                "Адрес должен заканчиваться на /v1/chat/completions.")
    if "memory" in low or "failed to allocate" in low:
        return (f"Не хватает памяти: {text}. Запустите сервер с меньшим -c, "
                "квантованным KV-кэшем или более лёгкой сборкой модели.")
    return f"Сервер вернул ошибку: {text}"


def friendly_error(exc: Exception, settings: Settings) -> str:
    """Превращает исключение в понятное пользователю сообщение с подсказкой, что делать."""
    openai = settings.backend == "openai"
    # ConnectTimeout наследует и ConnectionError, и Timeout - ConnectionError проверяем первым
    if isinstance(exc, requests.ConnectionError):
        if openai:
            return (f"Не удаётся подключиться к серверу ({settings.openai_url}). Запущен ли llama-server? "
                    "Адрес и порт должны совпадать с --host/--port сервера (по умолчанию 8080).")
        return (f"Не удаётся подключиться к Ollama ({settings.ollama_url}). "
                "Запущена ли Ollama? Проверьте командой: ollama list")
    if isinstance(exc, requests.Timeout):
        if openai:
            return (f"Сервер не отвечает больше {settings.request_timeout} с. Модель могла загружаться "
                    "или слишком долго рассуждать: попробуйте reasoning_effort \"medium\" или отключите think.")
        return (f"Ollama не отвечает больше {settings.request_timeout} с. Модель могла "
                "загружаться или слишком долго рассуждать: попробуйте ещё раз или отключите think.")
    if isinstance(exc, OllamaError):
        if openai:
            return _openai_error(exc, settings)
        text = str(exc)
        if exc.status == 404 or "not found" in text.lower():
            return (f"Модель «{settings.model}» не найдена. "
                    f"Скачайте её командой: ollama pull {settings.model}")
        if "memory" in text.lower():
            return (f"Не хватает памяти для модели: {text}. "
                    "Уменьшите num_ctx или выберите модель поменьше.")
        return f"Ollama вернула ошибку: {text}"
    return str(exc) or exc.__class__.__name__


def _stream(b64: str, think: Think, rid: int, out_q, is_current: Callable[[], bool],
            settings: Settings, post) -> bool:
    """
    Один запрос к серверу модели (Ollama или llama-server); события кладёт в очередь.
    Возвращает True, если поток дочитан, и False, если запрос оборван как устаревший.
    """
    openai = settings.backend == "openai"
    payload = (build_openai_payload if openai else build_payload)(b64, settings, think)
    events = iter_openai_events if openai else iter_events
    with post(settings.api_url, json=payload, stream=True,
              timeout=settings.request_timeout) as resp:
        if resp.status_code >= 400:
            raise OllamaError(_error_text(resp), resp.status_code)
        for kind, val in events(resp.iter_lines()):
            if not is_current():          # пришёл новый запрос - выходим, соединение закроется
                return False
            out_q.put((rid, kind, val))
    return True


def ask_model(img: Image.Image, rid: int, out_q, is_current: Callable[[], bool],
              settings: Settings, post=requests.post) -> None:
    """
    Отправляет снимок на сервер модели и кладёт события (rid, kind, value) в очередь out_q.
    Работает в отдельном потоке. Виды событий: think, chunk, notice, error, done.
    Ollama: если модель не поддерживает think - один раз повторяет запрос без рассуждения.
    post вынесен параметром, чтобы в тестах подменять сетевой вызов.
    """
    try:
        b64 = prepare_image_b64(img, settings.max_side)
        try:
            completed = _stream(b64, settings.think, rid, out_q, is_current, settings, post)
        except OllamaError as e:
            if (settings.backend == "ollama" and settings.think is not False
                    and _is_think_unsupported(e)):
                out_q.put((rid, "notice", THINK_UNSUPPORTED_MSG))
                completed = _stream(b64, False, rid, out_q, is_current, settings, post)
            else:
                raise
        if completed:
            out_q.put((rid, "done", None))
    except Exception as e:
        traceback.print_exc()             # подробности - в консоль
        out_q.put((rid, "error", friendly_error(e, settings)))


# ---------------------------------------------------------------- состояние окна ответа

class AnswerView:
    """
    Решает, что показывать в окне ответа, по событиям от потока с моделью.
    Не зависит от tkinter: возвращает «действия» ("set", текст) или ("append", текст).
    """

    def __init__(self, empty_msg: str = EMPTY_ANSWER_MSG) -> None:
        self.empty_msg = empty_msg    # что показать, если ответа не было
        self.rid = 0                  # номер текущего запроса
        self.got_chunk = False        # пришёл ли уже хоть один кусок ответа

    def new_request(self) -> Tuple[int, Action]:
        """Начинает новый запрос: старые ответы с этого момента игнорируются."""
        self.rid += 1
        self.got_chunk = False
        return self.rid, ("set", "Думаю…")

    def is_current(self, rid: int) -> bool:
        return rid == self.rid

    def handle(self, rid: int, kind: str, val) -> Optional[Action]:
        """Обрабатывает событие; None - окно менять не нужно."""
        if rid != self.rid:
            return None               # ответ от устаревшего запроса
        if kind == "notice":
            return None if self.got_chunk else ("set", str(val))
        if kind == "think":
            return None if self.got_chunk else ("set", f"Думаю… ({val} симв. рассуждения)")
        if kind == "chunk":
            if not self.got_chunk:    # первый кусок заменяет «Думаю…»
                self.got_chunk = True
                return ("set", str(val))
            return ("append", str(val))
        if kind == "error":
            if self.got_chunk:        # ответ уже частично показан - не стираем его
                return ("append", f"\n\n[Ошибка: {val}]")
            return ("set", f"Ошибка: {val}")
        if kind == "done":
            return None if self.got_chunk else ("set", self.empty_msg)
        return None


# ---------------------------------------------------------------- хоткеи и выделение области

class HotkeyGate:
    """
    Защита от дублей хоткея: автоповтор зажатой клавиши и события,
    накопившиеся в очереди, пока шёл предыдущий захват.
    Время передаётся снаружи, поэтому легко тестируется.
    """

    def __init__(self, min_interval: float = 1.0) -> None:
        self.min_interval = min_interval
        self.last_accepted = float("-inf")   # когда последний раз приняли нажатие
        self.last_end = float("-inf")        # когда закончился последний захват

    def accept(self, now: float) -> bool:
        """Вызывается при нажатии. True - нажатие принимаем, False - это повтор."""
        if now - self.last_accepted < self.min_interval:
            return False
        self.last_accepted = now
        return True

    def mark_done(self, now: float) -> None:
        """Отмечает окончание захвата (после него устаревшие события отбрасываются)."""
        self.last_end = now

    def is_stale(self, event_time: float) -> bool:
        """True, если событие возникло до конца последнего захвата - его надо пропустить."""
        return event_time <= self.last_end


def selection_to_bbox(x0: float, y0: float, x1: float, y1: float, fx: float, fy: float,
                      img_size: Tuple[int, int], min_size: int = 10
                      ) -> Optional[Tuple[int, int, int, int]]:
    """
    Переводит выделение мышкой (координаты экрана) в рамку для обрезки снимка.
    fx, fy - масштаб экран -> снимок (на случай DPI-масштабирования).
    Выход за края экрана обрезается; слишком маленькая рамка (случайный клик) даёт None.
    """
    width, height = img_size
    sw, sh = width / fx, height / fy      # размеры экрана в координатах tkinter
    lx, rx = sorted((min(max(x0, 0), sw), min(max(x1, 0), sw)))
    ty, by = sorted((min(max(y0, 0), sh), min(max(y1, 0), sh)))
    if (rx - lx) <= min_size or (by - ty) <= min_size:
        return None
    return int(lx * fx), int(ty * fy), int(rx * fx), int(by * fy)