"""
Логика без GUI и хоткеев: подготовка картинки, запрос к Ollama, разбор потока ответа,
состояние окна ответа и защита от дублей хоткея. Всё это покрыто тестами (tests/).
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
THINK_UNSUPPORTED_MSG = "Эта модель не поддерживает режим рассуждения - отвечаю без него…"


class OllamaError(Exception):
    """Ошибка, которую вернул сам сервер Ollama (HTTP-статус и/или текст ошибки)."""

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


def build_payload(image_b64: str, settings: Settings, think: Think) -> dict:
    """Собирает тело запроса к /api/chat."""
    return {
        "model": settings.model,
        "stream": True,
        "think": think,
        "keep_alive": settings.keep_alive,
        "options": {"num_ctx": settings.num_ctx},
        "messages": [{"role": "user", "content": settings.prompt, "images": [image_b64]}],
    }


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


def _error_text(resp) -> str:
    """Достаёт текст ошибки из ответа Ollama ({"error": "..."} или сырое тело)."""
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("error"):
            return str(body["error"])
    except Exception:
        pass
    return resp.text or f"HTTP {resp.status_code}"


def _is_think_unsupported(err: OllamaError) -> bool:
    """Модель не умеет рассуждать: Ollama отвечает 400 с упоминанием thinking."""
    return err.status == 400 and "think" in str(err).lower()


def friendly_error(exc: Exception, settings: Settings) -> str:
    """Превращает исключение в понятное пользователю сообщение с подсказкой, что делать."""
    # ConnectTimeout наследует и ConnectionError, и Timeout - ConnectionError проверяем первым
    if isinstance(exc, requests.ConnectionError):
        return (f"Не удаётся подключиться к Ollama ({settings.ollama_url}). "
                "Запущена ли Ollama? Проверьте командой: ollama list")
    if isinstance(exc, requests.Timeout):
        return (f"Ollama не отвечает больше {settings.request_timeout} с. Модель могла "
                "загружаться или слишком долго рассуждать: попробуйте ещё раз или отключите think.")
    if isinstance(exc, OllamaError):
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
    Один запрос к Ollama; события кладёт в очередь.
    Возвращает True, если поток дочитан, и False, если запрос оборван как устаревший.
    """
    payload = build_payload(b64, settings, think)
    with post(settings.ollama_url, json=payload, stream=True,
              timeout=settings.request_timeout) as resp:
        if resp.status_code >= 400:
            raise OllamaError(_error_text(resp), resp.status_code)
        for kind, val in iter_events(resp.iter_lines()):
            if not is_current():          # пришёл новый запрос - выходим, соединение закроется
                return False
            out_q.put((rid, kind, val))
    return True


def ask_model(img: Image.Image, rid: int, out_q, is_current: Callable[[], bool],
              settings: Settings, post=requests.post) -> None:
    """
    Отправляет снимок в Ollama и кладёт события (rid, kind, value) в очередь out_q.
    Работает в отдельном потоке. Виды событий: think, chunk, notice, error, done.
    Если модель не поддерживает think - один раз повторяет запрос без рассуждения.
    post вынесен параметром, чтобы в тестах подменять сетевой вызов.
    """
    try:
        b64 = prepare_image_b64(img, settings.max_side)
        try:
            completed = _stream(b64, settings.think, rid, out_q, is_current, settings, post)
        except OllamaError as e:
            if settings.think is not False and _is_think_unsupported(e):
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

    def __init__(self) -> None:
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
            return None if self.got_chunk else ("set", EMPTY_ANSWER_MSG)
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
