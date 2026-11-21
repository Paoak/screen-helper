"""Приложение: глобальные хоткеи, захват экрана, запросы к модели и окно ответа."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
import traceback
from typing import Optional

import keyboard
from PIL import ImageGrab

from .config import Settings
from .core import Action, AnswerView, HotkeyGate, ask_model
from .ui import Overlay, enable_dpi_awareness, select_area


class App:
    """Склеивает всё: хоткеи -> очередь событий -> захват -> модель -> окно ответа."""

    def __init__(self, settings: Settings):
        self.s = settings
        enable_dpi_awareness()            # до создания окон tkinter
        self.root = tk.Tk()
        self.root.withdraw()              # главное окно не нужно, живём на Toplevel
        self.overlay = Overlay(self.root)
        self.q: "queue.Queue" = queue.Queue()  # сюда пишут и хоткеи, и поток с моделью
        self.view = AnswerView()          # что показывать в окне ответа
        self.gate = HotkeyGate()          # защита от дублей хоткея

        # Из потока keyboard в tkinter лезть нельзя, поэтому шлём события в очередь
        keyboard.add_hotkey(settings.hotkey_full, lambda: self.on_hotkey(False))
        keyboard.add_hotkey(settings.hotkey_area, lambda: self.on_hotkey(True))
        keyboard.add_hotkey(settings.hotkey_hide, lambda: self.q.put((0, "toggle", None)))
        keyboard.add_hotkey(settings.hotkey_quit, lambda: self.q.put((0, "quit", None)))

        self.root.after(50, self.poll)

    def on_hotkey(self, use_area: bool) -> None:
        """Колбэк хоткея (поток keyboard): отсекаем автоповтор и шлём событие в очередь."""
        now = time.monotonic()
        if not self.gate.accept(now):
            return
        print("hotkey:", "область" if use_area else "экран")  # для отладки в консоли
        self.q.put((0, "capture", (use_area, now)))

    def capture(self, use_area: bool) -> None:
        """Прячем окно ответа, делаем снимок, (опционально) выбираем область, шлём в модель."""
        was_visible = self.overlay.is_visible()  # чтобы вернуть окно на место, если выбор отменят
        try:
            self.overlay.hide()
            self.root.update()
            time.sleep(0.15)              # даём экрану перерисоваться без нашего окна
            img = ImageGrab.grab()        # весь основной экран
            if use_area:
                box = select_area(self.root, img)
                if box is None:           # отмена или слишком маленькая рамка
                    print("capture: выбор области отменён")
                    if was_visible:       # прежний ответ или ошибка не должны пропадать из-за отмены
                        self.overlay.show()
                    return
                img = img.crop(box)
            self.start_request(img)
        except Exception:
            traceback.print_exc()
        finally:
            self.gate.mark_done(time.monotonic())

    def start_request(self, img) -> None:
        """Показываем «Думаю…» и запускаем запрос к модели в отдельном потоке."""
        rid, action = self.view.new_request()
        print(f"request #{rid}: отправляю в {self.s.model}")  # для отладки в консоли
        self.apply(action)
        threading.Thread(
            target=ask_model,
            args=(img, rid, self.q, lambda: self.view.is_current(rid), self.s),
            daemon=True,
        ).start()

    def apply(self, action: Optional[Action]) -> None:
        """Выполняет действие над окном ответа, выбранное AnswerView."""
        if action is None:
            return
        kind, text = action
        if kind == "set":
            self.overlay.set_text(text)
        elif kind == "append":
            self.overlay.append(text)

    def poll(self) -> None:
        """Раз в 50 мс разбираем очередь событий (выполняется в главном потоке tkinter)."""
        try:
            while True:
                rid, kind, val = self.q.get_nowait()
                if kind == "capture":
                    use_area, t = val
                    if not self.gate.is_stale(t):  # дубли, пришедшие во время выбора, пропускаем
                        self.capture(use_area)
                    else:
                        print("capture: нажатие пропущено (пришло во время предыдущего снимка)")
                elif kind == "toggle":
                    self.overlay.toggle()
                elif kind == "quit":
                    self.root.destroy()
                    return
                else:
                    if kind in ("done", "error") and self.view.is_current(rid):
                        print(f"request #{rid}: {'готово' if kind == 'done' else 'ошибка'}")
                    self.apply(self.view.handle(rid, kind, val))
        except queue.Empty:
            pass
        self.root.after(50, self.poll)

    def run(self) -> None:
        s = self.s
        print(f"Готово. {s.hotkey_full} - экран, {s.hotkey_area} - область, "
              f"{s.hotkey_hide} - скрыть, {s.hotkey_quit} - выход")
        print(f"Модель: {s.model} | think: {s.think} | num_ctx: {s.num_ctx}")
        try:
            self.root.mainloop()
        finally:
            keyboard.unhook_all()         # снимаем глобальные перехватчики клавиш