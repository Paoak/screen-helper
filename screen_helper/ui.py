"""Окна на tkinter: оверлей с ответом и полноэкранный выбор области. Без бизнес-логики."""
from __future__ import annotations

import ctypes
import tkinter as tk
from typing import Optional, Tuple

from PIL import Image, ImageTk

from .core import selection_to_bbox


def enable_dpi_awareness() -> None:
    """DPI-aware: координаты tkinter совпадают с реальными пикселями экрана (только Windows)."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass  # не Windows или вызов недоступен - не критично


class Overlay:
    """Небольшое окно без рамки, всегда поверх всех и без кражи фокуса у браузера."""

    def __init__(self, root: tk.Tk):
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)          # без рамки и заголовка
        self.win.attributes("-topmost", True)    # поверх всех окон
        self.win.attributes("-alpha", 0.92)      # лёгкая прозрачность

        w, h = 460, 260
        x = self.win.winfo_screenwidth() - w - 20  # правый верхний угол
        self.win.geometry(f"{w}x{h}+{x}+20")

        self.text = tk.Text(
            self.win, wrap="word", bg="#1e1e1e", fg="#e6e6e6",
            font=("Segoe UI", 11), bd=0, padx=10, pady=8, state="disabled",
        )
        self.text.pack(fill="both", expand=True)

        self.win.update_idletasks()
        self._make_no_activate()
        self.win.withdraw()

    def _make_no_activate(self) -> None:
        """Флаг WS_EX_NOACTIVATE: клик по окну не забирает фокус у браузера."""
        try:
            GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW = -20, 0x08000000, 0x00000080
            wid = self.win.winfo_id()
            hwnd = ctypes.windll.user32.GetParent(wid) or wid
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
            )
        except Exception:
            pass  # не Windows или не вышло - окно всё равно будет работать

    def set_text(self, s: str) -> None:
        """Заменить весь текст и показать окно."""
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", s)
        self.text.config(state="disabled")
        self.win.deiconify()

    def append(self, s: str) -> None:
        """Дописать кусок ответа."""
        self.text.config(state="normal")
        self.text.insert("end", s)
        self.text.see("end")
        self.text.config(state="disabled")

    def hide(self) -> None:
        self.win.withdraw()

    def toggle(self) -> None:
        if self.win.state() == "withdrawn":
            self.win.deiconify()
        else:
            self.win.withdraw()


def select_area(root: tk.Tk, img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Показывает уже сделанный снимок (слегка затемнённый) на весь экран
    и даёт выделить область мышкой.
    Возвращает рамку в пикселях снимка или None (Esc / ПКМ / слишком маленькая область).
    """
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    fx, fy = img.width / sw, img.height / sh  # масштаб экран -> снимок (на случай DPI)

    # Затемнённая копия снимка только для показа; обрезаем потом оригинал
    dim = Image.blend(img.convert("RGB"), Image.new("RGB", img.size, "black"), 0.35)
    dim = dim.resize((sw, sh))

    sel = tk.Toplevel(root)
    sel.overrideredirect(True)
    sel.geometry(f"{sw}x{sh}+0+0")
    sel.attributes("-topmost", True)

    canvas = tk.Canvas(sel, width=sw, height=sh, highlightthickness=0, cursor="cross")
    canvas.pack()
    photo = ImageTk.PhotoImage(dim, master=sel)
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.image = photo  # держим ссылку, иначе картинку удалит сборщик мусора

    st = {"box": None, "rect": None, "x0": 0, "y0": 0}

    def on_press(e):
        st["x0"], st["y0"] = e.x, e.y
        st["rect"] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#ff3b3b", width=2)

    def on_drag(e):
        if st["rect"] is not None:
            canvas.coords(st["rect"], st["x0"], st["y0"], e.x, e.y)

    def on_release(e):
        # Перевод координат и отсев случайных кликов - в core.selection_to_bbox (тестируется)
        st["box"] = selection_to_bbox(st["x0"], st["y0"], e.x, e.y, fx, fy, img.size)
        sel.destroy()

    def cancel(_e=None):
        sel.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    canvas.bind("<Button-3>", cancel)  # ПКМ - отмена
    sel.bind("<Escape>", cancel)
    canvas.bind("<Escape>", cancel)
    sel.focus_force()
    canvas.focus_set()
    root.wait_window(sel)  # ждём, пока слой закроется
    return st["box"]
