"""
ui/splash.py — Splash screen para GeoAnnotate3D

Usa la imagen oficial como fondo y agrega:
  - Barra de progreso animada (estilo dorado/naranja)
  - Texto de estado en la esquina inferior
  - Versión
  - Overlay oscuro sutil para legibilidad del texto
"""
from __future__ import annotations
import os
from PyQt5.QtWidgets import QSplashScreen, QApplication
from PyQt5.QtGui import (QPainter, QColor, QFont, QLinearGradient,
                          QPen, QPixmap, QBrush, QFontMetrics)
from PyQt5.QtCore import Qt, QRect, QTimer

VERSION = "1.0.0"

# ── Dimensiones de display (escalar la imagen para que no sea tan grande) ─────
SPLASH_W = 900
SPLASH_H = int(SPLASH_W * 668 / 1404)   # mantener aspect ratio ≈ 429


def _load_bg() -> QPixmap:
    """Carga la imagen de fondo desde la misma carpeta del módulo."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "splash_bg.png")
    if os.path.exists(path):
        px = QPixmap(path)
        if not px.isNull():
            return px.scaled(SPLASH_W, SPLASH_H,
                             Qt.KeepAspectRatio,
                             Qt.SmoothTransformation)
    # Fallback: fondo oscuro si no se encuentra la imagen
    px = QPixmap(SPLASH_W, SPLASH_H)
    px.fill(QColor(18, 22, 32))
    return px


def _build_pixmap(progress: float = 0.0, msg: str = "Iniciando…") -> QPixmap:
    """
    Genera el frame de splash con progreso y mensaje.
    progress: 0.0 → 1.0
    """
    bg = _load_bg()
    w, h = bg.width(), bg.height()

    px = QPixmap(w, h)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)

    # ── Imagen de fondo ───────────────────────────────────────────────────────
    p.drawPixmap(0, 0, bg)

    # ── Overlay oscuro inferior (zona de texto, sin tapar la imagen) ──────────
    bar_h = 54  # altura de la zona de estado
    overlay_grad = QLinearGradient(0, h - bar_h - 20, 0, h)
    overlay_grad.setColorAt(0.0, QColor(0, 0, 0, 0))
    overlay_grad.setColorAt(0.4, QColor(0, 0, 0, 140))
    overlay_grad.setColorAt(1.0, QColor(0, 0, 0, 210))
    p.fillRect(0, h - bar_h - 20, w, bar_h + 20, QBrush(overlay_grad))

    # ── Barra de progreso ─────────────────────────────────────────────────────
    bar_y   = h - 14
    bar_x   = 30
    bar_w   = w - 60
    bar_th  = 4   # grosor total del track
    fill_w  = int(bar_w * min(1.0, max(0.0, progress)))

    # Track (fondo de la barra)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(40, 30, 20, 160))
    p.drawRoundedRect(bar_x, bar_y, bar_w, bar_th, 2, 2)

    # Relleno de progreso — gradiente gris-cian, mismo ACCENT que el resto
    # de la app (rediseño 2026-09-05 v2). El overlay de fondo sigue oscuro
    # aquí a propósito (igual que el viewport 3D — ver NOTES_CLAUDE.md):
    # es la única forma de que el texto/barra sean legibles sobre la
    # imagen de marca, así que estos tonos son más claros/saturados que
    # el ACCENT plano para que resalten sobre negro, no una excepción al
    # tema — son la versión "sobre fondo oscuro" del mismo acento.
    if fill_w > 0:
        prog_grad = QLinearGradient(bar_x, 0, bar_x + fill_w, 0)
        prog_grad.setColorAt(0.0, QColor(10, 95, 103))
        prog_grad.setColorAt(0.4, QColor(14, 124, 134))
        prog_grad.setColorAt(0.7, QColor(46, 163, 173))
        prog_grad.setColorAt(1.0, QColor(110, 208, 214))
        p.setBrush(QBrush(prog_grad))
        p.drawRoundedRect(bar_x, bar_y, fill_w, bar_th, 2, 2)

        # Brillo al frente de la barra
        if fill_w > 10:
            glow_x = bar_x + fill_w - 12
            glow_grad = QLinearGradient(glow_x, 0, glow_x + 12, 0)
            glow_grad.setColorAt(0.0, QColor(140, 215, 220, 0))
            glow_grad.setColorAt(1.0, QColor(160, 225, 228, 180))
            p.setBrush(QBrush(glow_grad))
            p.drawRoundedRect(glow_x, bar_y - 1, 12, bar_th + 2, 2, 2)

    # ── Mensaje de estado ─────────────────────────────────────────────────────
    font_msg = QFont("Segoe UI", 9)
    p.setFont(font_msg)
    p.setPen(QColor(140, 215, 220))   # cian claro, legible sobre el overlay oscuro
    p.drawText(QRect(bar_x, h - 34, bar_w - 100, 20),
               Qt.AlignLeft | Qt.AlignVCenter, msg)

    # ── Versión ───────────────────────────────────────────────────────────────
    p.setPen(QColor(100, 160, 165, 180))
    font_ver = QFont("Segoe UI", 8)
    p.setFont(font_ver)
    p.drawText(QRect(bar_x, h - 34, bar_w, 20),
               Qt.AlignRight | Qt.AlignVCenter, f"v{VERSION}")

    p.end()
    return px


class SplashScreen(QSplashScreen):
    """
    Splash screen con la imagen oficial de GeoAnnotate3D.
    Barra de progreso dorada animada + mensajes de estado.
    """

    def __init__(self):
        px = _build_pixmap(0.0, "Iniciando…")
        super().__init__(px, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self._progress = 0.0
        self._msg = "Iniciando…"
        self._target = 0.0
        # Timer para animación suave de la barra
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(16)   # ~60fps
        self._anim_timer.timeout.connect(self._animate)

    def show_message(self, msg: str, progress: float = None):
        """
        Actualiza el mensaje y la barra de progreso.
        progress: 0.0–1.0 (si None, avanza automáticamente)
        """
        self._msg = msg
        if progress is not None:
            self._target = progress
        self._refresh()
        QApplication.processEvents()

    def set_progress(self, p: float):
        self._target = max(0.0, min(1.0, p))
        if not self._anim_timer.isActive():
            self._anim_timer.start()

    def _animate(self):
        """Suaviza el movimiento de la barra hacia _target."""
        diff = self._target - self._progress
        if abs(diff) < 0.002:
            self._progress = self._target
            self._anim_timer.stop()
        else:
            self._progress += diff * 0.12  # ease-out
        self._refresh()

    def _refresh(self):
        px = _build_pixmap(self._progress, self._msg)
        self.setPixmap(px)
        self.repaint()

    # Suppress default message drawing
    def drawContents(self, painter):
        pass
