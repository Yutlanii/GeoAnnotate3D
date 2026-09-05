"""
ui/status_bar.py — Barra de estado inferior de GeoAnnotate3D.

REDISEÑO 2026-09-05: colores movidos a ui/theme.py. Bug de contraste
real corregido: el conteo principal de puntos (`_pts_main`) usaba
`color:#a8acb0` (TEXT_FAINT — pensado para texto apenas visible) como
color base del dato más importante de la barra de estado; ahora usa
TEXT_DIM (el `<span>` interno en ACCENT para el número "activo" sigue
igual).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from PyQt5.QtWidgets import (
    QStatusBar, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QBrush

from core.project import Project
from ui.theme import (
    SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE,
    ACCENT, ACCENT_STRONG,
    OK, WARN,
)


class _BalanceBar(QWidget):
    """Barra apilada de colores por clase."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(13)
        self.setMinimumWidth(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._segments: List[Tuple[float, str]] = []

    def set_segments(self, segments: List[Tuple[float, str]]) -> None:
        self._segments = segments
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()

        # Fondo
        p.setBrush(QBrush(QColor(SURFACE_2)))
        p.setPen(Qt.NoPen)
        p.drawRect(0, 0, w, h)

        if not self._segments:
            p.end()
            return

        # Segmentos — dibujar directamente sin gaps primero
        x = 0
        for i, (frac, color_hex) in enumerate(self._segments):
            if i == len(self._segments) - 1:
                seg_w = w - x
            else:
                seg_w = max(2, int(w * frac))
            if seg_w <= 0:
                continue
            p.setBrush(QBrush(QColor(color_hex)))
            p.drawRect(x, 0, seg_w, h)
            x += seg_w + 1
            if x >= w:
                break

        p.end()


class AnnotationStatusBar(QStatusBar):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizeGripEnabled(False)
        self.setFixedHeight(80)   # un poco más alto para dar respiro
        self.setStyleSheet(f"""
            QStatusBar {{ background:{SURFACE}; border-top:1px solid {BORDER}; }}
            QStatusBar::item {{ border:none; }}
        """)
        self._build_ui()

    def _build_ui(self) -> None:
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(container)
        lay.setContentsMargins(16, 6, 16, 6)
        lay.setSpacing(20)

        # Conteo de puntos
        pts_w = QWidget(); pts_w.setStyleSheet("background: transparent;")
        pts_lay = QVBoxLayout(pts_w)
        pts_lay.setContentsMargins(0, 0, 0, 0)
        pts_lay.setSpacing(2)
        pts_lay.setAlignment(Qt.AlignVCenter)
        self._pts_main = QLabel("0 / 0 pts")
        self._pts_main.setStyleSheet(f"color:{TEXT_DIM}; font-size:13px;")
        self._pts_main.setTextFormat(Qt.RichText)
        self._pts_sub = QLabel("esperando nube…")
        self._pts_sub.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px;")
        pts_lay.addWidget(self._pts_main)
        pts_lay.addWidget(self._pts_sub)
        lay.addWidget(pts_w)

        # Balance de clases
        bal_w = QWidget(); bal_w.setStyleSheet("background: transparent;")
        bal_lay = QVBoxLayout(bal_w)
        bal_lay.setContentsMargins(0, 0, 0, 0)
        bal_lay.setSpacing(5)
        bal_w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        bal_header = QHBoxLayout()
        bal_header.setContentsMargins(0, 0, 0, 0)
        self._bal_lbl = QLabel("BALANCE DE CLASES")
        self._bal_lbl.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px; letter-spacing:1px;")
        self._bal_ratio = QLabel("")
        self._bal_ratio.setAlignment(Qt.AlignRight)
        self._bal_ratio.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px;")
        bal_header.addWidget(self._bal_lbl)
        bal_header.addWidget(self._bal_ratio)

        self._balance_bar = _BalanceBar()
        bal_lay.addLayout(bal_header)
        bal_lay.addWidget(self._balance_bar)
        lay.addWidget(bal_w, 1)

        # Advertencias
        warn_w = QWidget(); warn_w.setStyleSheet("background: transparent;")
        warn_lay = QVBoxLayout(warn_w)
        warn_lay.setContentsMargins(0, 0, 0, 0)
        warn_lay.setSpacing(2)
        warn_lay.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self._warn_count = QLabel("")
        self._warn_count.setAlignment(Qt.AlignRight)
        self._warn_count.setStyleSheet(f"color:{ACCENT_STRONG}; font-size:12px; font-weight:600;")
        self._warn_detail = QLabel("")
        self._warn_detail.setAlignment(Qt.AlignRight)
        self._warn_detail.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px;")
        warn_lay.addWidget(self._warn_count)
        warn_lay.addWidget(self._warn_detail)
        lay.addWidget(warn_w)

        lay.addWidget(self._vsep())

        # FPS + Octree
        tech_w = QWidget(); tech_w.setStyleSheet("background: transparent;")
        tech_lay = QVBoxLayout(tech_w)
        tech_lay.setContentsMargins(0, 0, 0, 0)
        tech_lay.setSpacing(2)
        tech_lay.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self._fps_lbl = QLabel("idle")
        self._fps_lbl.setAlignment(Qt.AlignRight)
        self._fps_lbl.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px;")
        self._oct_lbl = QLabel("OCT: —")
        self._oct_lbl.setAlignment(Qt.AlignRight)
        self._oct_lbl.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px;")
        tech_lay.addWidget(self._fps_lbl)
        tech_lay.addWidget(self._oct_lbl)
        lay.addWidget(tech_w)

        lay.addWidget(self._vsep())

        # Timer
        self._timer_lbl = QLabel("00:00:00")
        self._timer_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px; min-width:58px; font-family:'Consolas';")
        self._timer_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self._timer_lbl)

        self.addPermanentWidget(container, 1)

    def _vsep(self) -> QWidget:
        sep = QWidget()
        sep.setFixedSize(1, 36)
        sep.setStyleSheet(f"background: {BORDER_SOFT};")
        return sep

    # ── API ───────────────────────────────────────────────────────────

    def show_loading(self, msg) -> None:
        if msg is None:
            self._pts_sub.setText("")
            return
        self._pts_sub.setText(str(msg))

    def hide_loading(self) -> None:
        self._pts_sub.setText("listo")

    def update_loading_progress(self, pct: int, msg: str) -> None:
        self._pts_sub.setText(f"{pct}%  {msg}")

    def set_cloud(self, pc, project: Project) -> None:
        mmap_badge = f" <span style='color:{ACCENT};font-size:10.5px;'>MMAP</span>" if getattr(pc,'is_mmap',False) else ""
        self._pts_main.setText(
            f"<span style='color:{ACCENT}'>0</span> / {pc.n_points:,} pts{mmap_badge}")
        self._pts_sub.setText("etiquetados · sesión activa")

    def set_octree_ready(self, ready: bool) -> None:
        if ready:
            self._oct_lbl.setText("OCT ✓")
            self._oct_lbl.setStyleSheet(f"color:{OK}; font-size:10.5px;")
        else:
            self._oct_lbl.setText("OCT: error")
            self._oct_lbl.setStyleSheet(f"color:{WARN}; font-size:10.5px;")

    def update_fps(self, fps: float) -> None:
        if fps == 0:
            self._fps_lbl.setText("idle")
            self._fps_lbl.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px;")
        elif fps >= 40:
            self._fps_lbl.setText(f"{fps:.0f} fps")
            self._fps_lbl.setStyleSheet(f"color:{OK}; font-size:10.5px;")
        elif fps >= 20:
            self._fps_lbl.setText(f"{fps:.0f} fps")
            self._fps_lbl.setStyleSheet(f"color:{ACCENT}; font-size:10.5px;")
        else:
            self._fps_lbl.setText(f"{fps:.0f} fps")
            self._fps_lbl.setStyleSheet(f"color:{WARN}; font-size:10.5px;")

    def update_render_info(self, rendered: int, total: int) -> None:
        if total > 0:
            self._pts_main.setText(
                f"<span style='color:{ACCENT}'>{rendered:,}</span> / {total:,} pts")

    def update_stats_with_schema(self, label_store, schema: list) -> None:
        """Actualiza conteo y barra de balance con colores del schema."""
        n_labeled = label_store.n_labeled
        n_total   = label_store.n_total
        counts    = label_store.per_class_counts()

        # Actualizar conteo
        self._pts_main.setText(
            f"<span style='color:{ACCENT}'>{n_labeled:,}</span> / {n_total:,} pts")
        self._pts_sub.setText("etiquetados · sesión activa")

        if not counts:
            self._bal_ratio.setText("")
            self._balance_bar.set_segments([])
            self._warn_count.setText("")
            self._warn_detail.setText("")
            return

        color_map = {sc.id: sc.color for sc in schema}
        name_map  = {sc.id: sc.name  for sc in schema}
        total_labeled = sum(counts.values())

        mx = max(counts.values())
        mn = min(counts.values())
        self._bal_ratio.setText(f"ratio {mx // max(mn, 1)}:1")

        segs: List[Tuple[float, str]] = []
        warnings: List[str] = []
        for cid, cnt in sorted(counts.items()):
            frac  = cnt / total_labeled
            color = color_map.get(cid, ACCENT)
            segs.append((frac, color))
            if frac < 0.01:
                warnings.append(name_map.get(cid, str(cid)))

        self._balance_bar.set_segments(segs)

        if warnings:
            self._warn_count.setText(f"{len(warnings)} advertencias")
            self._warn_detail.setText(", ".join(warnings[:3]) + " <1%")
        else:
            self._warn_count.setText("")
            self._warn_detail.setText("")

    # Alias para compatibilidad
    def update_stats(self, label_store) -> None:
        self.update_stats_with_schema(label_store, [])

    def update_session_time(self, h: int, m: int, s: int) -> None:
        self._timer_lbl.setText(f"{h:02d}:{m:02d}:{s:02d}")
