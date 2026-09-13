"""
ui/profile_view.py — Ventana de vista de perfil / corte vertical.

Muestra la franja extraída por ProfileTool (ver annotation/profile.py y
annotation/tools.py::ProfileTool) como un scatter 2D: eje X = distancia
a lo largo de la línea trazada, eje Y = altura (Z), con exageración
vertical ajustable (como en cualquier visor de perfiles topográficos,
donde la escala horizontal y vertical casi nunca conviene que sean 1:1)
y pan/zoom con mouse.

Sin dependencia de matplotlib — un QWidget con QPainter es más liviano
y evita agregar una dependencia nueva solo para un scatter simple.

La transformación datos→pantalla está separada en `fit_transform()`
(función pura, sin Qt) para poder probarla headless sin necesidad de
disparar un paintEvent real.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QWidget, QPushButton, QDoubleSpinBox, QFileDialog,
                             QMessageBox)

# Nunca rasterizar más de esto de una — un tope de seguridad, no el
# límite real de rendimiento (ver nota en _render_arrays): el cuello de
# botella real de la versión anterior no era la CANTIDAD de puntos sino
# dibujar cada uno con drawEllipse() individual (una llamada QPainter por
# punto, con antialiasing) — con el buffer raster vectorizado de abajo,
# varios cientos de miles de puntos cuestan lo mismo que unos pocos miles.
MAX_RENDER_POINTS = 2_000_000


def fit_transform(t_min: float, t_max: float, z_min: float, z_max: float,
                  w: int, h: int, margin: float, exaggeration: float,
                  pan: Tuple[float, float] = (0.0, 0.0), zoom: float = 1.0
                  ) -> Tuple[float, float, float, float]:
    """
    Calcula (scale_x, scale_y, origin_x, origin_y) tales que:
        screen_x = origin_x + (t - t_min) * scale_x
        screen_y = origin_y - (z - z_min) * scale_y   (Y de pantalla crece hacia abajo)

    `exaggeration` multiplica SOLO scale_y (relativo a scale_x) — así el
    perfil no queda aplastado cuando el terreno varía poco en altura
    frente a una distancia horizontal grande (el caso normal en LiDAR
    aéreo). `zoom`/`pan` son estado de vista (rueda del mouse / arrastre),
    aplicados sobre la escala "ajustar todo" de base.
    """
    t_span = max(t_max - t_min, 1e-6)
    z_span = max(z_max - z_min, 1e-6)
    avail_w = max(1.0, w - 2 * margin)
    avail_h = max(1.0, h - 2 * margin)
    base_scale_x = avail_w / t_span
    base_scale_y = avail_h / z_span
    scale_x = base_scale_x * zoom
    scale_y = base_scale_y * zoom * exaggeration
    origin_x = margin + pan[0]
    origin_y = h - margin + pan[1]
    return scale_x, scale_y, origin_x, origin_y


class ProfileCanvas(QWidget):
    """Widget que pinta el scatter 2D del perfil, con pan (arrastre) y
    zoom (rueda del mouse)."""

    MARGIN = 44.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 280)
        self.setMouseTracking(True)
        self._t = np.zeros(0, np.float32)
        self._z = np.zeros(0, np.float32)
        self._colors: Optional[np.ndarray] = None   # (N,3) uint8 o None
        self._legend: list = []   # [(nombre_clase, (r,g,b)), ...]
        self._length_m = 0.0
        self._exaggeration = 2.0
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self._dragging = False
        self._drag_last = None
        self._hover_screen: Optional[QPointF] = None
        # Índices a rasterizar/usar para hover — con el tope de seguridad
        # de MAX_RENDER_POINTS (ver comentario junto a esa constante); a
        # diferencia de la versión anterior, esto YA NO limita a ~60k por
        # costo de dibujado (el buffer raster no lo necesita), solo evita
        # un caso patológico de memoria/tiempo con franjas absurdamente
        # anchas.
        self._render_idx: Optional[np.ndarray] = None

    def set_data(self, t: np.ndarray, z: np.ndarray,
                colors: Optional[np.ndarray], length_m: float,
                legend: Optional[list] = None) -> None:
        self._t = np.asarray(t, np.float32)
        self._z = np.asarray(z, np.float32)
        self._colors = colors
        self._legend = legend or []
        self._length_m = float(length_m)
        n = len(self._t)
        step = max(1, n // MAX_RENDER_POINTS)
        self._render_idx = np.arange(0, n, step)
        self._hover_screen = None
        self.reset_view()

    def _screen_xy(self, idx: np.ndarray):
        """
        Coordenadas de pantalla (arrays numpy, vectorizado) de los puntos
        en `idx` — reemplaza el `to_screen()` punto-por-punto de antes,
        que era la razón real de la lentitud al hacer zoom/pan (una
        llamada Python + QPointF por punto, en cada repaint).
        """
        t_min, t_max, z_min, z_max = self._bounds()
        sx, sy, ox, oy = self._transform()
        xs = ox + (self._t[idx] - t_min) * sx
        ys = oy - (self._z[idx] - z_min) * sy
        return xs, ys

    def _render_arrays(self, w: int, h: int) -> Optional[QImage]:
        """
        Rasteriza los puntos del perfil a un buffer QImage escribiendo
        directamente sobre su memoria como un array numpy (fancy
        indexing, vectorizado) — nada de Python por punto. Esto es lo
        que hace que hacer zoom/pan ya no se sienta lento: antes cada
        repaint llamaba drawEllipse() una vez por punto (hasta 60k
        veces), cada una con su propio costo de antialiasing; ahora todo
        el scatter es un solo `drawImage()`.
        """
        xs, ys = self._screen_xy(self._render_idx)
        xi = xs.astype(np.int32)
        yi = ys.astype(np.int32)
        mask = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if not np.any(mask):
            return None
        xi, yi = xi[mask], yi[mask]

        if self._colors is not None:
            cols = np.asarray(self._colors, np.uint8)[self._render_idx][mask]
        else:
            cols = np.empty((len(xi), 3), np.uint8)
            cols[:, 0] = 90; cols[:, 1] = 200; cols[:, 2] = 230

        img = QImage(w, h, QImage.Format_RGB888)
        img.fill(QColor("#1c1e1f"))
        bpl = img.bytesPerLine()
        ptr = img.bits()
        ptr.setsize(h * bpl)
        # OJO: `buf` es una VISTA real sobre la memoria de `img` (no una
        # copia) — por eso escribir en `buf` con fancy indexing cambia el
        # QImage directamente. Se indexa como (h, bpl) en vez de
        # reshape(h, w, 3): bpl casi siempre trae padding de alineación
        # (>= w*3), y un reshape sobre un slice no-contiguo silenciosamente
        # devolvería una COPIA (rompiendo el punto de todo esto).
        buf = np.frombuffer(ptr, np.uint8, count=h * bpl).reshape(h, bpl)

        # Cada punto se pinta como un bloque de 2x2 px (no 1x1) para que
        # siga siendo visible sin antialiasing — desplazamientos fijos,
        # todavía vectorizado (4 escrituras de array en vez de 4*N).
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            xj = np.clip(xi + dx, 0, w - 1)
            yj = np.clip(yi + dy, 0, h - 1)
            col0 = xj * 3
            buf[yj, col0]     = cols[:, 0]
            buf[yj, col0 + 1] = cols[:, 1]
            buf[yj, col0 + 2] = cols[:, 2]
        return img

    def set_exaggeration(self, v: float) -> None:
        self._exaggeration = max(0.1, float(v))
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self.update()

    def _bounds(self):
        if len(self._t) == 0:
            return 0.0, 1.0, 0.0, 1.0
        return (float(self._t.min()), float(self._t.max()),
                float(self._z.min()), float(self._z.max()))

    def _transform(self):
        t_min, t_max, z_min, z_max = self._bounds()
        return fit_transform(t_min, t_max, z_min, z_max,
                             self.width(), self.height(), self.MARGIN,
                             self._exaggeration, tuple(self._pan), self._zoom)

    # ── Pintado ──────────────────────────────────────────────────────────────

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#1c1e1f"))

        if len(self._t) == 0:
            p.setPen(QColor("#84888c"))
            p.drawText(self.rect(), Qt.AlignCenter, "Sin puntos en la franja")
            p.end()
            return

        t_min, t_max, z_min, z_max = self._bounds()
        sx, sy, ox, oy = self._transform()

        def to_screen(t, z):
            return QPointF(ox + (t - t_min) * sx, oy - (z - z_min) * sy)

        # Grilla + ejes
        p.setPen(QPen(QColor("#3a3d3f"), 1))
        n_ticks = 6
        for i in range(n_ticks + 1):
            t_val = t_min + (t_max - t_min) * i / n_ticks
            pt = to_screen(t_val, z_min)
            p.drawLine(QPointF(pt.x(), self.MARGIN), QPointF(pt.x(), self.height() - self.MARGIN))
            p.setPen(QColor("#9aa0a6"))
            p.drawText(QPointF(pt.x() - 14, self.height() - self.MARGIN + 16),
                      f"{t_val:.1f}")
            p.setPen(QPen(QColor("#3a3d3f"), 1))
        for i in range(n_ticks + 1):
            z_val = z_min + (z_max - z_min) * i / n_ticks
            pt = to_screen(t_min, z_val)
            p.drawLine(QPointF(self.MARGIN, pt.y()), QPointF(self.width() - self.MARGIN, pt.y()))
            p.setPen(QColor("#9aa0a6"))
            p.drawText(QPointF(6, pt.y() + 4), f"{z_val:.1f}")
            p.setPen(QPen(QColor("#3a3d3f"), 1))

        # Puntos — rasterizados en un buffer QImage vectorizado con numpy
        # (una sola operación por canal en vez de una llamada drawEllipse
        # por punto) y volcados con un único drawImage(); ver
        # _render_arrays() para el porqué de este cambio (antes esto era
        # lo que hacía lento el zoom/pan con franjas de más de unos
        # pocos miles de puntos).
        w, h = self.width(), self.height()
        if w > 0 and h > 0:
            img = self._render_arrays(w, h)
            if img is not None:
                p.drawImage(0, 0, img)

        # Etiqueta de ejes
        p.setPen(QColor("#c7cacd"))
        f = QFont(); f.setPixelSize(11); p.setFont(f)
        p.drawText(QPointF(self.width() / 2 - 60, self.height() - 8),
                  "Distancia a lo largo de la línea (m)")

        # Leyenda de clases (si se coloreó por clasificación)
        if self._legend:
            lx, ly = self.width() - 150, self.MARGIN + 6
            p.setPen(Qt.NoPen); p.setBrush(QColor(0, 0, 0, 140))
            p.drawRect(int(lx) - 6, int(ly) - 14,
                      150, 16 * len(self._legend) + 6)
            fsmall = QFont(); fsmall.setPixelSize(10); p.setFont(fsmall)
            for i, (cname, (r, g, b)) in enumerate(self._legend):
                yy = ly + i * 16
                p.setPen(Qt.NoPen); p.setBrush(QColor(r, g, b))
                p.drawEllipse(QPointF(lx, yy), 4, 4)
                p.setPen(QColor("#e8e9eb"))
                p.drawText(QPointF(lx + 10, yy + 4), cname[:18])

        # Lectura al pasar el mouse: punto más cercano y su (distancia,
        # altura) — no cambia nada, solo informa. Búsqueda VECTORIZADA
        # (numpy sobre todo `_render_idx` de una vez) en vez del bucle
        # Python de antes, que se repetía en cada movimiento de mouse —
        # con franjas grandes eso se sentía tan lento como el propio
        # dibujado de puntos.
        if self._hover_screen is not None and self._render_idx is not None and len(self._render_idx) > 0:
            hs = self._hover_screen
            xs, ys = self._screen_xy(self._render_idx)
            d2_all = (xs - hs.x()) ** 2 + (ys - hs.y()) ** 2
            j = int(np.argmin(d2_all))
            best_i = int(self._render_idx[j]) if d2_all[j] < 12.0 ** 2 else None
            if best_i is not None:
                pt = to_screen(float(self._t[best_i]), float(self._z[best_i]))
                p.setPen(QPen(QColor("#ffffff"), 1.5))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(pt, 5, 5)
                txt = f"d={float(self._t[best_i]):.2f} m  z={float(self._z[best_i]):.2f} m"
                fh = QFont(); fh.setPixelSize(11); fh.setBold(True); p.setFont(fh)
                tw = p.fontMetrics().horizontalAdvance(txt)
                tx = min(max(pt.x() + 10, self.MARGIN), self.width() - tw - 8)
                ty = max(pt.y() - 10, self.MARGIN + 12)
                p.setPen(QColor("#1c1e1f")); p.setBrush(QColor(255, 255, 255, 220))
                p.drawRect(int(tx) - 4, int(ty) - 13, tw + 8, 17)
                p.setPen(QColor("#1c1e1f"))
                p.drawText(QPointF(tx, ty), txt)
        p.end()

    # ── Interacción: pan (arrastre) y zoom (rueda) ────────────────────────────

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_last = ev.pos()

    def mouseMoveEvent(self, ev) -> None:
        if self._dragging and self._drag_last is not None:
            dx = ev.pos().x() - self._drag_last.x()
            dy = ev.pos().y() - self._drag_last.y()
            self._pan[0] += dx
            self._pan[1] += dy
            self._drag_last = ev.pos()
        # El hover se actualiza SIEMPRE (arrastrando o no) — así el punto
        # resaltado se sigue reubicando mientras haces pan, en vez de
        # quedar pegado a una posición de pantalla que ya no corresponde
        # a donde está el cursor.
        self._hover_screen = QPointF(ev.pos())
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        self._dragging = False
        self._drag_last = None

    def leaveEvent(self, ev) -> None:
        self._hover_screen = None
        self.update()

    def wheelEvent(self, ev) -> None:
        delta = ev.angleDelta().y()
        factor = 1.15 if delta > 0 else (1 / 1.15)
        self._zoom = max(0.1, min(50.0, self._zoom * factor))
        self.update()


def profile_rows_to_csv(t: np.ndarray, z: np.ndarray,
                        colors: Optional[np.ndarray] = None) -> str:
    """
    Arma el contenido CSV del perfil — separado de `_export_csv` (que
    solo se ocupa del diálogo de guardar) para poder probarlo sin Qt.
    """
    lines = ["distancia_m,altura_m,r,g,b" if colors is not None else "distancia_m,altura_m"]
    for i in range(len(t)):
        if colors is not None:
            r, g, b = (int(c) for c in colors[i])
            lines.append(f"{float(t[i]):.4f},{float(z[i]):.4f},{r},{g},{b}")
        else:
            lines.append(f"{float(t[i]):.4f},{float(z[i]):.4f}")
    return "\n".join(lines) + "\n"


class ProfileDialog(QDialog):
    """
    Ventana NO modal (se puede seguir usando el resto de la app con esta
    abierta) que muestra el perfil extraído por ProfileTool. Se reusa la
    misma instancia entre perfiles sucesivos (ProfileTool guarda la
    referencia) en vez de abrir una ventana nueva cada vez.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Perfil / corte vertical")
        self.setModal(False)
        self.resize(680, 420)

        lay = QVBoxLayout(self)
        self._info_lbl = QLabel("")
        self._info_lbl.setStyleSheet("color:#c7cacd;font-size:11px;")
        lay.addWidget(self._info_lbl)

        self._canvas = ProfileCanvas(self)
        lay.addWidget(self._canvas, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Exageración vertical:"))
        self._exag_spin = QDoubleSpinBox()
        self._exag_spin.setRange(0.1, 50.0)
        self._exag_spin.setSingleStep(0.5)
        self._exag_spin.setValue(2.0)
        self._exag_spin.valueChanged.connect(self._canvas.set_exaggeration)
        row.addWidget(self._exag_spin)
        row.addStretch()
        btn_reset = QPushButton("Restablecer vista")
        btn_reset.clicked.connect(self._canvas.reset_view)
        row.addWidget(btn_reset)
        self._btn_csv = QPushButton("Exportar CSV…")
        self._btn_csv.clicked.connect(self._export_csv)
        row.addWidget(self._btn_csv)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.close)
        row.addWidget(btn_close)
        lay.addLayout(row)

    def update_data(self, t: np.ndarray, z: np.ndarray,
                   colors: Optional[np.ndarray], length_m: float,
                   buffer_m: float, legend: Optional[list] = None) -> None:
        self._canvas.set_data(t, z, colors, length_m, legend=legend)
        self._exag_spin.blockSignals(True)
        self._exag_spin.setValue(self._canvas._exaggeration)
        self._exag_spin.blockSignals(False)
        self._info_lbl.setText(
            f"Longitud: {length_m:.2f} m · Ancho de franja: ±{buffer_m:.2f} m · "
            f"{len(t):,} puntos")
        self.show()
        self.raise_()
        self.activateWindow()

    def _export_csv(self) -> None:
        """
        Guarda los puntos del perfil actual como CSV (distancia, altura,
        y color RGB si se coloreó por clasificación) — para graficarlo o
        procesarlo fuera de la app (Excel, un script propio, etc.), en
        vez de que la vista de perfil sea la única forma de verlo.
        """
        t, z = self._canvas._t, self._canvas._z
        if len(t) == 0:
            QMessageBox.information(self, "Exportar CSV", "No hay puntos en el perfil actual.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar perfil como CSV", "perfil.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(profile_rows_to_csv(t, z, self._canvas._colors))
        except Exception as e:
            QMessageBox.critical(self, "Exportar CSV", f"No se pudo guardar el archivo:\n{e}")
            return
        QMessageBox.information(self, "Exportar CSV", f"Perfil guardado en:\n{path}")
