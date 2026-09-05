"""
annotation/label_store.py
=========================
Almacén de etiquetas con soporte de undo/redo.

Responsabilidades:
- Mantener el array labels uint8 sincronizado con project.labels
- Registrar operaciones de anotación en el stack de undo
- Exponer annotate(indices, class_id) como única forma de escribir etiquetas
- Auto-save periódico a labels.npy

La regla de oro: NADIE escribe en project.labels directamente.
Todo pasa por LabelStore.annotate().
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal


# Máximo de operaciones en el stack de undo
MAX_UNDO = 50


# ---------------------------------------------------------------------------
# Operación de anotación (unidad de undo)
# ---------------------------------------------------------------------------

@dataclass
class AnnotationOp:
    """
    Una operación atómica de anotación.

    Guardamos los índices afectados y las etiquetas ANTERIORES (para undo).
    Las etiquetas NUEVAS no necesitan guardarse — se reaplican con class_id.
    """
    indices:       np.ndarray   # (K,) int32 — índices afectados
    prev_labels:   np.ndarray   # (K,) uint8 — etiquetas antes de la op
    new_class_id:  int          # clase que se aplicó


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class LabelStore(QObject):
    """
    Fuente de verdad de las etiquetas de anotación.

    Señales:
        labels_changed(np.ndarray, np.ndarray)  — (indices, nuevo_class_id_array)
            Emitida después de cada annotate() o undo()/redo()
            Permite al canvas refrescar solo los puntos afectados.
        stats_changed()
            Emitida cuando cambian los conteos por clase.
    """

    labels_changed = pyqtSignal(object, object)   # indices, class_ids
    stats_changed  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels: Optional[np.ndarray]   = None   # (N,) uint8
        self._undo_stack: Deque[AnnotationOp] = deque(maxlen=MAX_UNDO)
        self._redo_stack: Deque[AnnotationOp] = deque(maxlen=MAX_UNDO)

        # Auto-save cada 60 segundos
        self._autosave_t = QTimer(self)
        self._autosave_t.setInterval(60_000)
        self._autosave_t.timeout.connect(self._autosave)
        self._autosave_path: Optional[str] = None

        # Debounce de stats_changed: per_class_counts() y n_labeled recorren
        # TODO el array de labels (cientos de ms en nubes de 80-100M pts).
        # Sin esto, cada trazo del pincel disparaba 2 escaneos completos
        # de forma síncrona — la causa principal de la lentitud restante
        # al pintar en nubes densas. Con debounce, mientras se pinta
        # rápido/continuo se coalescen en como mucho ~6-7 actualizaciones
        # por segundo (imperceptible para un contador de texto).
        self._stats_pending = False
        self._stats_timer = QTimer(self)
        self._stats_timer.setSingleShot(True)
        self._stats_timer.setInterval(150)
        self._stats_timer.timeout.connect(self._emit_stats_changed)

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    def attach(self, labels: np.ndarray, autosave_path: Optional[str] = None) -> None:
        """
        Adjunta el array de etiquetas del proyecto.
        El LabelStore trabaja directamente sobre este array (sin copia).
        """
        assert labels.dtype == np.uint8, "labels debe ser uint8"
        self._labels = labels
        self._autosave_path = autosave_path
        self._undo_stack.clear()
        self._redo_stack.clear()
        if autosave_path:
            self._autosave_t.start()

    # ------------------------------------------------------------------
    # Escritura de etiquetas
    # ------------------------------------------------------------------

    def annotate(self, indices: np.ndarray, class_id: int) -> None:
        """
        Aplica class_id a los puntos en indices.
        Usa FC.write_labels (C, GIL liberado).
        """
        if self._labels is None or len(indices) == 0:
            return

        idx = np.ascontiguousarray(indices, dtype=np.int32)
        prev = self._labels[idx].copy()

        op = AnnotationOp(indices=idx, prev_labels=prev, new_class_id=class_id)
        self._undo_stack.append(op)
        self._redo_stack.clear()

        # Escritura in-place via C (sin GIL)
        try:
            from core._fast import FC
            FC.write_labels(self._labels, idx, int(class_id))
        except Exception:
            self._labels[idx] = np.uint8(class_id)

        self.labels_changed.emit(idx, np.full(len(idx), class_id, np.uint8))
        self._request_stats_update()

    def _request_stats_update(self) -> None:
        """Coalesce múltiples annotate() seguidos en una sola stats_changed."""
        if not self._stats_pending:
            self._stats_pending = True
            self._stats_timer.start()

    def _emit_stats_changed(self) -> None:
        self._stats_pending = False
        self.stats_changed.emit()

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def undo(self) -> bool:
        """
        Deshace la última operación.
        Retorna True si se pudo deshacer.
        """
        if not self._undo_stack or self._labels is None:
            return False
        op = self._undo_stack.pop()
        self._redo_stack.append(op)
        self._labels[op.indices] = op.prev_labels
        self.labels_changed.emit(op.indices, op.prev_labels)
        self.stats_changed.emit()
        return True

    def redo(self) -> bool:
        """
        Rehace la última operación deshecha.
        Retorna True si se pudo rehacer.
        """
        if not self._redo_stack or self._labels is None:
            return False
        op = self._redo_stack.pop()
        self._undo_stack.append(op)
        self._labels[op.indices] = np.uint8(op.new_class_id)
        self.labels_changed.emit(op.indices,
                                 np.full(len(op.indices), op.new_class_id, np.uint8))
        self.stats_changed.emit()
        return True

    # ------------------------------------------------------------------
    # Estadísticas
    # ------------------------------------------------------------------

    @property
    def n_labeled(self) -> int:
        if self._labels is None:
            return 0
        return int(np.count_nonzero(self._labels))

    @property
    def n_total(self) -> int:
        if self._labels is None:
            return 0
        return len(self._labels)

    def per_class_counts(self) -> dict:
        if self._labels is None:
            return {}
        try:
            from core._fast import FC
            counts = FC.per_class_counts(self._labels)
            # Remove class 0 (unlabeled)
            counts.pop(0, None)
            return counts
        except Exception:
            lbl = self._labels[self._labels > 0]
            if len(lbl) == 0:
                return {}
            unique, counts = np.unique(lbl, return_counts=True)
            return {int(u): int(c) for u, c in zip(unique, counts)}

    # ------------------------------------------------------------------
    # Estado de undo/redo
    # ------------------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    # ------------------------------------------------------------------
    # Auto-save
    # ------------------------------------------------------------------

    def _autosave(self) -> None:
        if self._labels is None or self._autosave_path is None:
            return
        try:
            np.save(self._autosave_path, self._labels)
        except Exception as exc:
            print(f"[LabelStore] auto-save falló: {exc}")
