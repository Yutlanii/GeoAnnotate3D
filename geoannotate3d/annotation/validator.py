"""
annotation/validator.py — Validación del dataset antes de exportar

La validación responde a:
  ¿Vale la pena exportar este dataset para entrenar una red neuronal?

Checks que realiza:
  1. Cobertura total         — ¿qué % de la nube está etiquetado?
  2. Balance de clases       — ¿alguna clase tiene muy pocos puntos?
  3. Cobertura por tile      — ¿hay tiles sin ningún punto anotado?
  4. Clases sin puntos       — ¿hay clases definidas con 0 puntos?
  5. Split sostenible        — ¿hay suficientes tiles para train/val/test?
  6. Consistencia schema     — ¿todas las clases tienen nombre y color?

Cada check produce un resultado: OK / WARNING / ERROR
  OK      → todo bien
  WARNING → puede funcionar pero puede afectar la calidad del modelo
  ERROR   → el export producirá basura o fallará

Uso:
  report = validate_dataset(pc, project, label_store, tile_manager)
  report.has_errors    → True si hay algún ERROR
  report.has_warnings  → True si hay algún WARNING
  report.summary()     → texto con todos los resultados
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Resultado de un check individual
# ─────────────────────────────────────────────────────────────────────────────

OK      = "OK"
WARNING = "WARNING"
ERROR   = "ERROR"


@dataclass
class CheckResult:
    level:   str    # OK / WARNING / ERROR
    title:   str    # Nombre corto del check
    message: str    # Descripción completa
    value:   float = 0.0  # Valor numérico asociado (para la UI)

    @property
    def icon(self) -> str:
        return {"OK": "✓", "WARNING": "⚠", "ERROR": "✗"}[self.level]

    def __str__(self):
        return f"{self.icon} {self.title}: {self.message}"


# ─────────────────────────────────────────────────────────────────────────────
# Reporte completo
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    checks: List[CheckResult] = field(default_factory=list)

    def add(self, level: str, title: str, message: str, value: float = 0.0):
        self.checks.append(CheckResult(level, title, message, value))

    @property
    def has_errors(self) -> bool:
        return any(c.level == ERROR for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.level == WARNING for c in self.checks)

    @property
    def n_errors(self) -> int:
        return sum(1 for c in self.checks if c.level == ERROR)

    @property
    def n_warnings(self) -> int:
        return sum(1 for c in self.checks if c.level == WARNING)

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            lines.append(str(c))
        return "\n".join(lines)

    def errors_and_warnings(self) -> List[CheckResult]:
        return [c for c in self.checks if c.level != OK]


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(pc, project, label_store, tile_manager=None) -> ValidationReport:
    """
    Ejecuta todos los checks y devuelve un ValidationReport.

    Parámetros:
        pc           — PointCloud
        project      — Project con schema y labels
        label_store  — LabelStore con estadísticas actualizadas
        tile_manager — TileManager (opcional, para checks de tiles)
    """
    report = ValidationReport()
    labels = project.labels if project else None

    if labels is None or len(labels) == 0:
        report.add(ERROR, "Sin etiquetas",
                   "No hay ningún punto etiquetado. Anota la nube antes de exportar.")
        return report

    n_total   = len(labels)
    n_labeled = int(np.count_nonzero(labels))
    pct_total = 100.0 * n_labeled / max(n_total, 1)

    # ── Check 1: Cobertura total ──────────────────────────────────────────────
    if n_labeled == 0:
        report.add(ERROR, "Cobertura 0%",
                   "No hay ningún punto etiquetado.", value=0.0)
    elif pct_total < 5.0:
        report.add(WARNING, f"Cobertura baja ({pct_total:.1f}%)",
                   f"Solo {n_labeled/1e6:.1f}M de {n_total/1e6:.1f}M pts etiquetados. "
                   f"Las redes neuronales necesitan al menos 5-10% para generalizar bien.",
                   value=pct_total)
    elif pct_total < 20.0:
        report.add(WARNING, f"Cobertura moderada ({pct_total:.1f}%)",
                   f"{n_labeled/1e6:.1f}M pts etiquetados de {n_total/1e6:.1f}M. "
                   "Recomendado ≥20% para entrenamiento robusto.",
                   value=pct_total)
    else:
        report.add(OK, f"Cobertura ({pct_total:.1f}%)",
                   f"{n_labeled/1e6:.1f}M pts etiquetados.",
                   value=pct_total)

    # ── Check 2: Clases con 0 puntos ──────────────────────────────────────────
    if project and project.schema:
        unique_cls = set(np.unique(labels[labels > 0]).tolist())
        empty_classes = [s.name for s in project.schema if s.id not in unique_cls]
        if empty_classes:
            report.add(WARNING, "Clases sin puntos",
                       f"Estas clases no tienen ningún punto etiquetado: "
                       f"{', '.join(empty_classes)}. "
                       "El split train/val/test no las incluirá.")

        # ── Check 3: Balance de clases ────────────────────────────────────────
        if n_labeled > 0 and project.schema:
            counts = {}
            for s in project.schema:
                counts[s.id] = int(np.sum(labels == s.id))
            present = {cid: cnt for cid, cnt in counts.items() if cnt > 0}
            if len(present) >= 2:
                max_cnt = max(present.values())
                min_cnt = min(present.values())
                ratio   = max_cnt / max(min_cnt, 1)
                min_name = next(s.name for s in project.schema
                                if s.id == min(present, key=present.get))
                max_name = next(s.name for s in project.schema
                                if s.id == max(present, key=present.get))
                if ratio > 50:
                    report.add(WARNING, f"Clases muy desbalanceadas ({ratio:.0f}:1)",
                               f"'{max_name}' tiene {ratio:.0f}x más puntos que '{min_name}'. "
                               "Considera class_weights o sobremuestreo para el entrenamiento.",
                               value=ratio)
                elif ratio > 10:
                    report.add(WARNING, f"Clases desbalanceadas ({ratio:.0f}:1)",
                               f"'{max_name}' tiene {ratio:.0f}x más puntos que '{min_name}'. "
                               "Los class_weights del export ayudarán.",
                               value=ratio)
                else:
                    report.add(OK, f"Balance de clases ({ratio:.1f}:1)",
                               "Distribución aceptable entre clases.",
                               value=ratio)

        # ── Check 4: Consistencia del schema ──────────────────────────────────
        bad_schema = [s.name for s in project.schema
                      if not s.name or not s.color or s.color == "#000000"]
        if bad_schema:
            report.add(WARNING, "Clases sin nombre/color",
                       f"Clases con configuración incompleta: {', '.join(bad_schema)}")

    # ── Check 5: Tiles ────────────────────────────────────────────────────────
    if tile_manager is not None:
        tiles = tile_manager.tiles
        n_tiles = len(tiles)
        if n_tiles > 0:
            n_with_annot  = sum(1 for t in tiles if t.labeled_pct > 0)
            n_complete     = sum(1 for t in tiles if t.labeled_pct >= 95.0)
            n_partial      = n_with_annot - n_complete
            n_empty        = n_tiles - n_with_annot

            pct_annot = 100.0 * n_with_annot / n_tiles

            if n_with_annot == 0:
                report.add(ERROR, "Sin tiles anotados",
                           "Ningún tile tiene puntos etiquetados.")
            elif n_with_annot < 3:
                report.add(ERROR, "Muy pocos tiles anotados",
                           f"Solo {n_with_annot}/{n_tiles} tiles tienen anotación. "
                           "Necesitas al menos 3 para hacer split train/val/test.",
                           value=float(n_with_annot))
            else:
                msg = (f"{n_complete} completos, {n_partial} parciales, "
                       f"{n_empty} sin anotar de {n_tiles} tiles totales.")
                if n_empty > n_tiles * 0.7:
                    report.add(WARNING, f"Muchos tiles vacíos ({n_empty}/{n_tiles})", msg,
                               value=pct_annot)
                else:
                    report.add(OK, f"Tiles anotados: {n_with_annot}/{n_tiles}", msg,
                               value=pct_annot)

            # Check split: ¿hay suficientes tiles para train/val/test?
            if n_with_annot >= 3:
                n_val  = max(1, int(n_with_annot * 0.20))
                n_test = max(1, int(n_with_annot * 0.10))
                n_train = n_with_annot - n_val - n_test
                if n_train < 1:
                    report.add(ERROR, "Split imposible",
                               f"Con {n_with_annot} tiles no hay suficiente para "
                               "train/val/test. Anota más tiles.")
                elif n_train < 3:
                    report.add(WARNING, "Split ajustado",
                               f"Train: {n_train}, Val: {n_val}, Test: {n_test} tiles. "
                               "El modelo puede sobreajustarse con tan pocos tiles de train.",
                               value=float(n_train))
                else:
                    report.add(OK, f"Split válido",
                               f"Train: {n_train}, Val: {n_val}, Test: {n_test} tiles.",
                               value=float(n_train))

    # ── Check 6: Volumen mínimo de datos ──────────────────────────────────────
    MIN_PTS_OK      = 1_000_000   # 1M pts mínimo razonable
    MIN_PTS_WARNING = 100_000     # 100K pts — muy poco
    if n_labeled < MIN_PTS_WARNING:
        report.add(ERROR, f"Muy pocos puntos ({n_labeled/1e3:.0f}K)",
                   "Menos de 100K pts etiquetados. "
                   "Los modelos de ML necesitan millones de puntos para aprender bien.")
    elif n_labeled < MIN_PTS_OK:
        report.add(WARNING, f"Pocos puntos ({n_labeled/1e3:.0f}K)",
                   f"Se recomiendan al menos 1M pts etiquetados para entrenar bien. "
                   f"Tienes {n_labeled/1e3:.0f}K.",
                   value=float(n_labeled))

    return report
