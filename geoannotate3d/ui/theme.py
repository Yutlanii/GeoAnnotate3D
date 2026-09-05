"""
ui/theme.py — Paleta central de GeoAnnotate3D.
=================================================================
Única fuente de verdad de la paleta — todo lo demás importa de aquí en
vez de escribir hex sueltos.

Historial de tema:
  1. Café/ámbar oscuro (original) — rechazado: "nada tipo cyberpunk".
  2. Blanco + acento azul acero (2026-09-05, primera mitad de sesión).
  3. ACTUAL — Gris neutro + acento cian apagado, estilo Autodesk/CAD
     clásico (AutoCAD, Revit, ReCap): gris carbón medio (nunca blanco
     puro ni negro), un solo acento discreto solo en lo activo/seleccionado.
     Plano, sobrio, esquinas casi rectas (RADIUS bajo) — nada llamativo.

Excepciones que NO son "tema" (se quedan como están, son datos):
  - El viewport 3D (dark) — ver BG_R,G,B en render/canvas.py.
  - Colores de clases semánticas (SemanticClass.color, core/project.py).
  - Gradientes de datos (ej. _AGLBar en geo_panel.py).
"""
from __future__ import annotations

# ── Neutrales (gris medio, NUNCA blanco puro) ────────────────────────────
BG          = "#e8e9eb"   # fondo general de la ventana
SURFACE     = "#f4f5f6"   # paneles, tarjetas
SURFACE_2   = "#eceded"   # superficie secundaria (barras de herramientas, inputs)
SURFACE_3   = "#dadcde"   # pistas de slider/progreso, chips deshabilitados
BORDER      = "#c9cbce"   # borde estándar
BORDER_SOFT = "#d6d8da"   # divisor sutil

TEXT         = "#2b2d30"  # texto primario
TEXT_DIM     = "#55585c"  # texto secundario
TEXT_MUTE    = "#84888c"  # texto muted / placeholder
TEXT_FAINT   = "#a8acb0"  # texto muy tenue (deshabilitado)

# ── Acento (cian apagado — clásico CAD, discreto) ────────────────────────
ACCENT         = "#0e7c86"
ACCENT_STRONG  = "#0a5f67"   # hover/pressed
ACCENT_SOFT    = "#dceef0"   # tinte de fondo para fila/estado activo
ACCENT_BORDER  = "#a8d3d6"

# ── Estados ───────────────────────────────────────────────────────────────
OK       = "#3f8a5c"   # guardado / completado / éxito
OK_SOFT  = "#e2f0e6"
WARN     = "#ad4b32"   # modo borrar / error / peligro
WARN_SOFT= "#f5e6e1"
INFO_TEAL= "#1a7a6e"   # capa raster (para diferenciar de vectorial=accent)

# ── Tipografía ────────────────────────────────────────────────────────────
FONT_UI   = "Segoe UI"        # clásica en Windows, ya es la base de main.py
FONT_MONO = "Consolas"        # reservada para coordenadas/valores numéricos

# ── Radios / tamaños compartidos ─────────────────────────────────────────
# Bajos a propósito: el estilo CAD clásico es plano, casi sin esquinas
# redondeadas (a diferencia del tema claro anterior, más "SaaS moderno").
RADIUS    = 4
RADIUS_SM = 3


def qcolor_alpha(hexcolor: str, alpha_hex: str) -> str:
    """'#0e7c86' + 'aa' -> '#aa0e7c86' estilo ARGB, para tintes puntuales."""
    return f"#{alpha_hex}{hexcolor.lstrip('#')}"
