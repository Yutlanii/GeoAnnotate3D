"""
ui/styles.py
============
QSS global de GeoAnnotate3D — REDISEÑO 2026-09-04.

Tema anterior: café/ámbar oscuro (rechazado por el usuario: "nada tipo
cyberpunk, nada de colores oscuros"). Tema nuevo: paleta clásica clara,
un solo acento azul acero, iconos reales (ver ui/icons.py), tipografía
del sistema (Segoe UI) en vez de monoespaciada forzada en toda la app —
el monoespaciado (Consolas) ahora se reserva para coordenadas/valores
numéricos, igual que en cualquier herramienta técnica clásica (no todo
el texto de la UI necesita leerse como una terminal).

Todos los valores de color vienen de ui/theme.py — es la única fuente
de verdad de la paleta. Si cambia el tema, cambia ahí, no aquí ni en
los strings inline de cada panel (ese era el problema del tema viejo:
~150 hex repetidos sueltos por todo ui/*.py).
"""
from ui.theme import (
    BG, SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE, TEXT_FAINT,
    ACCENT, ACCENT_STRONG, ACCENT_SOFT, ACCENT_BORDER,
    OK, WARN, FONT_UI,
)

QSS = f"""
/* ── Base ──────────────────────────────────────────────── */
* {{
    font-family: '{FONT_UI}', 'Segoe UI', system-ui, sans-serif;
    font-size: 12px;
    outline: none;
}}

QMainWindow,
QWidget {{
    background: {BG};
    color: {TEXT};
}}

/* ── Dock widgets ───────────────────────────────────────── */
QDockWidget {{
    color: {ACCENT_STRONG};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background: {SURFACE};
    padding: 8px 12px;
    border-bottom: 1px solid {BORDER};
    color: {ACCENT_STRONG};
    font-size:10.5px;
    letter-spacing: 1.6px;
    text-transform: uppercase;
}}

QDockWidget::close-button,
QDockWidget::float-button {{
    background: transparent;
    border: none;
    padding: 0;
}}

/* ── Separadores de dock ────────────────────────────────── */
QMainWindow::separator {{
    background: {BORDER};
    width: 1px;
    height: 1px;
}}

/* ── Scroll bars ────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {SURFACE_2};
    width: 8px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: #c9cdd3;
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: #b3b8c0; }}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {SURFACE_2};
    height: 8px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: #c9cdd3;
    border-radius: 4px;
    min-width: 20px;
}}
QScrollBar::handle:horizontal:hover {{ background: #b3b8c0; }}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Botones ────────────────────────────────────────────── */
QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT_DIM};
    padding: 6px 14px;
}}
QPushButton:hover {{
    border-color: {ACCENT_BORDER};
    color: {ACCENT_STRONG};
    background: {ACCENT_SOFT};
}}
QPushButton:pressed {{
    background: {SURFACE_3};
}}
QPushButton:disabled {{
    color: {TEXT_FAINT};
    border-color: {BORDER_SOFT};
}}
QPushButton[accent="true"] {{
    background: {ACCENT};
    border-color: {ACCENT};
    color: #f4f5f6;
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{
    background: {ACCENT_STRONG};
    border-color: {ACCENT_STRONG};
}}

/* ── Labels ─────────────────────────────────────────────── */
QLabel {{
    color: {TEXT};
    background: transparent;
}}
QLabel[role="section"] {{
    color: {TEXT_MUTE};
    font-size:10.5px;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}}
QLabel[role="value"] {{
    color: {ACCENT_STRONG};
}}
QLabel[role="muted"] {{
    color: {TEXT_MUTE};
}}

/* ── Sliders ────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    background: {SURFACE_3};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 13px;
    height: 13px;
    margin: -5px 0;
    border-radius: 7px;
    border: 2px solid #f4f5f6;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}

/* ── ComboBox ───────────────────────────────────────────── */
QComboBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 8px;
    color: {TEXT_DIM};
}}
QComboBox:hover {{
    border-color: {ACCENT_BORDER};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    color: {TEXT};
    selection-background-color: {ACCENT_SOFT};
    selection-color: {ACCENT_STRONG};
}}

/* ── CheckBox ───────────────────────────────────────────── */
QCheckBox {{
    color: {TEXT_DIM};
    spacing: 7px;
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {SURFACE};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT_BORDER};
}}

/* ── Line Edit ──────────────────────────────────────────── */
QLineEdit {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 8px;
    color: {TEXT};
}}
QLineEdit:focus {{
    border-color: {ACCENT};
}}

/* ── Progress bar ───────────────────────────────────────── */
QProgressBar {{
    background: {SURFACE_3};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 3px;
}}

/* ── Status bar ─────────────────────────────────────────── */
QStatusBar {{
    background: {SURFACE};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTE};
}}
QStatusBar QLabel {{
    color: {TEXT_MUTE};
    font-size:10.5px;
    padding: 0 6px;
}}

/* ── Separadores de línea ───────────────────────────────── */
QFrame[frameShape="4"],
QFrame[frameShape="5"] {{
    color: {BORDER_SOFT};
}}

/* ── Tool tips ──────────────────────────────────────────── */
/* En Windows, QToolTip a veces solo honra `color` de la QSS y no
   `background` (el fondo nativo del tooltip del SO se cuela) — con el
   esquema anterior (fondo oscuro + texto casi blanco) eso producía
   texto muy claro sobre un fondo claro nativo, casi ilegible (reportado
   por el usuario). Fondo claro + texto oscuro es seguro en ambos casos:
   si la QSS se aplica del todo, se ve bien; si solo el `color` se
   aplica y el fondo nativo (claro) se cuela, sigue siendo legible. */
QToolTip {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    color: {TEXT};
    padding: 5px 9px;
    border-radius: 5px;
    font-size: 10.5px;
}}

/* ── Menus ──────────────────────────────────────────────── */
QMenu {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 4px 0;
    color: {TEXT_DIM};
}}
QMenu::item {{
    padding: 7px 32px 7px 14px;
    font-size: 11px;
}}
QMenu::item:selected {{
    background: {ACCENT_SOFT};
    color: {ACCENT_STRONG};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER_SOFT};
    margin: 3px 0;
}}

/* ── Message boxes ──────────────────────────────────────── */
QMessageBox {{
    background: {SURFACE};
}}
QMessageBox QLabel {{
    color: {TEXT};
}}

/* ── Dialog ─────────────────────────────────────────────── */
QDialog {{
    background: {SURFACE};
    border: 1px solid {BORDER};
}}

/* ── Tabs (paneles con pestañas: ToolPanel Herramientas/Vista/Capas) ── */
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTE};
    padding: 8px 14px;
    font-size: 11.5px;
    font-weight: 600;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{
    color: {ACCENT_STRONG};
    border-bottom: 2px solid {ACCENT};
}}
QTabWidget::pane {{
    border: none;
    border-top: 1px solid {BORDER};
}}

/* ── Tablas ──────────────────────────────────────────────── */
QTableWidget {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    color: {TEXT};
    gridline-color: {BORDER_SOFT};
    font-size: 11px;
}}
QHeaderView::section {{
    background: {SURFACE_2};
    color: {TEXT_MUTE};
    border: none;
    padding: 6px;
    font-size:10.5px;
    border-bottom: 1px solid {BORDER};
}}
"""
