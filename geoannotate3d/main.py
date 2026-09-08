#!/usr/bin/env python3
"""
GeoAnnotate3D — Semantic annotation tool for geospatial point clouds.
Usage:
    python main.py
    python main.py path/to/cloud.laz
"""
import sys
import os

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
           
APP_NAME    = "GeoAnnotate3D"
APP_VERSION = "1.0.0"


def _apply_dark_titlebar(hwnd: int) -> None:
    try:
        import ctypes
        for attr in (20, 19):
            try:
                value = ctypes.c_int(1)
                res = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attr, ctypes.byref(value), ctypes.sizeof(value))
                if res == 0:
                    break
            except Exception:
                pass
    except Exception:
        pass


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setFont(QFont("Segoe UI", 10))

    # ── Splash screen ─────────────────────────────────────────────────────────
    from ui.splash import SplashScreen
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # ── Carga progresiva con feedback visual ──────────────────────────────────
    splash.show_message("Compilando motor C…", progress=0.05)
    try:
        from core.build_extension import build_if_needed
        build_if_needed()
    except Exception:
        pass
    splash.show_message("Motor C listo ✓", progress=0.15)

    splash.show_message("Cargando motor 3D (VTK)…", progress=0.20)
    import vtk  # noqa — import pesado
    splash.show_message("VTK listo ✓", progress=0.45)

    splash.show_message("Cargando módulos de procesamiento…", progress=0.50)
    import numpy  # noqa
    try: import scipy  # noqa
    except ImportError: pass
    try: import laspy  # noqa
    except ImportError: pass
    splash.show_message("Módulos listos ✓", progress=0.65)

    splash.show_message("Inicializando interfaz…", progress=0.70)
    from ui.main_window import MainWindow
    splash.show_message("Construyendo ventana…", progress=0.85)

    # ── Ventana principal ─────────────────────────────────────────────────────
    window = MainWindow()
    splash.show_message("Listo  ✓", progress=1.0)
    window.showMaximized()
    app.processEvents()

    # Cerrar splash y mostrar onboarding cuando todo esté listo
    def _after_splash():
        splash.close()
        # Onboarding (6 pasos) — solo si no lo ha visto antes.
        # Antes había un SEGUNDO diálogo ("Abrir nube de puntos…") que
        # salía justo después de este, redundante — mismo contenido
        # (pasos 1-4, botón abrir nube) ya cubierto aquí. Quitado.
        from ui.welcome_dialog import WelcomeDialog
        WelcomeDialog.show_if_needed(window)

    QTimer.singleShot(900, _after_splash)

    # Barra de título oscura en Windows
    def _dark_title():
        try:
            hwnd = int(window.winId())
            _apply_dark_titlebar(hwnd)
        except Exception:
            pass
    QTimer.singleShot(200, _dark_title)

    # Abrir archivo si se pasa como argumento
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        QTimer.singleShot(600, lambda: window.open_file(sys.argv[1]))

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
