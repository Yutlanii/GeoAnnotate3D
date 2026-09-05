"""
tests/verify_fixes.py — Arnés de verificación headless (sin ventana VTK real).

Corre partes REALES del código (LabelStore, TileManager, sphere_query_octree,
ToolPanel, compute_colors_u8) contra datos sintéticos, en modo offscreen
(QT_QPA_PLATFORM=offscreen), para comprobar bugs concretos ANTES de reportar
un cambio como terminado.

LIMITACIÓN CONOCIDA: `AnnotationCanvas` (render/canvas.py) crea un
QVTKRenderWindowInteractor con una ventana OpenGL nativa de Win32 — en este
entorno sandboxeado, sin driver de GPU real, VTK hace SEGFAULT al intentar
crear el contexto OpenGL incluso con QT_QPA_PLATFORM=offscreen (confirmado:
"failed to get valid pixel format"). Por eso este arnés NO instancia
AnnotationCanvas ni verifica el render 3D en sí — prueba la lógica de datos
que alimenta al render (selección espacial, colores, cache de tiles, stats).
El render visual real todavía requiere que el usuario lo confirme en su
máquina con GPU.

Uso:
    QT_QPA_PLATFORM=offscreen python tests/verify_fixes.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication

app = QApplication.instance() or QApplication(["verify_fixes"])

FAILURES = []


def check(name, cond, detail=""):
    status = "OK  " if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def wait_ms(ms):
    """Procesa el event loop de Qt durante ms milisegundos (para QTimer)."""
    from PyQt5.QtCore import QTimer, QEventLoop
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


# ══════════════════════════════════════════════════════════════════════════
# TEST A — sphere_query_octree: debía devolver SIEMPRE vacío para nubes
# grandes (grid preferido + sin BFS). Verificamos que ahora usa el grid.
# ══════════════════════════════════════════════════════════════════════════
def test_sphere_query_octree_grid_path():
    from utils.spatial import sphere_query_octree
    from core._fast import FC

    if not FC.available:
        print("[SKIP] test_sphere_query_octree_grid_path — _fastcore no compilado")
        return

    rng = np.random.default_rng(0)
    xyz = rng.uniform(-50, 50, size=(200_000, 3)).astype(np.float32)

    class FakeOctree:
        """Simula exactamente el estado de un Octree real para una nube
        >=30M pts (grid preferido) y >=50M pts (sin BFS, root=None) —
        el caso exacto que rompía sphere_query_octree."""
        pass

    oc = FakeOctree()
    oc.ready     = True
    oc._xyz_ref  = xyz
    oc._grid     = FC.grid_build(xyz, 5.0)
    oc._grid_ready = True
    oc._kdtree_ready = False   # "prefer grid" — como hace Octree.build() real
    oc._kdtree   = None
    oc.root      = None        # BFS saltado (nube grande)

    center = np.array([0.0, 0.0, 0.0], np.float32)
    radius = 10.0
    result = sphere_query_octree(oc, center, radius)

    # Verificación cruzada con fuerza bruta (ground truth)
    d2 = ((xyz - center) ** 2).sum(1)
    expected = np.where(d2 <= radius * radius)[0]

    check("sphere_query_octree usa el grid (no devuelve vacío)",
          len(result) > 0,
          f"len(result)={len(result)}, esperado ~{len(expected)}")
    check("sphere_query_octree resultado == fuerza bruta (grid)",
          set(result.tolist()) == set(expected.tolist()),
          f"grid={len(result)} vs brute={len(expected)}")

    # Ahora el caso kdtree (root=None, grid apagado)
    from scipy.spatial import cKDTree
    oc2 = FakeOctree()
    oc2.ready = True; oc2._xyz_ref = xyz
    oc2._grid_ready = False; oc2._grid = None
    oc2._kdtree = cKDTree(xyz); oc2._kdtree_ready = True; oc2._kdtree_idx = None
    oc2.root = None
    result2 = sphere_query_octree(oc2, center, radius)
    check("sphere_query_octree usa cKDTree cuando no hay grid",
          set(result2.tolist()) == set(expected.tolist()),
          f"kdtree={len(result2)} vs brute={len(expected)}")

    # Caso sin nada listo (recién cargada) → debe caer a fuerza bruta real,
    # NO a vacío-por-hasattr-siempre-true (el bug original).
    oc3 = FakeOctree()
    oc3.ready = True; oc3._xyz_ref = xyz
    oc3._grid_ready = False; oc3._grid = None
    oc3._kdtree_ready = False; oc3._kdtree = None
    oc3.root = None
    result3 = sphere_query_octree(oc3, center, radius)
    check("sphere_query_octree cae a fuerza bruta genuina (sin índices listos)",
          set(result3.tolist()) == set(expected.tolist()),
          f"brute-fallback={len(result3)} vs brute={len(expected)}")


# ══════════════════════════════════════════════════════════════════════════
# TEST B — TileManager: mover la grilla debía reusar el índice de tiles
# cacheado de ANTES de moverla (mismo nombre de archivo). Verificamos que
# ahora dos transforms distintos producen rutas de caché distintas.
# ══════════════════════════════════════════════════════════════════════════
def test_tile_manager_cache_path_changes_with_transform():
    from core.tile_manager import TileManager
    from core.project import Project

    class FakePC:
        pass

    rng = np.random.default_rng(1)
    xyz = rng.uniform(0, 200, size=(50_000, 3)).astype(np.float32)
    xyz[:, 2] = 0.0

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        src = str(Path(td) / "fake_cloud.las")
        Path(src).write_bytes(b"0")  # solo necesita existir para el hash/nombre

        pc = FakePC()
        pc.xyz = xyz
        pc.bounds = np.array([xyz.min(0), xyz.max(0)], np.float32)
        pc.source_file = src
        pc._cache_path = None
        pc.is_mmap = False

        project = Project.new(src)
        project.init_labels(len(xyz))

        # TileManager "antes de mover la grilla"
        tm1 = TileManager(pc, project, tile_size_m=50.0,
                          offset_x=0.0, offset_y=0.0, rotation_deg=0.0)
        path1 = tm1._tile_index_path()

        # TileManager "después de mover la grilla" — instancia NUEVA, como
        # hace main_window._build_tile_manager() en cada cambio de transform
        tm2 = TileManager(pc, project, tile_size_m=50.0,
                          offset_x=17.0, offset_y=-8.0, rotation_deg=15.0)
        path2 = tm2._tile_index_path()

        check("rutas de cache de tile-index difieren tras mover la grilla",
              path1 != path2, f"path1={path1}  path2={path2}")

        # Construir el índice real para tm1, luego "mover" a tm2 y verificar
        # que tm2 NO reutiliza ciegamente el archivo de tm1 (antes: mismo
        # nombre → tm2.build_tile_index() cargaba el índice de tm1).
        tm1.build_tile_index()
        found_before = tm1.get_tile_indices_fast(tm1.tiles[0])

        tm2.build_tile_index()
        # tm2 debe haber calculado SU PROPIO índice (archivo distinto),
        # no el heredado de tm1.
        check("TileManager movido construye su propio índice (no reutiliza el viejo)",
              tm2._tile_index_path() != tm1._tile_index_path())

        # Verificación funcional: un tile de tm2 (grid rotada/movida) debe
        # corresponder a la geometría de tm2, no a la de tm1. Buscar el
        # primer tile con puntos para que la verificación no se salte.
        checked_geometry = False
        for t2 in tm2.tiles:
            idx2 = tm2.get_tile_indices_fast(t2)
            if idx2 is not None and len(idx2) > 0:
                pts = xyz[idx2, :2]
                ox, oy, ca, sa = tm2.origin_x, tm2.origin_y, tm2.cos_rot, tm2.sin_rot
                dx = pts[:, 0] - ox; dy = pts[:, 1] - oy
                xr =  dx*ca + dy*sa
                yr = -dx*sa + dy*ca
                inside = ((xr >= t2.min_xr) & (xr < t2.max_xr) &
                          (yr >= t2.min_yr) & (yr < t2.max_yr))
                check("puntos del tile de tm2 caen dentro de los bounds de tm2 (geometría correcta)",
                      bool(inside.all()), f"{inside.sum()}/{len(inside)} dentro")
                checked_geometry = True
                break
        check("se encontró al menos un tile con puntos para verificar geometría",
              checked_geometry)

        # Liberar los memmaps del índice antes de que se borre el tempdir
        # (Windows no permite borrar un archivo con un mmap todavía abierto).
        try:
            del tm1._tile_index, tm2._tile_index, found_before
        except Exception:
            pass
        import gc; gc.collect()


# ══════════════════════════════════════════════════════════════════════════
# TEST C — LabelStore: debounce de stats_changed no debe perder la
# actualización final tras una ráfaga de annotate() (simulando pintar rápido).
# ══════════════════════════════════════════════════════════════════════════
def test_label_store_stats_debounce():
    from annotation.label_store import LabelStore

    ls = LabelStore()
    labels = np.zeros(1000, np.uint8)
    ls.attach(labels)

    received = []
    ls.stats_changed.connect(lambda: received.append(ls.per_class_counts()))

    # Ráfaga de 20 annotate() distintos — simula 20 "dabs" de pincel seguidos
    for i in range(20):
        ls.annotate(np.array([i], np.int32), class_id=1)

    check("stats_changed se coalesce (no 1 emisión por cada annotate)",
          len(received) < 20, f"emisiones={len(received)} (esperado << 20)")

    wait_ms(250)  # dejar que el QTimer de 150ms dispare

    check("tras esperar el debounce, stats_changed sí llegó al menos una vez",
          len(received) >= 1)
    check("el conteo final tras el debounce es correcto (20 pts en clase 1)",
          ls.per_class_counts().get(1) == 20,
          f"counts={ls.per_class_counts()}")


# ══════════════════════════════════════════════════════════════════════════
# TEST D — ToolPanel.set_color_mode sincroniza el combo sin recursión y sin
# re-emitir color_mode_changed (evita eco al auto-seleccionar modo).
# ══════════════════════════════════════════════════════════════════════════
def test_tool_panel_color_mode_sync():
    from ui.tool_panel import ToolPanel

    tp = ToolPanel()
    fired = []
    tp.color_mode_changed.connect(lambda m: fired.append(m))

    check("combo arranca en 'Anotación' (default)",
          tp._color_combo.currentText() == "Anotación")

    tp.set_color_mode("RGB")
    check("set_color_mode('RGB') actualiza el texto del combo",
          tp._color_combo.currentText() == "RGB",
          f"texto actual={tp._color_combo.currentText()!r}")
    check("set_color_mode NO re-emite color_mode_changed (sin eco)",
          len(fired) == 0, f"fired={fired}")

    # Un cambio real del usuario SÍ debe emitir la señal
    tp._color_combo.setCurrentText("Elevación")
    check("cambio manual del usuario sí emite color_mode_changed",
          fired == ["Elevación"], f"fired={fired}")


# ══════════════════════════════════════════════════════════════════════════
# TEST E — compute_colors_u8: el modo Anotación debe reflejar el label real
# (sanity check de la ruta de color que alimenta refresh_colors_partial).
# ══════════════════════════════════════════════════════════════════════════
def test_compute_colors_u8_annotation_reflects_labels():
    from render.colors import compute_colors_u8, build_annotation_lut_u8
    from core.project import DEFAULT_SCHEMA

    xyz = np.zeros((3, 3), np.float32)
    lut = build_annotation_lut_u8(DEFAULT_SCHEMA)
    labels = np.array([0, 1, 3], np.uint8)  # sin etiquetar / suelo / árbol

    col = compute_colors_u8(xyz, {}, "Anotación", annotation_labels=labels,
                            annotation_lut_u8=lut)

    check("compute_colors_u8 devuelve un color distinto por clase",
          not np.array_equal(col[0], col[1]) and not np.array_equal(col[1], col[2]),
          f"col={col.tolist()}")
    expected_ground = lut[1]
    check("el color de la clase 'suelo' (id=1) coincide con la LUT",
          np.array_equal(col[1], expected_ground))


# ══════════════════════════════════════════════════════════════════════════
# TEST F — Rediseño de UI (2026-09-05): iconos reales cargan, el riel de
# navegación emite las señales correctas, y la duplicación del panel de
# capas fue eliminada de ToolPanel (la funcionalidad real vive solo en
# OverlayPanel, accesible desde "Capas de referencia" en el riel).
# ══════════════════════════════════════════════════════════════════════════
def test_redesign_icons_and_rail():
    from ui.icons import pixmap, available
    from ui.tool_panel import ToolPanel, _TOOL_ICON
    from ui.class_panel import ClassPanel

    # Todas las herramientas registradas tienen un icono real (antes Disco y
    # Region Growing quedaban sin dibujar — _draw_tool_icon no tenía su caso).
    from annotation.tools import ALL_TOOLS
    missing = [t.name for t in ALL_TOOLS if t.name not in _TOOL_ICON]
    check("todas las herramientas de ALL_TOOLS tienen icono real asignado",
          len(missing) == 0, f"sin icono: {missing}")

    for t in ALL_TOOLS:
        name = _TOOL_ICON.get(t.name)
        if name:
            check(f"icono real de '{t.name}' ({name}.svg) no está vacío",
                  available(name) and not pixmap(name, "#2c5480", 16).isNull())

    tp = ToolPanel()
    check("ToolPanel ya NO expone señales de capas (duplicación eliminada)",
          not hasattr(tp, "layer_add_requested") and
          not hasattr(tp, "layer_toggle_requested") and
          not hasattr(tp, "layer_remove_requested"))

    import ui.main_window as mw
    rail_got = []
    rail = mw._StepRail()
    rail.step_clicked.connect(lambda s: rail_got.append(("step", s)))
    rail.capas_clicked.connect(lambda: rail_got.append(("capas",)))
    rail.set_active_step(3)
    check("riel: paso 3 queda 'active', pasos 1-2 quedan 'done'",
          rail._items[3]._state == "active" and
          rail._items[1]._state == "done" and rail._items[2]._state == "done")
    rail._items[4].clicked.emit()
    rail._capas_item.clicked.emit()
    check("riel: clics emiten step_clicked/capas_clicked correctamente",
          rail_got == [("step", 4), ("capas",)], f"got={rail_got}")

    # OverlayPanel sigue siendo la ÚNICA fuente real de gestión de capas
    from ui.overlay_panel import OverlayPanel
    op = OverlayPanel(None)
    check("OverlayPanel (única gestión de capas) sigue construyéndose sin error",
          op is not None)


if __name__ == "__main__":
    tests = [
        test_sphere_query_octree_grid_path,
        test_tile_manager_cache_path_changes_with_transform,
        test_label_store_stats_debounce,
        test_tool_panel_color_mode_sync,
        test_compute_colors_u8_annotation_reflects_labels,
        test_redesign_icons_and_rail,
    ]
    for t in tests:
        print(f"\n── {t.__name__} ──")
        try:
            t()
        except Exception as exc:
            import traceback
            print(f"[ERROR] {t.__name__} lanzó una excepción:")
            traceback.print_exc()
            FAILURES.append(f"{t.__name__} (exception: {exc})")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULTADO: {len(FAILURES)} fallo(s):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("RESULTADO: todas las verificaciones pasaron ✓")
        sys.exit(0)
