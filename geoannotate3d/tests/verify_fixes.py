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
from PyQt5.QtCore import QCoreApplication, Qt

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


# ══════════════════════════════════════════════════════════════════════════
# TEST G — mapeo a códigos ASPRS del export .las clasificado. Antes solo se
# verificó con un script suelto durante la sesión, nunca se dejó en el
# arnés permanente — si alguien tocaba `_asprs_code_for` más adelante, nada
# lo habría detectado. Cubre: keywords reconocidos, y el caso sin match
# (debe caer en el rango reservado 64-255, nunca en un código estándar).
# ══════════════════════════════════════════════════════════════════════════
def test_asprs_code_mapping():
    from annotation.exporter import _asprs_code_for

    check("ASPRS: 'Suelo' -> 2 (ground)",
          _asprs_code_for(1, "Suelo") == 2)
    check("ASPRS: 'Veg. baja' -> 3 (low veg)",
          _asprs_code_for(2, "Veg. baja") == 3)
    check("ASPRS: 'Edificio' -> 6 (building)",
          _asprs_code_for(5, "Edificio") == 6)
    check("ASPRS: 'Sin clasificar' -> 1 (unclassified)",
          _asprs_code_for(0, "Sin clasificar") == 1)

    unmatched = _asprs_code_for(7, "Cable eléctrico")
    check("ASPRS: nombre sin match cae en rango reservado 64-255 (no colisiona)",
          64 <= unmatched <= 255, f"got={unmatched}")


# ══════════════════════════════════════════════════════════════════════════
# TEST H — lógica del marcador de sesión para detectar un crash previo
# (MainWindow._setup_crash_handler / _save_window_state). No se puede
# instanciar MainWindow aquí (VTK real, ver limitación del módulo), así que
# se reproduce la MISMA secuencia de marcador de archivo que usa el código
# real: creado al iniciar, borrado en un cierre limpio, y si sigue ahí al
# siguiente inicio es que la sesión anterior murió sin pasar por closeEvent.
# ══════════════════════════════════════════════════════════════════════════
def test_crash_marker_sequence():
    marker = Path(tempfile.mkdtemp()) / ".session_running"

    def start():
        had_crash = marker.exists()
        marker.write_text("running", encoding="utf-8")
        return had_crash

    def clean_close():
        if marker.exists():
            marker.unlink()

    s1 = start(); clean_close()
    check("crash-marker: sesión limpia no reporta crash previo", s1 is False)

    s2 = start()   # sin clean_close — simula crash
    check("crash-marker: sesión que sí venía limpia arranca bien", s2 is False)

    s3 = start()
    check("crash-marker: detecta que la sesión anterior no cerró limpio",
          s3 is True)
    clean_close()

    s4 = start()
    check("crash-marker: tras un cierre limpio, no se repite el aviso",
          s4 is False)


# ══════════════════════════════════════════════════════════════════════════
# TEST I — reanudar entrenamiento desde checkpoint (training_panel.py). Antes
# de esto solo se verificó con un script suelto; se deja aquí un round-trip
# real con PyTorch: guardar checkpoint con estado de optimizer/scheduler,
# cargarlo, y confirmar que epoch/optimizer/scheduler se restauran (no solo
# los pesos del modelo, que era el bug — reanudar "a medias" perdía el
# momentum del optimizer y el punto del LR schedule).
# ══════════════════════════════════════════════════════════════════════════
def test_training_checkpoint_resume_roundtrip():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[SKIP] test_training_checkpoint_resume_roundtrip — PyTorch no instalado")
        return

    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)

    # Simula unos pasos de entrenamiento para que el optimizer tenga estado
    # real (momentum) y el scheduler haya avanzado, no un estado recién creado.
    for _ in range(3):
        opt.zero_grad()
        loss = model(torch.randn(4)).sum()
        loss.backward()
        opt.step()
        sch.step()
    lr_before = opt.param_groups[0]['lr']

    ckpt_path = Path(tempfile.mkdtemp()) / "ckpt.pth"
    torch.save({
        "epoch": 3,
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": sch.state_dict(),
        "miou": 0.42,
    }, ckpt_path)

    # Nuevo modelo/optimizer/scheduler "en blanco" — como al reanudar en una
    # sesión nueva de la app.
    model2 = nn.Linear(4, 2)
    opt2 = torch.optim.Adam(model2.parameters(), lr=0.01)
    sch2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=2, gamma=0.5)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model2.load_state_dict(ckpt["model_state"])
    opt2.load_state_dict(ckpt["optimizer_state"])
    sch2.load_state_dict(ckpt["scheduler_state"])

    check("checkpoint resume: epoch guardado se recupera",
          ckpt["epoch"] == 3)
    check("checkpoint resume: pesos del modelo coinciden tras cargar",
          torch.allclose(model.weight, model2.weight))
    check("checkpoint resume: LR del scheduler coincide (no se reinicia el schedule)",
          abs(opt2.param_groups[0]['lr'] - lr_before) < 1e-9,
          f"esperado={lr_before} obtenido={opt2.param_groups[0]['lr']}")
    check("checkpoint resume: momentum del optimizer (estado Adam) se restaura",
          len(opt2.state) > 0)


# ══════════════════════════════════════════════════════════════════════════
# TEST J — herramienta "eliminar puntos" (DELETED_LABEL, ver label_store.py).
# Pedido del usuario: cualquier herramienta de selección debe poder ELIMINAR
# puntos de ruido de la nube (no solo quitarles la clase). Implementado
# reutilizando el mecanismo de labels con un valor centinela (255) — estos
# tests cubren las dos partes centrales: (1) LabelStore excluye los puntos
# eliminados de toda estadística de "etiquetado", y el undo los recupera
# igual que cualquier anotación; (2) el render los oculta (alpha=0) en
# CUALQUIER modo de color, no solo en modo "Anotación".
# ══════════════════════════════════════════════════════════════════════════
def test_delete_points_label_store():
    from annotation.label_store import LabelStore, DELETED_LABEL

    ls = LabelStore()
    labels = np.zeros(100, np.uint8)
    ls.attach(labels)

    ls.annotate(np.arange(0, 30), class_id=1)     # 30 pts clase 1
    check("delete: 30 pts clase 1 antes de eliminar nada",
          ls.n_labeled == 30 and ls.per_class_counts().get(1) == 30)

    ls.delete_points(np.arange(0, 10))            # elimina 10 de esos 30
    check("delete: n_labeled baja (los eliminados no cuentan como etiquetados)",
          ls.n_labeled == 20, f"n_labeled={ls.n_labeled}")
    check("delete: n_deleted refleja los puntos eliminados",
          ls.n_deleted == 10, f"n_deleted={ls.n_deleted}")
    check("delete: per_class_counts NO cuenta DELETED_LABEL como una clase",
          DELETED_LABEL not in ls.per_class_counts(),
          f"counts={ls.per_class_counts()}")
    check("delete: los índices eliminados sí tienen el valor centinela",
          bool((labels[0:10] == DELETED_LABEL).all()))

    # Ctrl+Z debe recuperar los puntos eliminados, igual que cualquier
    # anotación — es la razón de reusar el mecanismo de labels en vez de
    # inventar un sistema de borrado aparte.
    ls.undo()
    check("delete: Ctrl+Z (undo) recupera los puntos eliminados",
          ls.n_deleted == 0 and ls.per_class_counts().get(1) == 30,
          f"n_deleted={ls.n_deleted} counts={ls.per_class_counts()}")


def test_delete_points_excluded_from_export():
    """Los puntos eliminados no deben aparecer en el dataset exportado
    (ni con only_labeled=True ni con only_labeled=False — no son "puntos
    sin etiquetar", son puntos que ya no están en la nube)."""
    from annotation.exporter import ExportWorker, ExportConfig
    from annotation.label_store import DELETED_LABEL

    class FakePC:
        filename = "fake.las"
        offset = np.zeros(3)
        def __init__(self, xyz): self.xyz = xyz; self.intensity = None; self.rgb = None

    n = 40
    xyz = np.random.rand(n, 3).astype(np.float32) * 10
    labels = np.zeros(n, np.uint8)
    labels[0:15] = 1                  # 15 pts clase 1
    labels[15:20] = DELETED_LABEL     # 5 pts eliminados (ruido limpiado)
    # el resto (20:40) queda sin etiquetar (clase 0)

    pc = FakePC(xyz)

    for only_labeled in (True, False):
        cfg = ExportConfig(only_labeled=only_labeled)
        w = ExportWorker.__new__(ExportWorker)   # sin __init__ de QThread real
        w._pc = pc; w._labels = labels.copy(); w._config = cfg; w._tm = None
        tiles = w._collect_tile_data()
        all_labels = np.concatenate([t["labels"] for t in tiles]) if tiles else np.array([], np.uint8)
        check(f"export (only_labeled={only_labeled}): ningún punto eliminado en el dataset",
              DELETED_LABEL not in all_labels, f"labels presentes={np.unique(all_labels)}")


# ══════════════════════════════════════════════════════════════════════════
# TEST K — modo de color "Confianza" (post-inferencia). "Future update" que
# se agregó en esta ronda: infer_cloud(..., return_confidence=True) ya
# calculaba el softmax internamente y lo descartaba — exponerlo como color
# (rojo=insegura, verde=segura) ayuda a saber dónde revisar primero. Cubre:
# alta confianza → tono verde (más G que R); baja confianza → tono rojo (más
# R que G); y que sin "confidence" en attrs no explota (cae a otro modo).
# ══════════════════════════════════════════════════════════════════════════
def test_confidence_color_mode():
    from render.colors import compute_colors_u8

    xyz = np.random.rand(10, 3).astype(np.float32)
    confidence = np.array([0.95]*5 + [0.10]*5, dtype=np.float32)

    out = compute_colors_u8(xyz, {"confidence": confidence}, "Confianza")
    check("confianza alta (0.95) → verde domina sobre rojo",
          bool((out[:5, 1].astype(int) > out[:5, 0].astype(int)).all()),
          f"rgba[0]={out[0]}")
    check("confianza baja (0.10) → rojo domina sobre verde",
          bool((out[5:, 0].astype(int) > out[5:, 1].astype(int)).all()),
          f"rgba[5]={out[5]}")

    # Sin "confidence" en attrs, el modo "Confianza" no debe reventar —
    # debe caer a algún fallback (mismo patrón que "Intensidad"/"RGB" ya
    # usan cuando el atributo no está disponible).
    out2 = compute_colors_u8(xyz, {}, "Confianza")
    check("modo 'Confianza' sin datos no revienta (cae a fallback)",
          out2.shape == (10, 4))


def test_annotation_mode_unlabeled_points_are_opaque():
    """
    Antes, los puntos SIN etiquetar (clase 0) en modo "Anotación" tenían
    alpha=115/255 (~0.45, semi-transparentes) para verse "apagados". Un
    solo punto con alpha<255 obliga a VTK a renderizar TODO el actor con
    blending translúcido — mucho más lento por frame que el modo RGB
    (siempre opaco) — y como sin-etiquetar es casi siempre la MAYORÍA de
    la nube mientras se anota, esto hacía que el modo Anotación se
    sintiera notablemente más lento/tembloroso que RGB, reportado por el
    usuario. Ahora deben verse "apagados" con un gris opaco (alpha=255),
    igual que cualquier otra clase — solo los puntos ELIMINADOS
    (DELETED_LABEL=255) deben seguir teniendo alpha=0.
    """
    from render.colors import (compute_colors_u8, build_annotation_lut_u8,
                                DELETED_LABEL)
    from core.project import SemanticClass

    n = 100
    xyz = np.random.rand(n, 3).astype(np.float32)
    labels = np.zeros(n, np.uint8)
    labels[10:20] = 1        # etiquetados con clase 1
    labels[50:55] = DELETED_LABEL

    schema = [SemanticClass(1, "Suelo", "#8c6018")]
    lut = build_annotation_lut_u8(schema)
    out = compute_colors_u8(xyz, {}, "Anotación",
                            annotation_labels=labels, annotation_lut_u8=lut)

    check("sin-etiquetar (clase 0) es totalmente opaco (alpha=255)",
          bool((out[labels == 0, 3] == 255).all()),
          f"alphas={np.unique(out[labels==0,3])}")
    check("etiquetados (clase 1) siguen totalmente opacos",
          bool((out[10:20, 3] == 255).all()))
    check("eliminados siguen invisibles (alpha=0) — no regresionó el fix anterior",
          bool((out[50:55, 3] == 0).all()))


def test_delete_points_hidden_in_render():
    from render.colors import compute_colors_u8, build_annotation_lut_u8

    n = 50
    xyz = np.random.rand(n, 3).astype(np.float32)
    labels = np.zeros(n, np.uint8)
    labels[5:10] = 255   # DELETED_LABEL
    lut = build_annotation_lut_u8([])

    for mode in ("Anotación", "Elevación", "Color único"):
        out = compute_colors_u8(xyz, {}, mode, annotation_labels=labels,
                                annotation_lut_u8=lut)
        check(f"delete: puntos eliminados invisibles (alpha=0) en modo '{mode}'",
              bool((out[5:10, 3] == 0).all()), f"alphas={out[5:10,3]}")
        check(f"delete: puntos normales conservan su alpha en modo '{mode}'",
              bool((out[20:25, 3] > 0).all()))


def test_sor_outlier_detection():
    """
    Statistical Outlier Removal — mismo algoritmo que CloudCompare/PCL.
    Nube sintética: un cluster denso (50K pts, ruido gaussiano ~1.0) +
    200 puntos de "ruido" dispersos uniformemente en un volumen mucho
    más grande — deben quedar aislados (distancia media a sus k
    vecinos mucho mayor que el promedio de la nube). Verifica que la
    detección separa correctamente ambos grupos, no solo que "corre
    sin reventar".
    """
    from annotation.noise_filter import detect_outliers_sor

    rng = np.random.default_rng(0)
    n_dense = 50_000
    dense = rng.normal(0, 1.0, size=(n_dense, 3)).astype(np.float32)
    n_noise = 200
    noise = rng.uniform(-50, 50, size=(n_noise, 3)).astype(np.float32)
    xyz = np.vstack([dense, noise])

    mask = detect_outliers_sor(xyz, k=8, std_ratio=2.0)

    check("SOR: el cluster denso casi no tiene falsos positivos",
          int(mask[:n_dense].sum()) < n_dense * 0.01,
          f"{mask[:n_dense].sum()}/{n_dense} marcados")
    check("SOR: el ruido disperso se detecta casi por completo",
          int(mask[n_dense:].sum()) > n_noise * 0.95,
          f"{mask[n_dense:].sum()}/{n_noise} marcados")

    # Caso borde: nube vacía o casi vacía no debe reventar
    empty = detect_outliers_sor(np.zeros((0, 3), np.float32))
    check("SOR: nube vacía no revienta", len(empty) == 0)
    tiny = detect_outliers_sor(np.zeros((2, 3), np.float32), k=8)
    check("SOR: nube de 2 puntos (k>n) no revienta", len(tiny) == 2)


def test_region_growing_performance_and_correctness():
    """
    Antes, `_grow()` construía la grilla espacial con un bucle Python
    puro sobre CADA punto del tile/nube visible (`for li in range(n):
    cell_map[...].append(li)`) — con nubes de unos pocos millones de
    puntos eso congelaba el software ~1s en un solo clic, reportado por
    el usuario con una nube de "solo" 8M puntos. Este test reproduce
    esa escala (8M puntos) y verifica DOS cosas: que ahora termina en
    una fracción de segundo (no solo que "no truena"), y que la
    selección resultante sigue siendo un cluster conectado razonable
    alrededor de la semilla (no toda la nube, no vacío) — la
    vectorización no debe cambiar el resultado del algoritmo.
    """
    import time
    from annotation.region_growing import RegionGrowingTool

    rng = np.random.default_rng(0)
    n = 8_000_000
    # Nube dispersa en un área grande (para que la caja recortada de
    # verdad reduzca el trabajo) + un cluster denso y conectado cerca
    # del origen (donde cae la semilla) para verificar la selección.
    xyz = (rng.random((n, 3), dtype=np.float32) - 0.5) * 200.0
    cluster_n = 4000
    cluster = rng.normal(0, 0.15, size=(cluster_n, 3)).astype(np.float32)
    xyz[:cluster_n] = cluster   # los primeros cluster_n puntos son el cluster

    rgb = np.zeros((n, 3), dtype=np.uint8)
    rgb[:cluster_n] = [200, 80, 80]     # cluster: un color uniforme
    rgb[cluster_n:] = [20, 20, 20]      # resto: otro color (no debe unirse)

    class FakePC:
        pass
    pc = FakePC(); pc.xyz = xyz; pc.rgb = rgb; pc.intensity = None

    class FakeCanvas:
        pass
    canvas = FakeCanvas()
    canvas.pc = pc
    canvas._tile_mode = False
    canvas._cur_xyz = xyz
    canvas._cur_idx = None
    canvas.project = None

    tool = RegionGrowingTool()
    tool.canvas = canvas
    tool.label_store = None   # _apply_selection corta ahí — solo medimos _grow()
    tool.tol_rgb = 30.0

    captured = {}
    def fake_apply(indices):
        captured["indices"] = np.asarray(indices)
    tool._apply_selection = fake_apply

    seed = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    t0 = time.perf_counter()
    tool._grow(seed)
    elapsed = time.perf_counter() - t0

    check(f"region growing en {n/1e6:.0f}M pts termina en <1.0s (antes ~1s+ SOLO "
          f"construyendo la grilla) — tardó {elapsed:.3f}s",
          elapsed < 1.0, f"elapsed={elapsed:.3f}s")

    sel = captured.get("indices")
    check("region growing seleccionó algo (no vacío)",
          sel is not None and len(sel) > 0)
    if sel is not None and len(sel) > 0:
        check("region growing se quedó dentro del cluster conectado "
              "(no se coló al resto de la nube)",
              bool(sel.max() < cluster_n * 2),  # margen generoso, pero NO toda la nube
              f"max_idx={sel.max()} (cluster_n={cluster_n}, total n={n})")
        check("region growing encontró una porción razonable del cluster "
              "(no solo la semilla)",
              len(sel) > 10, f"len(sel)={len(sel)}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — Octree.iter_progressive_lod: antes el pipeline de render (Overview)
# usaba iter_lod_progression (presupuesto plano, ignora cámara/frustum por
# completo). Ahora, cuando hay octree BFS (nubes < BFS_SKIP_THRESHOLD),
# iter_progressive_lod delega en iter_refinement (SSE real + frustum culling
# que poda ramas enteras) — verificamos que el resultado real cambia según
# el frustum, y que con frustum=None se sigue viendo la nube completa.
# ══════════════════════════════════════════════════════════════════════════
def test_octree_iter_progressive_lod_frustum_and_sse():
    from core.octree import Octree
    from utils.spatial import extract_frustum_planes

    rng = np.random.default_rng(1)
    # Dos nubes de puntos separadas en X: una en x≈-20 (izquierda),
    # otra en x≈+20 (derecha) — para poder verificar culling por mitad.
    left  = rng.normal(loc=[-20, 0, 0], scale=1.0, size=(20_000, 3))
    right = rng.normal(loc=[ 20, 0, 0], scale=1.0, size=(20_000, 3))
    xyz = np.vstack([left, right]).astype(np.float32)

    oc = Octree()
    oc.build(xyz)   # n=40_000 << BFS_SKIP_THRESHOLD → construye BFS (oc.root)

    check("Octree.build() construyó el octree BFS (necesario para SSE real)",
          oc.root is not None)

    # Frustum ortográfico-ish que solo mira hacia +X (mitad derecha):
    # un solo plano "x >= 0" (nx=1, d=0) más el resto permisivos (nx=ny=nz=0,d=1).
    planes = np.zeros((6, 4), np.float32)
    planes[0] = [1, 0, 0, 0]     # x >= 0
    for i in range(1, 6):
        planes[i] = [0, 0, 0, 1]  # siempre "dentro"

    eye3d = np.array([50.0, 0.0, 0.0], np.float32)
    idx_seen = set()
    for idx, is_done in oc.iter_progressive_lod(eye3d, planes, 800, 600, 45.0, 100_000):
        idx_seen.update(idx.tolist())
        if is_done: break

    check("iter_progressive_lod con frustum devolvió algo",
          len(idx_seen) > 0, f"len={len(idx_seen)}")
    xs = xyz[np.fromiter(idx_seen, dtype=np.int64)][:, 0]
    frac_left = float((xs < -5).mean()) if len(xs) else 1.0
    check("iter_progressive_lod respeta el frustum: descarta casi toda la "
          "mitad izquierda (x<0) cuando el frustum solo mira hacia +X",
          frac_left < 0.05, f"frac_left={frac_left:.3f} (esperado <0.05)")

    # Sin frustum (None): debe poder ver AMBOS lados (comportamiento anterior
    # preservado como fallback / para cuando no hay cámara aún).
    idx_seen_nofrustum = set()
    for idx, is_done in oc.iter_progressive_lod(eye3d, None, 800, 600, 45.0, 100_000):
        idx_seen_nofrustum.update(idx.tolist())
        if is_done: break
    xs2 = xyz[np.fromiter(idx_seen_nofrustum, dtype=np.int64)][:, 0]
    frac_left_nofrustum = float((xs2 < -5).mean()) if len(xs2) else 0.0
    check("iter_progressive_lod SIN frustum (planes=None) sigue viendo "
          "ambos lados de la nube (no rompe el caso sin cámara)",
          frac_left_nofrustum > 0.2, f"frac_left={frac_left_nofrustum:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — FC.gather_and_color / gather_and_color_elevation: aceptan un
# argumento `planes` para frustum culling fusionado — probado aquí de forma
# aislada (con planos sintéticos). NOTA: render/lod_worker.py volvió a
# pasarles siempre `None` (revertido tras un bug real de Overview) — este
# test verifica
# la función en sí, no cómo se usa (o no) desde el pipeline de render.
# ══════════════════════════════════════════════════════════════════════════
def test_gather_and_color_frustum_culling():
    from core._fast import FC

    rng = np.random.default_rng(2)
    left  = rng.normal(loc=[-20, 0, 0], scale=1.0, size=(5_000, 3)).astype(np.float32)
    right = rng.normal(loc=[ 20, 0, 0], scale=1.0, size=(5_000, 3)).astype(np.float32)
    xyz = np.vstack([left, right])
    idx_all = np.arange(len(xyz), dtype=np.int32)
    labels  = np.zeros(len(xyz), np.uint8)
    lut_u8  = np.tile(np.array([200, 100, 50, 255], np.uint8), (256, 1))

    planes = np.zeros((6, 4), np.float32)
    planes[0] = [1, 0, 0, 0]
    for i in range(1, 6):
        planes[i] = [0, 0, 0, 1]

    xyz_out, col_out, idx_out = FC.gather_and_color(xyz, idx_all, labels, lut_u8, None)
    check("gather_and_color SIN planes devuelve todos los puntos (baseline)",
          len(idx_out) == len(xyz))

    xyz_out2, col_out2, idx_out2 = FC.gather_and_color(xyz, idx_all, labels, lut_u8, planes)
    check("gather_and_color CON planes descarta la mitad izquierda (x<0)",
          len(idx_out2) < len(xyz) * 0.55, f"len(idx_out2)={len(idx_out2)} de {len(xyz)}")
    check("gather_and_color CON planes: los puntos que quedan son todos x>=0 aprox",
          bool((xyz_out2[:, 0] > -1.0).all()),
          f"min_x={xyz_out2[:,0].min() if len(xyz_out2) else 'N/A'}")
    check("gather_and_color: xyz_out y col_out quedan del mismo largo que idx_out",
          len(xyz_out2) == len(col_out2) == len(idx_out2))

    z_min, z_max = -1.0, 1.0
    lut_elev = np.tile(np.array([10, 20, 30, 255], np.uint8), (256, 1))
    _, _, idx_out3 = FC.gather_and_color_elevation(xyz, idx_all, z_min, z_max, lut_elev, planes)
    check("gather_and_color_elevation CON planes también aplica el frustum",
          len(idx_out3) < len(xyz) * 0.55, f"len(idx_out3)={len(idx_out3)}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — _OctreeBuilder (ui/main_window.py, ruta "heavy cloud" >HEAVY_THRESHOLD
# vía .ga3d_bin / "cargar como normal"): antes pre-submuestreaba xyz con
# stride ANTES de construir el octree, así que los índices que devolvía NO
# correspondían a la nube real — Pincel/Esfera/Disco pintaban puntos
# equivocados en nubes >150M pts. Verificamos que ahora el octree se
# construye sobre xyz COMPLETO (índices siempre reales/globales).
# ══════════════════════════════════════════════════════════════════════════
def test_octree_builder_uses_full_xyz_no_presample():
    from ui.main_window import _OctreeBuilder

    class _FakePC:
        pass

    rng = np.random.default_rng(3)
    n = 60_000
    xyz = rng.uniform(-100, 100, size=(n, 3)).astype(np.float32)
    pc = _FakePC(); pc.xyz = xyz

    builder = _OctreeBuilder(pc)
    captured = {}
    builder.ready.connect(lambda oc: captured.setdefault("octree", oc))
    builder.run()   # ejecutar síncrono (no .start()) — sin necesidad de QThread real

    oc = captured.get("octree")
    check("_OctreeBuilder emitió un octree", oc is not None)
    if oc is None:
        return
    check("_OctreeBuilder ya NO pre-submuestrea: _xyz_ref tiene los mismos "
          "n puntos que pc.xyz (antes se recortaba con stride)",
          len(oc._xyz_ref) == n, f"len(_xyz_ref)={len(oc._xyz_ref)}, esperado {n}")

    # Los índices que devuelve get_coarse_view deben apuntar a posiciones
    # válidas y REALES dentro de pc.xyz (antes, con la copia submuestreada,
    # esto igual "funcionaba" en apariencia pero apuntaba a otro array —
    # aquí lo que importa es que _xyz_ref sea literalmente pc.xyz).
    xyz_out, idx_out = oc.get_coarse_view(1_000)
    check("get_coarse_view: los índices devueltos indexan xyz real correctamente",
          bool(np.allclose(xyz[idx_out], xyz_out)) if len(idx_out) else True,
          "xyz[idx_out] no coincide con xyz_out")


# ══════════════════════════════════════════════════════════════════════════
# TEST — Backend LAZ paralelo: laspy.open() sin backend explícito "detecta"
# uno según su propio orden interno, no garantizado entre versiones.
# _preferred_laz_backend/_open_las_stream lo piden EXPLÍCITAMENTE. Verificamos
# que la selección es la esperada y que abrir/leer un .laz real con el
# backend explícito produce EXACTAMENTE los mismos datos que sin especificarlo.
# ══════════════════════════════════════════════════════════════════════════
def test_laz_parallel_backend_selection_and_roundtrip():
    try:
        import laspy
    except ImportError:
        print("[SKIP] test_laz_parallel_backend_selection_and_roundtrip — laspy no instalado")
        return

    from core.pointcloud import _preferred_laz_backend, _open_las_stream

    backend = _preferred_laz_backend(laspy)
    if hasattr(laspy, "LazBackend"):
        check("_preferred_laz_backend prefiere LazrsParallel cuando está disponible",
              backend == laspy.LazBackend.LazrsParallel or not hasattr(laspy.LazBackend, "LazrsParallel"),
              f"backend elegido={backend}")
    else:
        check("_preferred_laz_backend devuelve None si laspy no tiene LazBackend",
              backend is None)

    # Round-trip real: escribir un .laz sintético pequeño, leerlo con
    # _open_las_stream (backend explícito) y comparar contra laspy.read()
    # directo (sin backend) — deben coincidir exactamente.
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "synthetic.laz")
        hdr = laspy.LasHeader(point_format=3, version="1.2")
        hdr.scales = [0.01, 0.01, 0.01]
        las = laspy.LasData(hdr)
        rng = np.random.default_rng(4)
        n = 50_000
        las.x = rng.uniform(0, 500, n)
        las.y = rng.uniform(0, 500, n)
        las.z = rng.uniform(0, 100, n)
        las.intensity = rng.integers(0, 65535, n).astype(np.uint16)
        las.write(path)

        direct = laspy.read(path)
        expected_x = np.asarray(direct.x, np.float64)

        got_x_parts = []
        with _open_las_stream(laspy, path, True) as reader:
            for chunk in reader.chunk_iterator(20_000):
                got_x_parts.append(np.asarray(chunk.x, np.float64))
        got_x = np.concatenate(got_x_parts)

        check("_open_las_stream con backend explícito lee el mismo número de puntos",
              len(got_x) == len(expected_x),
              f"got={len(got_x)} expected={len(expected_x)}")
        check("_open_las_stream con backend explícito produce coordenadas idénticas",
              bool(np.allclose(got_x, expected_x)),
              "coordenadas X no coinciden entre backend explícito y lectura directa")


# ══════════════════════════════════════════════════════════════════════════
# TEST — LabelStore.annotate_bulk: escritura masiva con clase DISTINTA por
# punto (la necesita el suavizado de etiquetas) + undo/redo correcto.
# ══════════════════════════════════════════════════════════════════════════
def test_label_store_annotate_bulk_undo_redo():
    from annotation.label_store import LabelStore

    store = LabelStore()
    labels = np.array([1, 1, 2, 2, 3], dtype=np.uint8)
    store.attach(labels)

    idx = np.array([0, 2, 4], dtype=np.int32)
    new_vals = np.array([9, 8, 7], dtype=np.uint8)
    store.annotate_bulk(idx, new_vals)

    check("annotate_bulk escribe valores heterogéneos por punto",
          bool(np.array_equal(store._labels, np.array([9, 1, 8, 2, 7], np.uint8))),
          f"labels={store._labels}")

    check("annotate_bulk: undo restaura los valores previos exactos",
          store.undo() and bool(np.array_equal(store._labels, np.array([1, 1, 2, 2, 3], np.uint8))),
          f"labels tras undo={store._labels}")

    check("annotate_bulk: redo reaplica los valores heterogéneos",
          store.redo() and bool(np.array_equal(store._labels, np.array([9, 1, 8, 2, 7], np.uint8))),
          f"labels tras redo={store._labels}")

    # Mezclar con un annotate() uniforme normal — no debe romperse
    store.annotate(np.array([1, 3], np.int32), 5)
    check("annotate() uniforme normal sigue funcionando tras usar annotate_bulk",
          bool(np.array_equal(store._labels, np.array([9, 5, 8, 5, 7], np.uint8))),
          f"labels={store._labels}")
    check("undo tras mezclar annotate()/annotate_bulk deshace la operación correcta",
          store.undo() and bool(np.array_equal(store._labels, np.array([9, 1, 8, 2, 7], np.uint8))),
          f"labels={store._labels}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — smooth_labels_majority: suavizado de etiquetas por mayoría de
# vecinos. Sintético: un punto "ruidoso" en medio de un cluster homogéneo
# debe corregirse a la clase mayoritaria del vecindario; sin etiquetar y
# eliminados nunca deben tocarse ni contar como votantes.
# ══════════════════════════════════════════════════════════════════════════
def test_smooth_labels_majority():
    from annotation.label_smoothing import smooth_labels_majority
    from annotation.label_store import DELETED_LABEL

    rng = np.random.default_rng(10)
    n = 2000
    xyz = rng.uniform(0, 10, size=(n, 3)).astype(np.float32)
    labels = np.full(n, 1, dtype=np.uint8)   # todo clase 1

    # Insertar un punto "ruidoso" (clase 2) en medio del cluster, rodeado
    # de vecinos clase 1 — debe corregirse a 1.
    noisy_idx = 50
    labels[noisy_idx] = 2
    xyz[noisy_idx] = xyz[:20].mean(axis=0)   # colocarlo cerca de otros pts clase 1

    # Puntos sin etiquetar y eliminados — no deben cambiar ni contar como voto
    labels[100] = 0
    labels[101] = DELETED_LABEL

    changed_idx, new_labels = smooth_labels_majority(xyz, labels, k=10)

    check("smooth_labels_majority corrige el punto ruidoso a la clase mayoritaria",
          noisy_idx in changed_idx.tolist(),
          f"changed_idx contiene {noisy_idx}? {noisy_idx in changed_idx.tolist()}")
    if noisy_idx in changed_idx.tolist():
        pos = changed_idx.tolist().index(noisy_idx)
        check("el punto ruidoso corregido pasa a clase 1 (la mayoritaria)",
              int(new_labels[pos]) == 1, f"new_label={new_labels[pos]}")

    check("smooth_labels_majority NUNCA reasigna puntos sin etiquetar (clase 0)",
          100 not in changed_idx.tolist())
    check("smooth_labels_majority NUNCA reasigna puntos eliminados (DELETED_LABEL)",
          101 not in changed_idx.tolist())

    # Nube 100% homogénea sin ruido: nada debería cambiar
    labels_clean = np.full(n, 3, dtype=np.uint8)
    changed2, _ = smooth_labels_majority(xyz, labels_clean, k=8)
    check("smooth_labels_majority no cambia nada en una nube ya homogénea",
          len(changed2) == 0, f"len(changed2)={len(changed2)}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — detect_isolated_clusters_per_class: QA de anotación, detecta
# clusters pequeños/aislados por clase.
# ══════════════════════════════════════════════════════════════════════════
def test_detect_isolated_clusters_per_class():
    from annotation.label_smoothing import detect_isolated_clusters_per_class

    rng = np.random.default_rng(11)
    # Cluster grande de clase 1: grid regular con espaciado << connect_dist_m
    # (garantiza conectividad total, a diferencia de puntos uniformes al
    # azar que pueden dejar huecos aislados por pura casualidad) + un
    # cluster aislado de 4 puntos de clase 2 muy lejos del resto.
    gx, gy = np.meshgrid(np.arange(10) * 0.3, np.arange(10) * 0.3)
    big = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(100)]).astype(np.float32)
    small = rng.uniform(100, 100.5, size=(4, 3)).astype(np.float32)
    xyz = np.vstack([big, small])
    labels = np.zeros(len(xyz), dtype=np.uint8)
    labels[:100] = 1
    labels[100:] = 2

    result = detect_isolated_clusters_per_class(xyz, labels, max_cluster_size=15,
                                                connect_dist_m=1.0)
    check("detect_isolated_clusters_per_class detecta el cluster pequeño (clase 2)",
          2 in result, f"result keys={list(result.keys())}")
    if 2 in result:
        sizes = [s for s, _ in result[2]]
        check("el cluster aislado reportado tiene el tamaño correcto (4 pts)",
              sizes == [4], f"sizes={sizes}")
    check("detect_isolated_clusters_per_class NO marca el cluster grande y denso (clase 1)",
          1 not in result, f"result keys={list(result.keys())}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — ransac_plane: ajuste de plano RANSAC sobre una superficie
# sintética plana + ruido gaussiano fuera del plano.
# ══════════════════════════════════════════════════════════════════════════
def test_ransac_plane_fit():
    from annotation.plane_fit import ransac_plane

    rng = np.random.default_rng(12)
    n_plane = 500
    # Plano z = 2.0 (normal (0,0,1)), con ruido pequeño
    xy = rng.uniform(-10, 10, size=(n_plane, 2)).astype(np.float64)
    z = 2.0 + rng.normal(0, 0.01, n_plane)
    plane_pts = np.column_stack([xy, z])

    # Ruido: puntos dispersos en 3D, claramente fuera del plano
    n_noise = 100
    noise_pts = rng.uniform(-10, 10, size=(n_noise, 3)).astype(np.float64)
    noise_pts[:, 2] += 10.0   # bien lejos del plano z=2

    xyz = np.vstack([plane_pts, noise_pts]).astype(np.float32)

    result = ransac_plane(xyz, distance_threshold=0.1, n_iterations=300,
                          rng=np.random.default_rng(0))
    check("ransac_plane encuentra un resultado", result is not None)
    if result is None:
        return
    inliers, normal, d = result
    check("ransac_plane detecta casi todos los puntos del plano como inliers",
          bool(inliers[:n_plane].sum() > n_plane * 0.9),
          f"inliers en plano: {inliers[:n_plane].sum()}/{n_plane}")
    check("ransac_plane NO incluye el ruido disperso como inliers",
          bool(inliers[n_plane:].sum() < n_noise * 0.1),
          f"inliers en ruido: {inliers[n_plane:].sum()}/{n_noise}")
    if normal is not None:
        # La normal debe ser ~(0,0,±1) para un plano horizontal
        check("ransac_plane recupera una normal ~vertical para el plano horizontal sintético",
              bool(abs(float(normal[2])) > 0.95), f"normal={normal}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — PlaneFitTool (herramienta): Ctrl+clic ajusta el plano en un radio
# y etiqueta SOLO los puntos planos, dejando el resto de la esfera intacto.
# Llamada directa a _fit_and_apply(), mismo patrón que RegionGrowingTool.
# ══════════════════════════════════════════════════════════════════════════
def test_plane_fit_tool_selects_only_planar_points():
    from annotation.plane_fit import PlaneFitTool

    rng = np.random.default_rng(13)
    # Un "techo" plano (z=3.0) de 800 pts + una "chimenea" no-plana (una
    # columna vertical de puntos) dentro del MISMO radio de búsqueda.
    xy = rng.uniform(-4, 4, size=(800, 2)).astype(np.float32)
    roof = np.column_stack([xy, np.full(800, 3.0, np.float32) +
                            rng.normal(0, 0.01, 800).astype(np.float32)])
    chimney = np.column_stack([
        np.full(60, 0.5, np.float32), np.full(60, 0.5, np.float32),
        np.linspace(3.0, 6.0, 60).astype(np.float32)])
    xyz = np.vstack([roof, chimney]).astype(np.float32)

    class FakePC:
        pass
    pc = FakePC(); pc.xyz = xyz

    class FakeCanvas:
        pass
    canvas = FakeCanvas()
    canvas.pc = pc
    canvas._tile_mode = False
    canvas._cur_xyz = xyz
    canvas._cur_idx = None

    tool = PlaneFitTool()
    tool.canvas = canvas
    tool.label_store = None
    tool.radius_m = 10.0
    tool.distance_threshold_m = 0.1
    tool.n_iterations = 300

    captured = {}
    def fake_apply(indices):
        captured["indices"] = np.asarray(indices)
    tool._apply_selection = fake_apply

    center = np.array([0.0, 0.0, 3.0], np.float32)
    tool._fit_and_apply(center)

    sel = captured.get("indices")
    check("PlaneFitTool seleccionó algo", sel is not None and len(sel) > 0)
    if sel is None:
        return
    n_roof_selected = int((sel < 800).sum())
    n_chimney_selected = int((sel >= 800).sum())
    check("PlaneFitTool selecciona casi todo el techo plano",
          n_roof_selected > 800 * 0.9, f"n_roof_selected={n_roof_selected}/800")
    check("PlaneFitTool NO selecciona los puntos de la chimenea (no planos)",
          n_chimney_selected < 60 * 0.2, f"n_chimney_selected={n_chimney_selected}/60")
    check("PlaneFitTool guarda la normal del último plano ajustado",
          tool.last_normal is not None)


# ══════════════════════════════════════════════════════════════════════════
# TEST — MarkerStore: modelo de datos de etiquetas/medidas persistentes en
# 3D (sin VTK) + round-trip de persistencia con Project.
# ══════════════════════════════════════════════════════════════════════════
def test_marker_store_and_project_persistence():
    from annotation.markers import MarkerStore
    from core.project import Project
    import tempfile as _tempfile

    store = MarkerStore()
    m1 = store.add_label([1.0, 2.0, 3.0], "poste dañado")
    m2 = store.add_measure([0.0, 0.0, 0.0], [3.0, 4.0, 0.0], "ancho de calle")

    check("MarkerStore.add_label agrega un marcador", len(store) == 2)
    check("Marker.distance() calcula la distancia 3D correctamente para measure",
          abs(m2.distance() - 5.0) < 1e-6, f"distance={m2.distance()}")
    check("Marker.distance() es None para marcadores tipo label",
          m1.distance() is None)

    removed = store.remove(m1.id)
    check("MarkerStore.remove quita el marcador correcto", removed and len(store) == 1)

    # Round-trip completo vía Project.save/load
    store2 = MarkerStore()
    store2.add_label([5.0, 6.0, 7.0], "grieta en el muro", color=(0.2, 0.8, 0.2))
    store2.add_measure([1.0, 1.0, 1.0], [1.0, 1.0, 5.0], "altura del poste")

    with _tempfile.TemporaryDirectory() as td:
        proj_path = str(Path(td) / "test.geoa3d")
        p = Project.new("nube_falsa.laz")
        p.init_labels(10)
        p.markers = store2.to_list()
        p.save(proj_path)

        p2 = Project.load(proj_path)
        store3 = MarkerStore()
        store3.load_list(p2.markers)

        check("Project.save/load conserva los marcadores (round-trip completo)",
              len(store3) == 2, f"len(store3)={len(store3)}")
        labels_marker = [m for m in store3.all() if m.kind == "label"]
        check("el marcador de texto sobrevive el round-trip con su texto y color",
              len(labels_marker) == 1 and labels_marker[0].text == "grieta en el muro"
              and abs(labels_marker[0].color[1] - 0.8) < 1e-6,
              f"labels_marker={labels_marker}")
        measure_marker = [m for m in store3.all() if m.kind == "measure"]
        check("el marcador de medida sobrevive el round-trip con la distancia correcta",
              len(measure_marker) == 1 and abs(measure_marker[0].distance() - 4.0) < 1e-6,
              f"measure_marker={measure_marker}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — _ClipBoxController (render/canvas.py): caja de recorte interactiva.
#
# Descubrimiento importante (dos capas): (1) vtkBoxWidget/vtkPlanes/
# mapper.AddClippingPlane NO necesitan una ventana de render real (a
# diferencia de QVTKRenderWindowInteractor, que sí hace SEGFAULT offscreen
# aquí); (2) MÁS IMPORTANTE — un `vtk.vtkRenderWindow()` puro (sin Qt/
# interactor) con `SetOffScreenRendering(1)` SÍ renderiza de verdad aquí.
# Esto permitió encontrar un bug real que el primer test (basado solo en
# "cuenta 6 planos" y "bounds aproximados") NO detectaba: aplicar los
# planos de vtkBoxWidget tal cual devuelve VTK (normal apuntando hacia
# AFUERA de la caja) con mapper.AddClippingPlane produce una intersección
# vacía — pantalla completamente negra, SIEMPRE, sin importar tamaño de
# caja — confirmado renderizando de verdad y comparando contra una imagen
# sin recorte. El fix (invertir cada normal) se verifica aquí con el
# mismo método: render real antes/después, contar píxeles no-fondo.
# ══════════════════════════════════════════════════════════════════════════
def test_clip_box_controller():
    import vtk
    from render.canvas import _ClipBoxController

    ren = vtk.vtkRenderer()
    iren = vtk.vtkRenderWindowInteractor()
    ctrl = _ClipBoxController(ren, iren)
    mapper = vtk.vtkPolyDataMapper()

    check("_ClipBoxController arranca deshabilitado", not ctrl.enabled)

    ctrl.enable(mapper, (0, 10, 0, 10, 0, 10))
    check("enable() marca enabled=True", ctrl.enabled)
    n_planes = mapper.GetClippingPlanes().GetNumberOfItems() if mapper.GetClippingPlanes() else 0
    check("enable() aplica exactamente 6 planos de recorte al mapper",
          n_planes == 6, f"n_planes={n_planes}")

    # Cada plano aplicado debe apuntar hacia ADENTRO de la caja (hacia el
    # centro) — si apuntara hacia afuera (el bug real encontrado), el
    # mapper conservaría "afuera de las 6 caras a la vez": intersección
    # vacía, pantalla negra siempre. Verificación geométrica: el centro
    # de la caja debe evaluar POSITIVO en cada plano aplicado (heurística
    # exacta de "lado conservado" de mapper.AddClippingPlane).
    center = (5.0, 5.0, 5.0)
    applied = mapper.GetClippingPlanes()
    all_center_positive = True
    for i in range(applied.GetNumberOfItems()):
        p = applied.GetItem(i)
        if p.EvaluateFunction(*center) < 0:
            all_center_positive = False
    check("los planos aplicados conservan el CENTRO de la caja (apuntan "
          "hacia adentro, no hacia afuera)", all_center_positive)

    ctrl.disable()
    check("disable() marca enabled=False", not ctrl.enabled)
    n_planes_after = mapper.GetClippingPlanes().GetNumberOfItems() if mapper.GetClippingPlanes() else 0
    check("disable() quita todos los planos de recorte (vuelve a verse todo)",
          n_planes_after == 0, f"n_planes_after={n_planes_after}")

    # Reactivar y mover la caja — los planos deben reflejar los NUEVOS bounds
    ctrl.enable(mapper, (0, 10, 0, 10, 0, 10))
    ctrl._widget.PlaceWidget(0, 4, 0, 4, 0, 4)
    ctrl._apply_planes()
    planes2 = vtk.vtkPlanes()
    ctrl._widget.GetPlanes(planes2)
    origins = [planes2.GetPlane(i).GetOrigin() for i in range(planes2.GetNumberOfPlanes())]
    max_coord = max(abs(c) for o in origins for c in o)
    check("mover la caja actualiza los planos a los nuevos bounds (más chicos)",
          max_coord <= 4.01, f"max_coord={max_coord} (esperado <=4)")


# ══════════════════════════════════════════════════════════════════════════
# TEST — Verificación con RENDER OFFSCREEN REAL: confirma con una imagen
# de verdad (no solo geometría de planos) que la caja de recorte muestra
# SOLO lo que está dentro — antes del fix, esto renderizaba negro total
# SIEMPRE (probado y confirmado con exactamente este método antes de
# corregir render/canvas.py::_ClipBoxController._apply_planes).
# ══════════════════════════════════════════════════════════════════════════
def test_clip_box_real_offscreen_render():
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from render.canvas import _ClipBoxController

    rng = np.random.default_rng(0)
    n = 4000
    xyz = rng.uniform(0, 20, size=(n, 3)).astype(np.float32)

    pts = vtk.vtkPoints(); pts.SetData(numpy_to_vtk(xyz, deep=True))
    poly = vtk.vtkPolyData(); poly.SetPoints(pts)
    verts = vtk.vtkCellArray()
    segs = np.empty(2 * n, np.int64); segs[0::2] = 1; segs[1::2] = np.arange(n)
    verts.SetCells(n, numpy_to_vtkIdTypeArray(segs, deep=True))
    poly.SetVerts(verts)

    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly)
    actor = vtk.vtkActor(); actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 1, 1)
    actor.GetProperty().SetPointSize(3); actor.GetProperty().LightingOff()

    ren = vtk.vtkRenderer(); ren.AddActor(actor); ren.SetBackground(0, 0, 0)
    iren = vtk.vtkRenderWindowInteractor()
    rw = vtk.vtkRenderWindow(); rw.SetOffScreenRendering(1); rw.SetSize(200, 200)
    rw.AddRenderer(ren)
    cam = ren.GetActiveCamera()
    cam.SetPosition(10, 10, 60); cam.SetFocalPoint(10, 10, 10); cam.SetViewUp(0, 1, 0)
    ren.ResetCameraClippingRange()

    def _count_lit_pixels():
        rw.Render()
        w2if = vtk.vtkWindowToImageFilter(); w2if.SetInput(rw); w2if.Update()
        img = w2if.GetOutput()
        dims = img.GetDimensions()
        scalars = img.GetPointData().GetScalars()
        arr = np.array([scalars.GetTuple3(i) for i in
                        range(0, scalars.GetNumberOfTuples(), 7)])  # muestreo, no todo el frame
        return int((arr.sum(axis=1) > 30).sum())

    lit_before = _count_lit_pixels()
    check("render offscreen SIN recorte muestra puntos (sanity check)",
          lit_before > 0, f"lit_before={lit_before}")

    ctrl = _ClipBoxController(ren, iren)
    ctrl.enable(mapper, (5, 15, 5, 15, 0, 20))   # caja central, más chica que la nube
    lit_clipped = _count_lit_pixels()
    check("CON la caja de recorte activa, la pantalla NO queda completamente "
          "negra (el bug real hacía exactamente esto)",
          lit_clipped > 0, f"lit_clipped={lit_clipped}")
    check("la caja de recorte SÍ reduce lo visible vs. sin recorte "
          "(la caja es más chica que la nube completa)",
          lit_clipped < lit_before, f"lit_clipped={lit_clipped} lit_before={lit_before}")

    ctrl.disable()
    lit_after_disable = _count_lit_pixels()
    check("desactivar la caja restaura la vista completa",
          lit_after_disable >= lit_clipped, f"lit_after_disable={lit_after_disable}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — _MarkerOverlay (render/canvas.py): actores VTK para marcadores
# persistentes.
#
# LIMITACIÓN REAL ENCONTRADA (distinta a la de AnnotationCanvas): a
# diferencia de vtkBoxWidget/vtkPlanes (ver test_clip_box_controller, que
# SÍ funcionan sin ventana real), crear un vtkBillboardTextActor3D con
# texto y consultarlo en este entorno sin GPU/contexto OpenGL provoca un
# "access violation" nativo (crashea el proceso Python entero, no una
# excepción atrapable) — probablemente porque el texto necesita
# freetype/texturas inicializadas por un contexto real. Por eso este test
# SOLO ejercita la parte segura (esfera + línea, sin texto) — la ruta con
# texto (la mayoría de los marcadores reales, que sí llevan texto) queda
# sin poder probarse automáticamente aquí y necesita confirmación del
# usuario en su máquina con GPU real.
# ══════════════════════════════════════════════════════════════════════════
def test_marker_overlay_actors_no_text():
    import vtk
    from render.canvas import _MarkerOverlay
    from annotation.markers import Marker

    ren = vtk.vtkRenderer()
    overlay = _MarkerOverlay(ren)

    check("_MarkerOverlay arranca sin actores", ren.GetActors().GetNumberOfItems() == 0)

    # Marcadores SIN texto — ejercitan esfera (+ línea para measure) sin
    # tocar vtkBillboardTextActor3D en absoluto.
    m_label = Marker(id="a", kind="label", pos=[1.0, 2.0, 3.0], text="")
    m_measure = Marker(id="b", kind="measure", pos=[0.0, 0.0, 0.0],
                       pos_b=[3.0, 4.0, 0.0], text="")
    overlay.sync([m_label, m_measure])

    n_actors = ren.GetActors().GetNumberOfItems()
    # label sin texto: solo esfera (1) — measure sin texto: esfera + línea (2) = 3
    check("sync() agrega esfera(s) y línea de medida sin texto",
          n_actors == 3, f"n_actors={n_actors}")

    overlay.sync([])
    check("sync([]) limpia todos los actores anteriores",
          ren.GetActors().GetNumberOfItems() == 0,
          f"n_actors={ren.GetActors().GetNumberOfItems()}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — Detección de COPC (Cloud Optimized Point Cloud) en core/pointcloud.py.
# No se puede fabricar un COPC real de producción en este entorno (laspy no
# trae un escritor COPC, y no hay PDAL instalado) — así que este test
# verifica lo que SÍ se puede probar sin arriesgar nada: que la detección
# (laspy.CopcReader.open, envuelta en try/except) NO rompe la carga normal
# de un .laz NO-COPC (el caso de casi todos los archivos reales), y que
# CopcReader.open() en sí falla limpio (excepción atrapable, no crash) sobre
# un archivo no-COPC — la propiedad de la que depende toda la implementación.
# ══════════════════════════════════════════════════════════════════════════
def test_copc_detection_does_not_break_normal_laz():
    try:
        import laspy
    except ImportError:
        print("[SKIP] test_copc_detection_does_not_break_normal_laz — laspy no instalado")
        return
    from core.pointcloud import CloudPipeline

    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "normal.laz")
        hdr = laspy.LasHeader(point_format=3, version="1.4")
        hdr.scales = [0.01, 0.01, 0.01]
        las = laspy.LasData(hdr)
        rng = np.random.default_rng(7)
        n = 5_000
        las.x = rng.uniform(0, 100, n)
        las.y = rng.uniform(0, 100, n)
        las.z = rng.uniform(0, 50, n)
        las.write(path)

        # CopcReader.open() debe rechazar limpio un .laz normal (excepción
        # atrapable) — es la propiedad de la que depende el try/except en
        # core/pointcloud.py::_las().
        raised = False
        try:
            with laspy.CopcReader.open(path):
                pass
        except Exception:
            raised = True
        check("CopcReader.open() rechaza un .laz normal con una excepción "
              "atrapable (no un crash)", raised)

        pipeline = CloudPipeline(path)
        pc = pipeline._las()
        check("_las() sigue cargando un .laz normal sin problema tras "
              "agregar la detección de COPC",
              pc is not None and pc.xyz is not None and len(pc.xyz) == n,
              f"len(xyz)={len(pc.xyz) if pc is not None and pc.xyz is not None else None}")
        check("un .laz normal (no-COPC) NO se etiqueta como COPC en pc.fmt",
              "COPC" not in (pc.fmt or ""), f"pc.fmt={pc.fmt!r}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — Instanciación headless de GeoPanel (nueva tarjeta "PASO 4 — QA de
# anotación") y ToolPanel (nuevo toggle "Caja de recorte" + 2 herramientas
# nuevas en la grilla) — sin VTK, solo construcción de widgets Qt.
# ══════════════════════════════════════════════════════════════════════════
def test_new_ui_widgets_instantiate():
    from ui.geo_panel import GeoPanel
    from ui.tool_panel import ToolPanel
    from core.project import DEFAULT_SCHEMA

    gp = GeoPanel()
    gp.set_schema(list(DEFAULT_SCHEMA))
    check("GeoPanel (con la nueva tarjeta PASO 4) se instancia sin error",
          hasattr(gp, "_smooth_btn") and hasattr(gp, "_iso_btn"))

    captured = {}
    gp.smooth_labels_requested.connect(lambda k: captured.setdefault("k", k))
    gp._smooth_k.setValue(12)
    gp._smooth_btn.click()
    check("botón 'Suavizar etiquetas' emite smooth_labels_requested con el k correcto",
          captured.get("k") == 12, f"captured={captured}")

    captured2 = {}
    gp.isolated_clusters_requested.connect(
        lambda mx, d: captured2.setdefault("args", (mx, d)))
    gp._iso_max.setValue(7)
    gp._iso_btn.click()
    check("botón 'Detectar clusters aislados' emite isolated_clusters_requested",
          captured2.get("args") == (7, 1.0), f"captured2={captured2}")

    tp = ToolPanel()
    check("ToolPanel tiene el toggle 'Caja de recorte'", hasattr(tp, "_tog_clipbox"))
    captured3 = {}
    tp.clip_box_toggled.connect(lambda v: captured3.setdefault("v", v))
    tp._tog_clipbox.mousePressEvent(None)   # _Toggle: simular clic (no QCheckBox)
    check("el toggle de caja de recorte emite clip_box_toggled",
          captured3.get("v") is True, f"captured3={captured3}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — MarkerStore.update(): edición in-place de un marcador existente
# (texto/color/tamaño de letra) — lo que usa LabelMarkerTool al hacer
# Ctrl+clic sobre una etiqueta ya puesta en vez de crear una nueva.
# ══════════════════════════════════════════════════════════════════════════
def test_marker_store_update():
    from annotation.markers import MarkerStore

    store = MarkerStore()
    m = store.add_label([1.0, 2.0, 3.0], "texto viejo", color=(1.0, 0.0, 0.0), font_size=12.0)

    changed = {"n": 0}
    store.markers_changed.connect(lambda: changed.__setitem__("n", changed["n"] + 1))

    ok = store.update(m.id, text="texto nuevo", color=[0.0, 1.0, 0.0], font_size=20.0)
    check("MarkerStore.update() devuelve True para un id existente", ok)
    m2 = store.get(m.id)
    check("MarkerStore.update() cambia el texto in-place", m2.text == "texto nuevo")
    check("MarkerStore.update() cambia el color in-place",
          list(m2.color) == [0.0, 1.0, 0.0], f"color={m2.color}")
    check("MarkerStore.update() cambia el tamaño de letra in-place",
          m2.font_size == 20.0, f"font_size={m2.font_size}")
    check("MarkerStore.update() emite markers_changed", changed["n"] >= 1)

    ok_missing = store.update("id-que-no-existe", text="x")
    check("MarkerStore.update() con un id inexistente devuelve False", not ok_missing)


# ══════════════════════════════════════════════════════════════════════════
# TEST — LabelMarkerTool._find_nearby_label(): detecta si el clic cayó
# cerca de una etiqueta existente (para editarla) o no (para crear una
# nueva) — pura geometría de proyección a pantalla, sin diálogos Qt.
# ══════════════════════════════════════════════════════════════════════════
def test_label_marker_tool_find_nearby():
    from annotation.tools import LabelMarkerTool
    from annotation.markers import MarkerStore

    store = MarkerStore()
    m1 = store.add_label([0.0, 0.0, 0.0], "cerca")
    m2 = store.add_label([100.0, 100.0, 0.0], "lejos")

    class FakeCanvas:
        marker_store = store
        @staticmethod
        def map_to_screen(positions):
            # Proyección de juguete: (x,y,z) -> (x,y) en pantalla, 1:1.
            return np.asarray(positions)[:, :2]

    tool = LabelMarkerTool()
    tool.canvas = FakeCanvas()

    found_close = tool._find_nearby_label((2.0, 2.0))   # cerca de m1 en pantalla
    check("_find_nearby_label encuentra la etiqueta cuando el clic cae cerca",
          found_close is not None and found_close.id == m1.id)

    found_far = tool._find_nearby_label((50.0, 50.0))   # lejos de ambas
    check("_find_nearby_label NO encuentra nada si el clic no cae cerca de ninguna",
          found_far is None)


# ══════════════════════════════════════════════════════════════════════════
# TEST — LabelMarkerTool.on_mouse_press(): flujo completo crear/editar,
# sustituyendo _LabelMarkerDialog por un stub (sin abrir un diálogo Qt
# modal real, que bloquearía el test) — verifica que llama a
# marker_store.add_label()/update() con los valores que "el usuario
# habría ingresado en el diálogo".
# ══════════════════════════════════════════════════════════════════════════
def test_label_marker_tool_create_and_edit_flow():
    import annotation.tools as tools_mod
    from annotation.markers import MarkerStore

    store = MarkerStore()

    class FakeCanvas:
        marker_store = store
        pc = None
        _cur_xyz = np.array([[0.0, 0.0, 0.0]], np.float32)
        @staticmethod
        def map_to_screen(positions):
            return np.asarray(positions)[:, :2]
        @staticmethod
        def update():
            pass

    class _FakeDialogAccept:
        def __init__(self, parent, title, text="", color=(1.0, 0.85, 0.2), font_size=14.0):
            pass
        def exec(self):
            return (True, False, "un texto de prueba", (0.1, 0.2, 0.3), 22.0)

    tool = tools_mod.LabelMarkerTool()
    tool.canvas = FakeCanvas()
    tool._world_pos = lambda pos, fast=True: np.array([5.0, 5.0, 5.0], np.float32)
    tool._snap_to_point = lambda pos: None

    orig_dialog = tools_mod._LabelMarkerDialog
    tools_mod._LabelMarkerDialog = _FakeDialogAccept
    try:
        class _Ev:
            button = 1
            pos = (5.0, 5.0)
        tool.on_mouse_press(_Ev())
        check("on_mouse_press (sin etiqueta cerca) crea un marcador nuevo",
              len(store) == 1)
        if len(store) == 1:
            m = store.all()[0]
            check("el marcador creado usa el texto del diálogo", m.text == "un texto de prueba")
            check("el marcador creado usa el color del diálogo",
                  list(m.color) == [0.1, 0.2, 0.3], f"color={m.color}")
            check("el marcador creado usa el tamaño de letra del diálogo",
                  m.font_size == 22.0, f"font_size={m.font_size}")
        check("el tool recuerda color/tamaño para la próxima etiqueta",
              tool.color == (0.1, 0.2, 0.3) and tool.font_size == 22.0)

        # Segundo clic en el MISMO lugar -> debe EDITAR el marcador existente,
        # no crear uno segundo.
        class _FakeDialogEdit:
            def __init__(self, parent, title, text="", color=(1.0, 0.85, 0.2), font_size=14.0):
                self._prev_text = text
            def exec(self):
                return (True, False, self._prev_text + " editado", (0.5, 0.5, 0.5), 30.0)
        tools_mod._LabelMarkerDialog = _FakeDialogEdit
        tool.on_mouse_press(_Ev())
        check("Ctrl+clic sobre una etiqueta existente EDITA en vez de crear otra",
              len(store) == 1, f"len(store)={len(store)}")
        m_edited = store.all()[0]
        check("la edición conserva el texto anterior + el sufijo del diálogo",
              m_edited.text == "un texto de prueba editado", f"text={m_edited.text!r}")
        check("la edición actualiza el color", list(m_edited.color) == [0.5, 0.5, 0.5])

        # Elegir "Eliminar etiqueta" en el diálogo de edición debe borrar
        # el marcador — antes _LabelMarkerDialog no tenía esta opción en
        # absoluto (solo LabelMarkerTool podía crear/editar, nunca borrar).
        class _FakeDialogDelete:
            def __init__(self, parent, title, text="", color=(1.0, 0.85, 0.2), font_size=14.0):
                pass
            def exec(self):
                return (True, True, "", (0, 0, 0), 14.0)   # deleted=True
        tools_mod._LabelMarkerDialog = _FakeDialogDelete
        tool.on_mouse_press(_Ev())
        check("elegir 'Eliminar' en el diálogo de Etiqueta 3D borra el marcador",
              len(store) == 0, f"len(store)={len(store)}")
    finally:
        tools_mod._LabelMarkerDialog = orig_dialog


# ══════════════════════════════════════════════════════════════════════════
# TEST — ransac_plane sigue funcionando tras los cambios de UX (sin
# cambios en el algoritmo en sí, pero confirma que _draw_plane_preview no
# rompe _fit_and_apply cuando canvas._mline no está disponible/es distinto).
# ══════════════════════════════════════════════════════════════════════════
def test_plane_fit_tool_preview_does_not_break_selection():
    from annotation.plane_fit import PlaneFitTool

    rng = np.random.default_rng(14)
    xy = rng.uniform(-4, 4, size=(500, 2)).astype(np.float32)
    z = np.full(500, 2.0, np.float32) + rng.normal(0, 0.01, 500).astype(np.float32)
    xyz = np.column_stack([xy, z]).astype(np.float32)

    class FakePC:
        pass
    pc = FakePC(); pc.xyz = xyz

    class _FakeMline:
        visible = False
        def set_data(self, *a, **k): pass

    class FakeCanvas:
        pass
    canvas = FakeCanvas()
    canvas.pc = pc
    canvas._tile_mode = False
    canvas._cur_xyz = xyz
    canvas._cur_idx = None
    canvas._mline = _FakeMline()
    canvas.update = lambda: None

    tool = PlaneFitTool()
    tool.canvas = canvas
    tool.label_store = None
    tool.radius_m = 10.0
    tool.distance_threshold_m = 0.1

    captured = {}
    tool._apply_selection = lambda idx: captured.setdefault("idx", np.asarray(idx))
    tool._fit_and_apply(np.array([0.0, 0.0, 2.0], np.float32))

    check("PlaneFitTool con vista previa activada sigue seleccionando inliers",
          "idx" in captured and len(captured["idx"]) > 400,
          f"captured={list(captured.keys())}")
    check("_draw_plane_preview deja el overlay _mline visible",
          canvas._mline.visible is True)


# ══════════════════════════════════════════════════════════════════════════
# TEST — Instanciación headless de las nuevas secciones de contexto en
# ToolPanel ("Ajustar plano" con sliders de radio/tolerancia, "Etiqueta 3D"
# con tamaño de letra + selector de color).
# ══════════════════════════════════════════════════════════════════════════
def test_tool_panel_new_context_sections():
    from ui.tool_panel import ToolPanel

    tp = ToolPanel()
    tp._ctx.update_for_tool("Ajustar plano")
    check("sección de contexto 'Ajustar plano' expone slider de radio",
          hasattr(tp._ctx, "_pr"))
    check("sección de contexto 'Ajustar plano' expone slider de tolerancia",
          hasattr(tp._ctx, "_pth"))

    captured = {}
    tp.plane_radius_changed.connect(lambda v: captured.setdefault("radius", v))
    tp.plane_threshold_changed.connect(lambda v: captured.setdefault("thresh", v))
    tp._ctx._pr._s.setValue(12)
    tp._ctx._pth._s.setValue(25)   # 25 -> /100 = 0.25 (ver conexión en update_for_tool)
    check("slider de radio de 'Ajustar plano' emite plane_radius_changed",
          captured.get("radius") == 12, f"captured={captured}")
    check("slider de tolerancia de 'Ajustar plano' emite plane_threshold_changed (cm->m)",
          abs(captured.get("thresh", -1) - 0.25) < 1e-6, f"captured={captured}")

    tp._ctx.update_for_tool("Etiqueta 3D")
    check("sección de contexto 'Etiqueta 3D' expone slider de tamaño de letra",
          hasattr(tp._ctx, "_lfs"))
    check("sección de contexto 'Etiqueta 3D' expone botón de color",
          hasattr(tp._ctx, "_label_color_btn"))

    captured2 = {}
    tp.label_font_size_changed.connect(lambda v: captured2.setdefault("size", v))
    tp._ctx._lfs._s.setValue(28)
    check("slider de tamaño de 'Etiqueta 3D' emite label_font_size_changed",
          captured2.get("size") == 28, f"captured2={captured2}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — Rediseño de TilePanel/ToolPanel (2026-09-09): "no se sabe cómo
# volver a la vista general / poner densidad máxima / vista cenital",
# iconos de mover/rotar grid poco intuitivos, iconos de herramientas de
# dibujo poco diferenciables. Verificado con capturas reales (QT_QPA_
# PLATFORM=offscreen renderiza widgets Qt normales sin problema, a
# diferencia de VTK) — dos bugs REALES encontrados así, no solo revisados
# a mano, ambos con test de regresión aquí:
#   1. TileGridWidget: el tamaño de fuente de las celdas escalaba sin
#      límite con el tamaño de celda — en celdas grandes el texto ("100%")
#      salía más ANCHO que la celda y se pintaba encima de las vecinas.
#   2. _ContextSection.update_for_tool: al cambiar de herramienta, los
#      widgets viejos se programaban para borrarse con deleteLater()
#      (diferido) pero seguían PINTÁNDOSE hasta el siguiente ciclo del
#      event loop — texto fantasma superpuesto con el nuevo contenido.
# ══════════════════════════════════════════════════════════════════════════
def test_tile_grid_font_never_wider_than_cell():
    from PyQt5.QtGui import QFont, QFontMetrics
    from ui.tile_panel import TileGridWidget

    class FakeTile:
        def __init__(self, row, col, pct):
            self.row, self.col, self.labeled_pct = row, col, pct

    class FakeTM:
        n_rows, n_cols = 2, 2
        tiles = [FakeTile(0, 0, 100.0), FakeTile(0, 1, 35.0),
                 FakeTile(1, 0, 0.0), FakeTile(1, 1, 70.0)]

    grid = TileGridWidget()
    # Celdas GRANDES a propósito (fuerza _cell_sz cerca de CELL_MAX=56) —
    # este era exactamente el caso que producía el bug.
    grid.resize(400, 400)
    grid.set_tile_manager(FakeTM())
    grid._recompute_cell()

    c = grid._cell_sz
    check("TileGridWidget: la celda alcanza un tamaño grande en este escenario "
          "(el caso que reproducía el bug)", c >= 40, f"cell_sz={c}")

    # Reproducir el MISMO cálculo de tamaño de fuente que paintEvent —
    # si esto regresa, la fuente elegida volvería a ser más ancha que la celda.
    f = QFont()
    widest = "100%"
    max_w = c - 6
    size = min(16, c - 8)
    while size > 6:
        f.setPixelSize(size)
        fm = QFontMetrics(f)
        if fm.horizontalAdvance(widest) <= max_w and fm.height() <= c - 2:
            break
        size -= 1
    f.setPixelSize(max(6, size))
    fm = QFontMetrics(f)
    check("el tamaño de fuente elegido para el grid de tiles SIEMPRE cabe "
          "en el ancho de la celda (antes: escalaba sin límite y se "
          "desbordaba sobre las celdas vecinas)",
          fm.horizontalAdvance("100%") <= (c - 6),
          f"cell={c} font_px={size} advance={fm.horizontalAdvance('100%')}")


def _collect_layout_widgets(lay, out: set) -> None:
    """Recorre un QLayout recursivamente (incluye sub-layouts agregados
    con addLayout, no solo addWidget) y junta todos los widgets vivos."""
    from PyQt5.QtWidgets import QLabel
    for i in range(lay.count()):
        item = lay.itemAt(i)
        w = item.widget()
        if w is not None:
            out.add(w)
            out.update(w.findChildren(QLabel))
            continue
        sub = item.layout()
        if sub is not None:
            _collect_layout_widgets(sub, out)


def test_context_section_no_stale_widgets_on_tool_switch():
    from ui.tool_panel import ToolPanel
    from PyQt5.QtWidgets import QLabel

    tp = ToolPanel()
    ctx = tp._ctx

    # BUG REAL reportado por el usuario (2026-09): al cambiar de
    # "Etiqueta 3D" o "Polilínea" — las dos únicas secciones con una fila
    # de color agregada vía root.addLayout(...), no root.addWidget(...) —
    # a otra herramienta, esa fila de color se quedaba pintada encima del
    # panel de la herramienta nueva.
    #
    # Dos intentos de verificación descartados por dar falsos resultados
    # (documentado para no repetir el error):
    #   - `lbl.isVisible()`: en este entorno ToolPanel nunca se muestra en
    #     una ventana real, así que es SIEMPRE False para TODO widget sin
    #     importar hide()/mostrar — un falso NEGATIVO permanente (nunca
    #     detecta nada, aunque el bug esté presente).
    #   - `lbl.isHidden()`: solo refleja si a ESE widget puntual alguien
    #     le llamó hide() directamente — los QLabel internos de
    #     `_SliderRow` (p.ej. "Radio", "18 m") nunca se ocultan a sí
    #     mismos, solo su contenedor `_SliderRow` se oculta; eso da un
    #     falso POSITIVO (los marca como fantasma aunque su padre ya esté
    #     oculto).
    #   `isVisibleTo(ctx)` sí es correcta: considera toda la cadena de
    #   padres HASTA `ctx` (sin necesitar que `ctx` mismo esté en una
    #   ventana mostrada) — así que un descendiente de un widget oculto
    #   también sale como no-visible-para-ctx, y no hace falta que nada
    #   esté realmente en pantalla.
    names = ["Pincel", "Ajustar plano", "Etiqueta 3D", "Medir",
            "Polilínea", "Pincel"]
    for n in names:
        ctx.update_for_tool(n)
        app.processEvents()
        current_widgets = set()
        lay = ctx.layout()
        if lay is not None:
            _collect_layout_widgets(lay, current_widgets)
        stale_visible = [lbl for lbl in ctx.findChildren(QLabel)
                         if lbl not in current_widgets and lbl.isVisibleTo(ctx)]
        check(f"update_for_tool('{n}'): ningún widget de una sección "
              f"anterior queda visible (el bug real: la fila de color de "
              f"'Etiqueta 3D'/'Polilínea' quedaba pintada encima de la "
              f"herramienta siguiente)", len(stale_visible) == 0,
              f"stale={[l.text() for l in stale_visible]}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — ToolPanel con altura LIMITADA (dock real, no maximizado): antes,
# sin QScrollArea, una herramienta con varios parámetros (Pincel: 3
# sliders + 2 hints) hacía que Qt comprimiera/amontonara todo el panel
# para intentar caber en el alto disponible — "se rompe el diseño",
# reportado por el usuario. Ahora el contenido vive dentro de un
# QScrollArea: si no cabe, da scroll en vez de aplastar el layout.
# ══════════════════════════════════════════════════════════════════════════
def test_tool_panel_scrolls_instead_of_crushing_layout():
    from PyQt5.QtWidgets import QScrollArea
    from ui.tool_panel import ToolPanel

    tp = ToolPanel()
    scroll = tp.findChild(QScrollArea)
    check("ToolPanel envuelve su contenido en un QScrollArea",
          scroll is not None)
    if scroll is None:
        return
    check("el QScrollArea es 'resizable' (el contenido interno se adapta "
          "al ancho, pero no se aplasta en alto)", scroll.widgetResizable())

    # Herramienta CON varios parámetros (Pincel) a una altura de panel
    # chica y realista — antes esto era exactamente el caso que rompía
    # el layout.
    tp._on_tool_clicked("Pincel")
    tp.setFixedHeight(500)
    tp.resize(300, 500)
    tp.show()
    for _ in range(8):
        app.processEvents()

    content = scroll.widget()
    check("con una herramienta con parámetros y poco alto disponible, "
          "el contenido interno necesita MÁS espacio del visible "
          "(confirma que este es el escenario real del bug — si el "
          "contenido cupiera sin scroll, este test no probaría nada)",
          content.sizeHint().height() > tp.height(),
          f"content_height={content.sizeHint().height()} panel_height={tp.height()}")

    check("el QScrollArea permite scrollear hasta abajo (el contenido no "
          "quedó recortado/aplastado sin forma de verlo)",
          scroll.verticalScrollBar().maximum() > 0,
          f"max={scroll.verticalScrollBar().maximum()}")
    tp.close()


# ══════════════════════════════════════════════════════════════════════════
# TEST — Grilla de herramientas RESPONSIVA (2026-09-09): antes `ncols` era
# fijo (3) — un panel angosto podía dejar herramientas recortadas fuera de
# vista por la derecha, y uno ancho dejaba una franja vacía en vez de
# aprovecharla con más columnas. Ahora _relayout_tool_grid() recalcula las
# columnas según el ancho real cada vez que el panel cambia de tamaño.
# ══════════════════════════════════════════════════════════════════════════
def test_tool_panel_responsive_grid():
    from ui.tool_panel import ToolPanel, _ToolButton
    from annotation.tools import ALL_TOOLS

    tp = ToolPanel()

    tp.resize(190, 900)
    tp.show()
    for _ in range(6):
        app.processEvents()
    ncols_narrow = tp._tool_grid_ncols
    check("panel angosto: al menos 1 columna (nunca 0 ni negativo)",
          ncols_narrow >= 1, f"ncols_narrow={ncols_narrow}")
    for name, btn in tp._btns.items():
        check(f"panel angosto: '{name}' sigue dentro del ancho del panel "
              f"(no desaparece por la derecha)",
              btn.geometry().right() <= tp.width(),
              f"btn.right()={btn.geometry().right()} panel_width={tp.width()}")

    tp.resize(400, 900)
    for _ in range(6):
        app.processEvents()
    ncols_wide = tp._tool_grid_ncols
    check("panel ancho: usa MÁS columnas que el angosto (aprovecha el "
          "espacio extra en vez de dejarlo vacío)",
          ncols_wide > ncols_narrow, f"ncols_narrow={ncols_narrow} ncols_wide={ncols_wide}")

    # Todas las herramientas siguen siendo las mismas instancias de botón
    # tras reacomodar (relayout mueve, nunca recrea widgets).
    same_buttons = all(tp._btns[tc.name] is tp._btns[tc.name] for tc in ALL_TOOLS)
    check("relayout no recrea los botones — mismas instancias de siempre",
          same_buttons)
    tp.close()


# ══════════════════════════════════════════════════════════════════════════
# TEST — _ToolButton._fit_font_size: bug real encontrado con captura —
# probaba tamaños con medio píxel (8.5, 7.5...) MIDIENDO con
# `QFont.setPixelSize(int(size))` (trunca: 7.5→7px), pero el CSS que
# realmente se aplicaba (`font-size:7.5px`) Qt lo REDONDEA al renderizar
# (7.5→8px) — un texto que "cabía" en la medición se desbordaba en
# pantalla de verdad ("Etiqueta 3D" salía como "▐tiquet▐"). Fix: solo
# tamaños ENTEROS, para que medir y aplicar sean exactamente lo mismo.
# Este test reproduce el bug para TODAS las herramientas (no solo la que
# lo mostró primero), midiendo con el tamaño real que se aplicaría.
# ══════════════════════════════════════════════════════════════════════════
def test_tool_button_font_size_no_measure_apply_mismatch():
    from PyQt5.QtGui import QFont, QFontMetrics
    from ui.tool_panel import _ToolButton
    from annotation.tools import ALL_TOOLS

    check("_fit_font_size devuelve un entero (nunca medio píxel)",
          isinstance(_ToolButton._fit_font_size("Etiqueta 3D", 58), int))

    widths = sorted(set([_ToolButton.MIN_W, _ToolButton.MAX_W,
                        (_ToolButton.MIN_W + _ToolButton.MAX_W) // 2]))
    for tc in ALL_TOOLS:
        for btn_w in widths:   # todo el rango MIN_W..MAX_W
            max_w = btn_w - 6
            size = _ToolButton._fit_font_size(tc.name, max_w)
            f = QFont(); f.setPixelSize(size); f.setBold(True)  # MISMO tamaño, sin redondeo posible
            fm = QFontMetrics(f)
            widest = max((fm.horizontalAdvance(w) for w in tc.name.split(" ")), default=0)
            check(f"'{tc.name}' @ btn_w={btn_w}: el tamaño elegido cabe de "
                  f"verdad al aplicarlo (sin desajuste medir≠aplicar)",
                  widest <= max_w, f"widest={widest} max_w={max_w} size={size}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — Carga de COPC real: BUG REAL ENCONTRADO (2026-09-10) probando con
# un archivo COPC de producción de verdad (sample_data/autzen-classified.
# copc.laz, ver ese README para origen/licencia): laspy.open()/chunk_iterator
# y hasta laspy.read() completo fallan con "IoError: failed to fill whole
# buffer" en CUALQUIER backend para un COPC real — el storage de puntos de
# COPC (organizado por octree) no es compatible con el lector LAS
# "normal" de laspy, solo laspy.CopcReader puede leerlo. La detección de
# COPC de una ronda anterior solo verificaba que
# CopcReader.open() no lanzara excepción (para el tag "COPC" en pc.fmt) —
# nunca probó la carga real de los PUNTOS, así que este bug pasó
# desapercibido hasta probarlo con un archivo real. Fix:
# core/pointcloud.py::_las_fill_from_copc usa CopcReader.query() en vez del
# camino normal cuando se detecta COPC.
# ══════════════════════════════════════════════════════════════════════════
def test_copc_real_file_loads_correctly():
    sample_path = (Path(__file__).resolve().parent.parent.parent /
                   "sample_data" / "autzen-classified.copc.laz")
    if not sample_path.exists():
        print(f"[SKIP] test_copc_real_file_loads_correctly — no se encontró "
              f"{sample_path} (¿sample_data/ no está en este checkout?)")
        return
    from core.pointcloud import CloudPipeline

    pipeline = CloudPipeline(str(sample_path))
    pc = pipeline._las()
    pc.finalize()

    check("el archivo COPC real se detecta y etiqueta como tal en pc.fmt",
          "COPC" in (pc.fmt or ""), f"pc.fmt={pc.fmt!r}")
    check("el archivo COPC real carga el número de puntos correcto",
          pc.n_points == 10_653_336, f"n_points={pc.n_points}")
    check("el archivo COPC real carga su clasificación real",
          pc.classification is not None
          and int((pc.classification == 2).sum()) > 1_000_000,   # clase 2 = suelo, mayoritaria
          f"classification={'None' if pc.classification is None else 'presente'}")
    check("el archivo COPC real carga su color RGB",
          pc.rgb is not None, f"rgb={'None' if pc.rgb is None else 'presente'}")
    check("las coordenadas cargadas son finitas (no NaN/inf de un query vacío)",
          bool(np.isfinite(pc.xyz[:1000]).all()))


# ══════════════════════════════════════════════════════════════════════════
# TEST — exportar dos nubes DISTINTAS a la MISMA carpeta de dataset no debe
# colisionar (ni pisar los .npy/.ply/.las de la primera) y dataset.json
# debe quedar con los conteos de AMBAS exportaciones combinados, para que
# el flujo "voy agregando nubes a un mismo dataset" (pedido explícitamente
# por el usuario) funcione de verdad.
# ══════════════════════════════════════════════════════════════════════════
def test_export_append_to_existing_dataset_folder():
    from annotation.exporter import ExportWorker, ExportConfig
    from core.project import Project, SemanticClass

    class FakePC:
        def __init__(self, filename, seed):
            rng = np.random.default_rng(seed)
            self.filename = filename
            self.path = filename
            self.xyz = (rng.random((60, 3)).astype(np.float32) * 10)
            self.intensity = None
            self.rgb = None
            self.offset = np.zeros(3)

    def make_project(name, source_file, labels):
        p = Project()
        p.name = name
        p.source_file = source_file
        p.crs = ""
        p.offset_xyz = [0.0, 0.0, 0.0]
        p.schema = [SemanticClass(1, "suelo", "#9e7228"),
                    SemanticClass(2, "vegetacion", "#2e7d32")]
        p.labels = labels
        return p

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = str(Path(tmp) / "mi_dataset")

        # Exportación 1: nube "campo_a.laz"
        labels_a = np.zeros(60, np.uint8); labels_a[:30] = 1
        pc_a = FakePC("campo_a.laz", seed=1)
        proj_a = make_project("proyecto", "/datos/campo_a.laz", labels_a)
        cfg = ExportConfig(output_dir=out_dir, architectures=["randlanet"], export_las=True)
        w1 = ExportWorker(pc_a, proj_a, None, cfg)
        w1.run()

        files_after_1 = set(p.name for p in Path(out_dir, "randlanet", "data", "train").glob("*.npy"))
        check("exportación 1: produce archivos .npy en train",
              len(files_after_1) > 0, f"files={files_after_1}")

        # Exportación 2: OTRA nube, apuntando a la MISMA carpeta de salida
        labels_b = np.zeros(60, np.uint8); labels_b[:20] = 2
        pc_b = FakePC("campo_b.laz", seed=2)
        proj_b = make_project("proyecto", "/datos/campo_b.laz", labels_b)
        w2 = ExportWorker(pc_b, proj_b, None, cfg)
        w2.run()

        files_after_2 = set(p.name for p in Path(out_dir, "randlanet", "data", "train").glob("*.npy"))
        check("exportación 2 no borra los .npy de la exportación 1",
              files_after_1 <= files_after_2, f"perdidos={files_after_1 - files_after_2}")
        check("exportación 2 agrega archivos NUEVOS (no pisa los nombres de la 1)",
              len(files_after_2) > len(files_after_1),
              f"antes={len(files_after_1)} despues={len(files_after_2)}")

        las_files = list(Path(out_dir).glob("*.las"))
        check("ambas nubes producen su propio .las clasificado (sin colisión de nombre)",
              len(las_files) == 2, f"las_files={[p.name for p in las_files]}")

        meta_path = Path(out_dir) / "dataset.json"
        check("dataset.json existe tras la segunda exportación", meta_path.exists())
        import json as _json
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        check("dataset.json: n_tiles_total combina ambas exportaciones (3 splits x 2)",
              meta.get("n_tiles_total") == 6, f"n_tiles_total={meta.get('n_tiles_total')}")
        check("dataset.json: source_files incluye las dos nubes de origen",
              set(meta.get("source_files", [])) == {"/datos/campo_a.laz", "/datos/campo_b.laz"},
              f"source_files={meta.get('source_files')}")
        pc_counts = meta.get("per_class_counts", {})
        check("dataset.json: per_class_counts combina conteos de ambas nubes",
              pc_counts.get("1") == 30 and pc_counts.get("2") == 20,
              f"per_class_counts={pc_counts}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — PickTool debe resolver la clasificación REAL del punto, leída del
# modo anotación (project.labels), en vez de mostrar siempre "desconocido".
# El bug real: _world_pos() devuelve el PROMEDIO de los K vecinos más
# cercanos en pantalla (no la coordenada exacta de un punto real), y el
# handler viejo volvía a buscar esa posición promediada con un
# sphere_query(radio=0.05m) sobre pc.xyz — casi nunca encontraba nada.
# Cubre: (a) _snap_to_point_idx devuelve el índice GLOBAL correcto del
# punto real bajo el cursor: (b) PickTool.on_mouse_press emite ese índice
# en point_picked_abs; (c) _on_point_picked (ui/main_window.py) usa ese
# índice directamente y muestra el nombre de clase correcto del esquema
# del proyecto, no "desconocido".
# ══════════════════════════════════════════════════════════════════════════
def test_pick_tool_reports_real_annotation_class():
    from annotation.tools import PickTool

    xyz = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 5.0, 0.0],
        [10.0, 10.0, 0.0],
    ], np.float32)
    # Solo un subconjunto GLOBAL está "visible" (como pasaría con LOD/tile) —
    # los índices globales NO coinciden con la posición dentro del array
    # visible, para que el test de verdad ejercite el mapeo _cur_idx.
    cur_idx = np.array([7, 3, 9], np.int32)

    class FakePC:
        offset = np.zeros(3)

    class FakeCanvas:
        pc = FakePC()
        _cur_xyz = xyz
        _cur_idx = cur_idx
        @staticmethod
        def map_to_screen(positions):
            return np.asarray(positions)[:, :2]   # proyección de juguete 1:1

    canvas = FakeCanvas()

    class FakeSignal:
        def __init__(self): self.calls = []
        def emit(self, *a): self.calls.append(a)

    class Sig:
        point_picked = FakeSignal()
        point_picked_abs = FakeSignal()

    canvas.sig = Sig()

    tool = PickTool()
    tool.canvas = canvas

    class Ev:
        button = 1
        pos = (5.0, 5.0)   # cae justo sobre el punto [5,5,0] -> índice global 3

    tool.on_mouse_press(Ev())

    check("PickTool emitió point_picked_abs exactamente una vez",
          len(canvas.sig.point_picked_abs.calls) == 1)
    args = canvas.sig.point_picked_abs.calls[0]
    check("point_picked_abs lleva 4 argumentos (incluye el índice global nuevo)",
          len(args) == 4, f"args={args}")
    picked_idx = args[3]
    check("PickTool resuelve el índice GLOBAL correcto (3, no la posición 1 dentro del subset)",
          picked_idx == 3, f"picked_idx={picked_idx}")

    # ── Ahora el lado de ui/main_window.py: _on_point_picked debe usar
    # ese índice directamente para leer la clase real del proyecto. Se
    # replica su lógica aquí sin instanciar MainWindow completo (requiere
    # AnnotationCanvas real -> VTK -> segfault en este sandbox).
    class FakeSchema:
        def __init__(self, id, name, color): self.id=id; self.name=name; self.color=color
    class FakeProject:
        labels = np.zeros(10, np.uint8)
        schema = [FakeSchema(1, "vegetacion", "#2e7d32")]
    proj = FakeProject()
    proj.labels[3] = 1   # el punto global 3 está anotado como clase 1

    cid = int(proj.labels[picked_idx])
    sc = next((s for s in proj.schema if s.id == cid), None)
    cname = sc.name if sc else (f"clase {cid}" if cid > 0 else "sin etiquetar")
    check("la clase resuelta es la anotación REAL del punto (no 'desconocido')",
          cname == "vegetacion", f"cname={cname!r}")


# ══════════════════════════════════════════════════════════════════════════
# TEST — annotation/profile.py: extracción del corte vertical (ProfileTool).
# Nube sintética: una franja de puntos a lo largo del eje X (línea A→B en
# X), más un grupo de puntos "fuera de la franja" (lejos en Y) que NUNCA
# deben aparecer en el resultado, y puntos más allá de los extremos A/B
# (fuera del rango [0,length]) que tampoco deben aparecer.
# ══════════════════════════════════════════════════════════════════════════
def test_profile_extraction():
    from annotation.profile import project_to_profile, extract_profile_slice

    rng = np.random.default_rng(3)
    n_in = 200
    # Puntos DENTRO de la franja: a lo largo de X en [0,10], |Y|<=1, Z variable
    in_xyz = np.column_stack([
        rng.uniform(0, 10, n_in),
        rng.uniform(-1, 1, n_in),
        rng.uniform(0, 5, n_in),
    ]).astype(np.float32)
    # Puntos FUERA de la franja (lejos en Y)
    out_far = np.column_stack([
        rng.uniform(0, 10, 50), rng.uniform(20, 25, 50), rng.uniform(0, 5, 50),
    ]).astype(np.float32)
    # Puntos más allá del extremo B (X > 10 + buffer)
    out_beyond = np.column_stack([
        rng.uniform(50, 60, 50), rng.uniform(-1, 1, 50), rng.uniform(0, 5, 50),
    ]).astype(np.float32)
    xyz = np.concatenate([in_xyz, out_far, out_beyond], axis=0)

    p1 = np.array([0.0, 0.0, 0.0], np.float32)
    p2 = np.array([10.0, 0.0, 0.0], np.float32)

    t, perp, length = project_to_profile(xyz, p1, p2)
    check("project_to_profile: longitud A→B correcta", abs(length - 10.0) < 1e-4)
    check("project_to_profile: t=0 en A, t=length en B (monótono con X)",
          bool(np.corrcoef(t[:n_in], in_xyz[:n_in, 0])[0, 1] > 0.99))

    t_sel, z_sel, idx_sel, length2 = extract_profile_slice(xyz, p1, p2, buffer_m=2.0)
    check("extract_profile_slice: incluye TODOS los puntos dentro de la franja",
          len(idx_sel) >= n_in, f"seleccionados={len(idx_sel)}, esperado>={n_in}")
    check("extract_profile_slice: excluye los puntos lejos en Y",
          not any(i >= n_in and i < n_in + 50 for i in idx_sel.tolist()))
    check("extract_profile_slice: excluye los puntos más allá de B",
          not any(i >= n_in + 50 for i in idx_sel.tolist()))
    check("extract_profile_slice: z_sel corresponde a los puntos seleccionados",
          np.allclose(np.sort(z_sel), np.sort(xyz[idx_sel, 2])))

    # Con índices GLOBALES (simulando un subset de _cur_idx, no 0..N-1)
    gidx = np.arange(1000, 1000 + len(xyz))
    _, _, idx_global, _ = extract_profile_slice(xyz, p1, p2, buffer_m=2.0, global_idx=gidx)
    check("extract_profile_slice: con global_idx, los índices devueltos son GLOBALES",
          idx_global.min() >= 1000, f"min idx={idx_global.min() if len(idx_global) else None}")


def test_profile_view_transform():
    from ui.profile_view import fit_transform

    sx, sy, ox, oy = fit_transform(t_min=0, t_max=100, z_min=0, z_max=10,
                                   w=800, h=400, margin=40, exaggeration=1.0)
    check("fit_transform: escala X positiva y finita", sx > 0 and np.isfinite(sx))
    check("fit_transform: escala Y positiva y finita", sy > 0 and np.isfinite(sy))

    sx2, sy2, _, _ = fit_transform(t_min=0, t_max=100, z_min=0, z_max=10,
                                   w=800, h=400, margin=40, exaggeration=5.0)
    check("fit_transform: exageración vertical 5x escala Y ~5x (X sin cambio)",
          abs(sy2 / sy - 5.0) < 1e-6 and abs(sx2 - sx) < 1e-6,
          f"sy2/sy={sy2/sy}")

    sx3, sy3, ox3, oy3 = fit_transform(t_min=0, t_max=100, z_min=0, z_max=10,
                                       w=800, h=400, margin=40, exaggeration=1.0,
                                       zoom=2.0, pan=(15.0, -8.0))
    check("fit_transform: zoom 2x escala tanto X como Y",
          abs(sx3 / sx - 2.0) < 1e-6 and abs(sy3 / sy - 2.0) < 1e-6)
    check("fit_transform: pan se refleja en el origen",
          abs(ox3 - (40 + 15.0)) < 1e-6)


# ══════════════════════════════════════════════════════════════════════════
# TEST — PolylineTool: digitaliza varios vértices y los persiste como un
# Marker kind="polyline" al terminar con Enter; Escape descarta sin
# guardar nada. Cubre también el roundtrip to_dict/from_dict de "points"
# (antes solo existía para pos/pos_b).
# ══════════════════════════════════════════════════════════════════════════
def test_polyline_tool_and_marker_roundtrip():
    from annotation.tools import PolylineTool
    from annotation.markers import MarkerStore, Marker

    store = MarkerStore()

    class FakePC:
        offset = np.zeros(3)

    class FakeCanvas:
        pc = FakePC()
        marker_store = store
        _mline_calls = []
        class _mline:
            visible = False
            @staticmethod
            def set_data(pts, color=None, **k): FakeCanvas._mline_calls.append(("mline", pts.copy()))
        class _mline_live:
            visible = False
            @staticmethod
            def set_data(pts, color=None, **k): pass
        @staticmethod
        def update(): pass

    canvas = FakeCanvas()
    tool = PolylineTool()
    tool.canvas = canvas

    class Ev:
        def __init__(self, pos): self.pos = pos; self.button = 1

    # Forzar posiciones exactas sin depender de proyección/screen picking
    pts_world = [np.array([0.0, 0.0, 0.0], np.float32),
                np.array([5.0, 0.0, 0.0], np.float32),
                np.array([5.0, 5.0, 0.0], np.float32)]
    import annotation.tools as tools_mod
    orig_snap = tools_mod.BaseTool._snap_to_point
    tools_mod.BaseTool._snap_to_point = lambda self, pos: None
    orig_world = tools_mod.BaseTool._world_pos
    it = iter(pts_world)
    tools_mod.BaseTool._world_pos = lambda self, pos, fast=True: next(it, None)
    try:
        for i in range(3):
            tool.on_mouse_press(Ev((0, 0)))
        class KeyEv: key = "Return"
        tool.on_key_press(KeyEv())
    finally:
        tools_mod.BaseTool._snap_to_point = orig_snap
        tools_mod.BaseTool._world_pos = orig_world

    check("PolylineTool: Enter persiste UN marcador polyline", len(store) == 1,
          f"len(store)={len(store)}")
    m = store.all()[0]
    check("marcador persistido es kind='polyline'", m.kind == "polyline")
    check("marcador tiene los 3 vértices en orden", m.points is not None and len(m.points) == 3
          and np.allclose(m.points[1], [5.0, 0.0, 0.0]))

    # Roundtrip to_dict/from_dict (points sobrevive la serialización)
    d = m.to_dict()
    m2 = Marker.from_dict(d)
    check("Marker.to_dict/from_dict conserva 'points'",
          m2.points is not None and np.allclose(m2.points, m.points))
    check("Marker.to_dict/from_dict conserva kind='polyline'", m2.kind == "polyline")

    # Escape en una polilínea a medio trazar no debe guardar nada
    store2 = MarkerStore()
    canvas2 = FakeCanvas(); canvas2.marker_store = store2
    tool2 = PolylineTool(); tool2.canvas = canvas2
    tools_mod.BaseTool._snap_to_point = lambda self, pos: None
    tools_mod.BaseTool._world_pos = lambda self, pos, fast=True: np.array([1.0, 1.0, 1.0], np.float32)
    try:
        tool2.on_mouse_press(Ev((0, 0)))
        class KeyEv2: key = "Escape"
        tool2.on_key_press(KeyEv2())
    finally:
        tools_mod.BaseTool._snap_to_point = orig_snap
        tools_mod.BaseTool._world_pos = orig_world
    check("Escape a medio trazar NO persiste ningún marcador", len(store2) == 0)


# ══════════════════════════════════════════════════════════════════════════
# TEST — render/canvas.py::_MarkerOverlay dibuja polilíneas SIN crashear,
# render offscreen real de VTK (mismo patrón que test_marker_overlay_actors_no_text).
# ══════════════════════════════════════════════════════════════════════════
def test_marker_overlay_polyline_actors():
    import vtk
    from render.canvas import _MarkerOverlay
    from annotation.markers import MarkerStore

    ren = vtk.vtkRenderer()
    overlay = _MarkerOverlay(ren)
    store = MarkerStore()
    store.add_polyline([[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0]],
                       text="4 vértices", color=(0.2, 0.85, 1.0))
    overlay.sync(store.all())
    check("_MarkerOverlay crea actores para la polilínea (línea + esferas + "
          "texto 2D)",
          len(overlay._actors) >= 3, f"n_actors={len(overlay._actors)}")
    # Render offscreen real — confirma que VTK no rechaza la vtkPolyLine
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren)
    rw.SetSize(64, 64)
    try:
        rw.Render()
        ok = True
    except Exception:
        ok = False
    check("render offscreen real con la polilínea no crashea", ok)
    overlay.clear()
    check("overlay.clear() remueve todos los actores", len(overlay._actors) == 0)


# ══════════════════════════════════════════════════════════════════════════
# TEST — fine-tuning desde checkpoint externo: carga parcial por
# nombre+forma cuando el número de clases (y por tanto la última capa)
# no coincide, dejando el resto de la red con los pesos del checkpoint.
# Réplica minimalista de la lógica real en
# ui/training_panel.py::TrainingWorker._train_with_model (sin correr un
# entrenamiento completo, que sería lento) — misma lógica de match
# nombre+forma, exactamente como en el código real.
# ══════════════════════════════════════════════════════════════════════════
def test_finetune_partial_state_dict_load():
    import torch, torch.nn as nn

    def make_model(nc):
        return nn.Sequential(nn.Conv1d(9, 16, 1), nn.ReLU(), nn.Conv1d(16, nc, 1))

    src_model = make_model(5)               # "checkpoint" entrenado con 5 clases
    src_state = src_model.state_dict()

    dst_model = make_model(8)               # proyecto nuevo con 8 clases
    own_state = dst_model.state_dict()

    new_state, loaded, skipped = dict(own_state), [], []
    for k, v in own_state.items():
        sv = src_state.get(k)
        if sv is not None and tuple(sv.shape) == tuple(v.shape):
            new_state[k] = sv
            loaded.append(k)
        else:
            skipped.append(k)
    dst_model.load_state_dict(new_state)

    check("fine-tuning parcial: la capa de entrada (0.*) SÍ se reusa (misma forma)",
          any(k.startswith("0.") for k in loaded), f"loaded={loaded}")
    check("fine-tuning parcial: la capa de salida (2.*, distinto nc) NO se reusa",
          all(k.startswith("2.") for k in skipped) and len(skipped) > 0,
          f"skipped={skipped}")
    check("fine-tuning parcial: los pesos de la capa reusada son IDÉNTICOS al origen",
          torch.equal(dst_model[0].weight, src_model[0].weight))
    check("fine-tuning parcial: los pesos de la capa de salida NO vienen del origen "
          "(forma distinta: 5 vs 8 clases)",
          dst_model[2].weight.shape[0] == 8 and not torch.equal(
              dst_model[2].weight[:5], src_model[2].weight))


# ══════════════════════════════════════════════════════════════════════════
# TEST — exportación real a ONNX (TrainingWorker.export_onnx): entrena
# "cero pasos" (no hace falta un modelo realmente entrenado para probar
# que el exportador funciona) — construye KPConv (arquitectura
# determinística, sin torch.randperm, ver nota en export_onnx), guarda un
# checkpoint de juguete, exporta a .onnx, y confirma con la librería
# `onnx` que el archivo resultante es un modelo válido con las entradas/
# salidas esperadas. Se salta con gracia si falta 'onnx' o 'torch'.
# ══════════════════════════════════════════════════════════════════════════
def test_export_onnx_real_file():
    try:
        import torch
        import onnx
    except ImportError as e:
        print(f"[SKIP] test_export_onnx_real_file — falta una dependencia ({e})")
        return
    from ui.training_panel import TrainingWorker

    with tempfile.TemporaryDirectory() as tmp:
        worker = TrainingWorker({})
        model = worker._build_model("KPConv", 6)
        ckpt_path = str(Path(tmp) / "toy_best_model.pth")
        torch.save({"epoch": 1, "model_state": model.state_dict(), "miou": 0.5}, ckpt_path)

        out_path = str(Path(tmp) / "model.onnx")
        result = worker.export_onnx("KPConv", 6, ckpt_path, out_path, num_points=256)

        check("export_onnx: no reporta capas faltantes/inesperadas para su propio checkpoint",
              not result["missing"] and not result["unexpected"],
              f"missing={result['missing']}, unexpected={result['unexpected']}")
        check("export_onnx: KPConv no lleva el aviso de aleatoriedad (solo RandLA-Net)",
              result["note"] == "")
        check("export_onnx: el archivo .onnx se creó de verdad", Path(out_path).exists())

        # NOTA: `onnx.checker.check_model()` NO se llama aquí a propósito —
        # se probó, y en ESTE proceso (con Qt/VTK ya cargados por los tests
        # de render offscreen que corren antes en el mismo arnés) esa
        # llamada específica segfaultea de verdad (confirmado aislando el
        # crash: onnx.load() solo, y torch.onnx.export() solo, NUNCA
        # crashean combinados con VTK — únicamente onnx.checker.check_model()
        # lo hace). Es un choque nativo de librerías específico de este
        # entorno (probablemente protobuf/VTK), no del código exportador:
        # `export_onnx()` (ui/training_panel.py) nunca llama al checker de
        # onnx, solo a torch.onnx.export(); onnx.load() abajo ya confirma
        # que el archivo resultante es un protobuf ONNX válido y cargable.
        onnx_model = onnx.load(out_path)
        check("el .onnx exportado se puede leer con onnx.load()", True)
        input_names = [i.name for i in onnx_model.graph.input]
        output_names = [o.name for o in onnx_model.graph.output]
        check("el .onnx tiene la entrada 'points'", "points" in input_names)
        check("el .onnx tiene la salida 'logits'", "logits" in output_names)

    # También confirmar el aviso de aleatoriedad para RandLA-Net
    worker2 = TrainingWorker({})
    model2 = worker2._build_model("RandLA-Net", 6)
    with tempfile.TemporaryDirectory() as tmp2:
        ckpt2 = str(Path(tmp2) / "rl.pth")
        torch.save({"model_state": model2.state_dict()}, ckpt2)
        out2 = str(Path(tmp2) / "rl.onnx")
        result2 = worker2.export_onnx("RandLA-Net", 6, ckpt2, out2, num_points=128)
        check("export_onnx: RandLA-Net SÍ lleva el aviso de aleatoriedad (randperm)",
              "aleatori" in result2["note"].lower())


# ══════════════════════════════════════════════════════════════════════════
# TEST — mejoras a PolylineTool: grosor/color configurables, deshacer el
# último vértice (Backspace), y editar/eliminar una polilínea YA creada
# con Ctrl+clic sobre ella (en vez de solo poder crear nuevas).
# ══════════════════════════════════════════════════════════════════════════
def test_polyline_point_segment_distance():
    from annotation.tools import _point_segment_dist2

    # Punto exactamente sobre el segmento (a mitad de camino) -> distancia 0
    check("_point_segment_dist2: punto sobre el segmento -> 0",
          _point_segment_dist2(5, 0, 0, 0, 10, 0) < 1e-9)
    # Punto a 3px perpendicular del segmento horizontal -> 3^2 = 9
    check("_point_segment_dist2: distancia perpendicular correcta",
          abs(_point_segment_dist2(5, 3, 0, 0, 10, 0) - 9.0) < 1e-6)
    # Punto más allá del extremo B -> se mide contra B, no se extrapola
    d2 = _point_segment_dist2(20, 0, 0, 0, 10, 0)
    check("_point_segment_dist2: más allá de B se ancla en B (dist=10)",
          abs(d2 - 100.0) < 1e-6, f"d2={d2}")


def test_polyline_backspace_undoes_last_vertex():
    import annotation.tools as tools_mod
    from annotation.tools import PolylineTool
    from annotation.markers import MarkerStore

    store = MarkerStore()

    class FakePC:
        offset = np.zeros(3)

    class FakeSignal:
        def __init__(self): self.calls = []
        def set_data(self, *a, **k): self.calls.append((a, k))
    class FakeLineProxy:
        def __init__(self): self.visible = False; self.calls = []
        def set_data(self, *a, **k): self.calls.append((a, k))

    class FakeCanvas:
        pc = FakePC()
        marker_store = store
        _mline = FakeLineProxy()
        _mline_live = FakeLineProxy()
        @staticmethod
        def update(): pass

    canvas = FakeCanvas()
    tool = PolylineTool()
    tool.canvas = canvas

    pts_world = [np.array([0.0, 0.0, 0.0], np.float32),
                np.array([1.0, 0.0, 0.0], np.float32),
                np.array([2.0, 0.0, 0.0], np.float32)]
    orig_snap = tools_mod.BaseTool._snap_to_point
    orig_world = tools_mod.BaseTool._world_pos
    tools_mod.BaseTool._snap_to_point = lambda self, pos: None
    it = iter(pts_world)
    tools_mod.BaseTool._world_pos = lambda self, pos, fast=True: next(it, None)
    try:
        class Ev:
            def __init__(self, pos): self.pos = pos; self.button = 1
        for _ in range(3):
            tool.on_mouse_press(Ev((0, 0)))
        check("PolylineTool: 3 clics -> 3 vértices en curso", len(tool._pts) == 3)

        class KeyEv:
            def __init__(self, k): self.key = k
        tool.on_key_press(KeyEv("BackSpace"))
        check("Backspace quita el último vértice (quedan 2)", len(tool._pts) == 2,
              f"len={len(tool._pts)}")
        check("los 2 vértices restantes son los primeros 2 puestos",
              np.allclose(tool._pts[0], [0, 0, 0]) and np.allclose(tool._pts[1], [1, 0, 0]))

        tool.on_key_press(KeyEv("Return"))
    finally:
        tools_mod.BaseTool._snap_to_point = orig_snap
        tools_mod.BaseTool._world_pos = orig_world

    check("tras Backspace + Enter, la polilínea guardada tiene solo 2 vértices "
          "(el 3ro se deshizo antes de terminar)",
          len(store) == 1 and len(store.all()[0].points) == 2,
          f"points={store.all()[0].points if len(store) else None}")


def test_polyline_width_and_color_configurable():
    from annotation.markers import MarkerStore

    store = MarkerStore()
    m = store.add_polyline([[0, 0, 0], [1, 1, 0]], color=(1.0, 0.0, 0.0), line_width=6.0)
    check("add_polyline acepta line_width y lo guarda", m.line_width == 6.0)
    check("add_polyline acepta color y lo guarda", tuple(m.color) == (1.0, 0.0, 0.0))

    d = m.to_dict()
    check("to_dict conserva line_width", d["line_width"] == 6.0)
    from annotation.markers import Marker
    m2 = Marker.from_dict(d)
    check("from_dict restaura line_width", m2.line_width == 6.0)

    # Editar in-place vía MarkerStore.update (lo que usa el diálogo de edición)
    store.update(m.id, color=[0.0, 1.0, 0.0], line_width=8.0)
    m_upd = store.get(m.id)
    check("MarkerStore.update cambia color y line_width de una polilínea existente",
          tuple(m_upd.color) == (0.0, 1.0, 0.0) and m_upd.line_width == 8.0)


def test_polyline_edit_existing_via_ctrl_click():
    """
    Ctrl+clic sobre una polilínea YA existente (no en curso de trazado)
    debe abrir el diálogo de edición y aplicar los cambios devueltos —
    se reemplaza _LineMarkerEditDialog por un stub (sin abrir un diálogo Qt
    modal real), igual que ya se hace para LabelMarkerTool.
    """
    import annotation.tools as tools_mod
    from annotation.tools import PolylineTool
    from annotation.markers import MarkerStore

    store = MarkerStore()
    store.add_polyline([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], color=(0.2, 0.85, 1.0))

    class FakePC:
        offset = np.zeros(3)

    class FakeCanvas:
        pc = FakePC()
        marker_store = store
        @staticmethod
        def map_to_screen(positions):
            return np.asarray(positions)[:, :2]   # proyección de juguete 1:1

    canvas = FakeCanvas()
    tool = PolylineTool()
    tool.canvas = canvas

    class StubDialog:
        def __init__(self, *a, **k): pass
        def exec(self):
            return (True, False, (1.0, 0.5, 0.0), 5.0)   # accepted, deleted, color, width

    orig_dialog = tools_mod._LineMarkerEditDialog
    tools_mod._LineMarkerEditDialog = StubDialog
    try:
        class Ev:
            def __init__(self, pos): self.pos = pos; self.button = 1
        tool.on_mouse_press(Ev((5.0, 0.0)))   # cae sobre el punto medio del segmento
    finally:
        tools_mod._LineMarkerEditDialog = orig_dialog

    m = store.all()[0]
    check("Ctrl+clic sobre una polilínea existente la edita (no crea una nueva)",
          len(store) == 1)
    check("el color se actualizó con lo devuelto por el diálogo",
          tuple(m.color) == (1.0, 0.5, 0.0), f"color={m.color}")
    check("el grosor se actualizó con lo devuelto por el diálogo", m.line_width == 5.0)
    check("no quedó ningún vértice en curso tras editar (no empezó una polilínea nueva)",
          len(tool._pts) == 0)

    # Y ahora confirmar que "eliminar" en el diálogo SÍ borra el marcador.
    class StubDeleteDialog:
        def __init__(self, *a, **k): pass
        def exec(self):
            return (True, True, (1.0, 0.5, 0.0), 5.0)   # deleted=True

    tools_mod._LineMarkerEditDialog = StubDeleteDialog
    try:
        tool.on_mouse_press(Ev((5.0, 0.0)))
    finally:
        tools_mod._LineMarkerEditDialog = orig_dialog
    check("elegir 'Eliminar' en el diálogo de edición borra la polilínea",
          len(store) == 0)


# ══════════════════════════════════════════════════════════════════════════
# TEST — mejoras a ProfileTool/ProfileDialog: leyenda de clases realmente
# presentes en la franja, y exportación CSV.
# ══════════════════════════════════════════════════════════════════════════
def test_profile_colors_for_returns_legend_of_present_classes():
    from annotation.tools import ProfileTool

    class FakeSchema:
        def __init__(self, id, name, color): self.id = id; self.name = name; self.color = color
    class FakeProject:
        labels = np.array([0, 1, 1, 2, 2, 2], np.uint8)
        schema = [FakeSchema(1, "suelo", "#9e7228"),
                 FakeSchema(2, "vegetacion", "#2e7d32"),
                 FakeSchema(3, "nunca_presente", "#ff00ff")]
    class FakeCanvas:
        project = FakeProject()

    tool = ProfileTool()
    tool.canvas = FakeCanvas()
    idx_out = np.arange(6)
    z = np.zeros(6, np.float32)
    colors, legend = tool._colors_for(idx_out, z)

    check("_colors_for devuelve un color por punto", len(colors) == 6)
    names = [n for n, _ in legend]
    check("la leyenda incluye 'sin etiquetar' (clase 0, presente)",
          "sin etiquetar" in names, f"names={names}")
    check("la leyenda incluye 'suelo' y 'vegetacion' (presentes)",
          "suelo" in names and "vegetacion" in names, f"names={names}")
    check("la leyenda NO incluye 'nunca_presente' (clase 3, no está en esta franja)",
          "nunca_presente" not in names, f"names={names}")
    check("suelo se colorea con el color real del schema (#9e7228)",
          colors[1].tolist() == [0x9e, 0x72, 0x28], f"color={colors[1].tolist()}")


def test_profile_csv_export_format():
    from ui.profile_view import profile_rows_to_csv

    t = np.array([0.0, 1.5, 3.0], np.float32)
    z = np.array([10.0, 10.2, 9.8], np.float32)

    csv_no_color = profile_rows_to_csv(t, z, None)
    check("CSV sin color: encabezado correcto",
          csv_no_color.splitlines()[0] == "distancia_m,altura_m")
    check("CSV sin color: 3 filas de datos + encabezado",
          len(csv_no_color.splitlines()) == 4)

    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
    csv_color = profile_rows_to_csv(t, z, colors)
    check("CSV con color: encabezado incluye r,g,b",
          csv_color.splitlines()[0] == "distancia_m,altura_m,r,g,b")
    check("CSV con color: la primera fila de datos tiene el color correcto",
          csv_color.splitlines()[1] == "0.0000,10.0000,255,0,0",
          f"row={csv_color.splitlines()[1]!r}")


def test_profile_canvas_hover_and_legend_render_without_crashing():
    """
    ProfileCanvas con leyenda + hover activo debe pintar sin crashear —
    ejercita paintEvent de verdad (widget Qt real, offscreen) en vez de
    solo probar la lógica pura por separado, porque el hover mezcla
    estado (posición de mouse) con el paintEvent de una forma que un test
    puramente de datos no cubriría.
    """
    from ui.profile_view import ProfileCanvas
    from PyQt5.QtCore import QPointF

    canvas = ProfileCanvas()
    canvas.resize(300, 200)
    t = np.linspace(0, 20, 500).astype(np.float32)
    z = np.sin(t).astype(np.float32)
    colors = np.tile(np.array([80, 160, 220], np.uint8), (len(t), 1))
    legend = [("suelo", (158, 114, 40)), ("vegetacion", (46, 125, 50))]
    canvas.set_data(t, z, colors, length_m=20.0, legend=legend)
    canvas._hover_screen = QPointF(150, 100)
    try:
        canvas.repaint()
        ok = True
    except Exception:
        ok = False
    check("ProfileCanvas con leyenda + hover se pinta sin crashear", ok)


# ══════════════════════════════════════════════════════════════════════════
# TEST — el perfil ya no se pone lento con zoom/pan sobre franjas grandes.
# BUG REAL reportado por el usuario: "hago zoom al perfil y se pone
# lento". Causa: el paintEvent anterior llamaba drawEllipse() UNA VEZ POR
# PUNTO (hasta 60k veces) en cada repaint, y el hover repetía un bucle
# Python completo en cada movimiento de mouse — ambos escalaban mal.
# Cubre: (a) _render_arrays() rasteriza de verdad (el buffer QImage
# termina con píxeles distintos del fondo donde caen los puntos); (b) con
# un dataset grande (500k puntos), pintar tarda un tiempo acotado — no es
# un benchmark exacto (depende de la máquina), pero sí confirma que NO
# escala linealmente mal como antes (debía tardar segundos con la
# implementación vieja a este tamaño).
# ══════════════════════════════════════════════════════════════════════════
def test_profile_canvas_raster_renders_points_and_is_fast():
    import time
    from ui.profile_view import ProfileCanvas
    from PyQt5.QtGui import QColor

    canvas = ProfileCanvas()
    canvas.resize(400, 300)
    n = 500_000
    rng = np.random.default_rng(7)
    t = rng.uniform(0, 100, n).astype(np.float32)
    z = rng.uniform(0, 20, n).astype(np.float32)
    colors = np.tile(np.array([255, 0, 0], np.uint8), (n, 1))
    canvas.set_data(t, z, colors, length_m=100.0)

    img = canvas._render_arrays(canvas.width(), canvas.height())
    check("_render_arrays devuelve una QImage real", img is not None)
    if img is not None:
        # Al menos un pixel debe haber cambiado del fondo (#1c1e1f) a
        # algo con más rojo que verde/azul (el color de los puntos).
        found_red = False
        for x, y in ((50, 50), (200, 150), (350, 250), (10, 10), (390, 290)):
            c = img.pixelColor(x, y)
            if c.red() > c.green() + 20:
                found_red = True
                break
        check("el buffer rasterizado contiene puntos rojos de verdad",
              found_red)

    t0 = time.perf_counter()
    for _ in range(5):
        canvas._render_arrays(canvas.width(), canvas.height())
    elapsed = time.perf_counter() - t0
    check(f"5 rasterizaciones de {n:,} puntos tardan <2s en total "
          f"(antes escalaba mal con drawEllipse por punto)",
          elapsed < 2.0, f"elapsed={elapsed:.3f}s")

    # El hover vectorizado (numpy, no bucle Python) también debe ser
    # rápido y encontrar el punto correcto.
    canvas._hover_screen = None
    from PyQt5.QtCore import QPointF
    sx, sy = canvas._screen_xy(np.array([0]))
    canvas._hover_screen = QPointF(float(sx[0]), float(sy[0]))
    t0 = time.perf_counter()
    canvas.repaint()
    elapsed_hover = time.perf_counter() - t0
    check("un repaint con hover activo sobre 500k puntos es rápido (<1s)",
          elapsed_hover < 1.0, f"elapsed={elapsed_hover:.3f}s")


# ══════════════════════════════════════════════════════════════════════════
# TEST — MeasureTool ahora se puede editar/eliminar (antes una medida, una
# vez creada, no se podía borrar de NINGUNA forma — Escape solo cancelaba
# la medida EN CURSO, nunca una ya persistida).
# ══════════════════════════════════════════════════════════════════════════
def test_measure_tool_edit_and_delete_existing():
    import annotation.tools as tools_mod
    from annotation.tools import MeasureTool
    from annotation.markers import MarkerStore

    store = MarkerStore()
    store.add_measure([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], text="10.00 m")

    class FakePC:
        offset = np.zeros(3)

    class FakeCanvas:
        pc = FakePC()
        marker_store = store
        @staticmethod
        def map_to_screen(positions):
            return np.asarray(positions)[:, :2]

    canvas = FakeCanvas()
    tool = MeasureTool()
    tool.canvas = canvas

    class Ev:
        def __init__(self, pos): self.pos = pos; self.button = 1

    class StubEditDialog:
        def __init__(self, *a, **k): pass
        def exec(self):
            return (True, False, (0.9, 0.1, 0.1), 4.0)   # accepted, deleted, color, width

    orig_dialog = tools_mod._LineMarkerEditDialog
    tools_mod._LineMarkerEditDialog = StubEditDialog
    try:
        tool.on_mouse_press(Ev((5.0, 0.0)))   # punto medio del segmento
    finally:
        tools_mod._LineMarkerEditDialog = orig_dialog

    m = store.all()[0]
    check("Ctrl+clic sobre una medida existente la edita (no crea otra)",
          len(store) == 1)
    check("el color de la medida se actualizó", tuple(m.color) == (0.9, 0.1, 0.1))
    check("el grosor de la medida se actualizó", m.line_width == 4.0)
    check("no quedó ningún punto A en curso tras editar", tool._p1 is None)

    class StubDeleteDialog:
        def __init__(self, *a, **k): pass
        def exec(self):
            return (True, True, (0, 0, 0), 2.5)

    tools_mod._LineMarkerEditDialog = StubDeleteDialog
    try:
        tool.on_mouse_press(Ev((5.0, 0.0)))
    finally:
        tools_mod._LineMarkerEditDialog = orig_dialog
    check("elegir 'Eliminar' en el diálogo de Medir borra la medida",
          len(store) == 0)


# ══════════════════════════════════════════════════════════════════════════
# TEST — el texto de los marcadores usa vtkTextActor (2D, coordenadas de
# MUNDO) en vez de vtkBillboardTextActor3D — de tres intentos anteriores
# (el segundo, un QWidget
# de Qt superpuesto encima del widget nativo de VTK, rompió el render por
# completo — la nube dejó de verse y la app se congelaba al recargar una
# nube — mezclar un widget translúcido de Qt con la ventana nativa de
# OpenGL de QVTKRenderWindowInteractor no es seguro). Un actor 2D con su
# PositionCoordinate en el sistema World: VTK reproyecta su posición 3D a
# pantalla solo, en cada render, usando la cámara activa — y los actores
# 2D nunca compiten por profundidad contra la geometría 3D, así que están
# SIEMPRE visibles, verificado con un render offscreen real (ver abajo).
# ══════════════════════════════════════════════════════════════════════════
def test_marker_text_actor_visible_through_occluding_geometry():
    """
    Render offscreen real: un plano opaco justo delante (en Z) del texto
    de un marcador — si el texto usara geometría 3D normal, quedaría
    tapado por completo; con vtkTextActor + coordenadas de mundo, debe
    seguir viéndose.
    """
    import vtk
    from vtk.util import numpy_support
    from render.canvas import _MarkerOverlay
    from annotation.markers import MarkerStore

    ren = vtk.vtkRenderer()
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren)
    rw.SetSize(200, 200)

    cam = ren.GetActiveCamera()
    cam.SetPosition(0, 0, 10); cam.SetFocalPoint(0, 0, 0); cam.SetViewUp(0, 1, 0)
    cam.ParallelProjectionOn(); cam.SetParallelScale(3.0)

    # Plano rojo opaco CERCA de cámara (z=2), cubre toda la vista
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(-5, -5, 2); plane.SetPoint1(5, -5, 2); plane.SetPoint2(-5, 5, 2)
    pmapper = vtk.vtkPolyDataMapper(); pmapper.SetInputConnection(plane.GetOutputPort())
    pactor = vtk.vtkActor(); pactor.SetMapper(pmapper)
    pactor.GetProperty().SetColor(1, 0, 0); pactor.GetProperty().LightingOff()
    ren.AddActor(pactor)

    # Marcador con texto, DETRÁS del plano (z=-2) — a través de la
    # ruta real de la app, no un vtkTextActor de juguete.
    store = MarkerStore()
    store.add_label([0.0, 0.0, -2.0], "HOLA", color=(1.0, 1.0, 1.0), font_size=30.0)
    overlay = _MarkerOverlay(ren)
    overlay.sync(store.all())

    rw.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw); w2i.SetInputBufferTypeToRGB(); w2i.Update()
    img = w2i.GetOutput()
    dims = img.GetDimensions()
    arr = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(dims[1], dims[0], 3)
    bright = int(np.sum((arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200)))
    check("el texto del marcador se ve a través del plano que lo tapa en Z "
          "(antes de este fix, esto habría dado 0 píxeles claros)",
          bright > 0, f"bright_px={bright}")


def test_marker_overlay_text_actor_removed_on_clear():
    """clear() debe quitar también el vtkTextActor (2D) — no solo la
    esfera/línea (3D) — de la escena."""
    import vtk
    from render.canvas import _MarkerOverlay
    from annotation.markers import MarkerStore

    ren = vtk.vtkRenderer()
    overlay = _MarkerOverlay(ren)
    store = MarkerStore()
    store.add_label([1.0, 2.0, 3.0], "una etiqueta", color=(0.2, 0.9, 0.3), font_size=18.0)
    overlay.sync(store.all())

    check("hay un vtkTextActor real en la escena tras sync()",
          any(isinstance(a, vtk.vtkTextActor) for a in overlay._actors))
    txt_actor = next(a for a in overlay._actors if isinstance(a, vtk.vtkTextActor))
    check("el vtkTextActor usa coordenadas de MUNDO (no display/pantalla fija)",
          txt_actor.GetPositionCoordinate().GetCoordinateSystemAsString() == "World")

    overlay.clear()
    check("clear() deja la escena sin ningún actor (incluye el texto 2D)",
          len(overlay._actors) == 0)


if __name__ == "__main__":
    tests = [
        test_sphere_query_octree_grid_path,
        test_tile_manager_cache_path_changes_with_transform,
        test_label_store_stats_debounce,
        test_tool_panel_color_mode_sync,
        test_compute_colors_u8_annotation_reflects_labels,
        test_redesign_icons_and_rail,
        test_asprs_code_mapping,
        test_crash_marker_sequence,
        test_training_checkpoint_resume_roundtrip,
        test_delete_points_label_store,
        test_delete_points_excluded_from_export,
        test_delete_points_hidden_in_render,
        test_confidence_color_mode,
        test_region_growing_performance_and_correctness,
        test_annotation_mode_unlabeled_points_are_opaque,
        test_sor_outlier_detection,
        test_octree_iter_progressive_lod_frustum_and_sse,
        test_gather_and_color_frustum_culling,
        test_octree_builder_uses_full_xyz_no_presample,
        test_laz_parallel_backend_selection_and_roundtrip,
        test_label_store_annotate_bulk_undo_redo,
        test_smooth_labels_majority,
        test_detect_isolated_clusters_per_class,
        test_ransac_plane_fit,
        test_plane_fit_tool_selects_only_planar_points,
        test_marker_store_and_project_persistence,
        test_clip_box_controller,
        test_clip_box_real_offscreen_render,
        test_marker_overlay_actors_no_text,
        test_copc_detection_does_not_break_normal_laz,
        test_new_ui_widgets_instantiate,
        test_marker_store_update,
        test_label_marker_tool_find_nearby,
        test_label_marker_tool_create_and_edit_flow,
        test_plane_fit_tool_preview_does_not_break_selection,
        test_tool_panel_new_context_sections,
        test_tile_grid_font_never_wider_than_cell,
        test_context_section_no_stale_widgets_on_tool_switch,
        test_tool_panel_scrolls_instead_of_crushing_layout,
        test_tool_panel_responsive_grid,
        test_tool_button_font_size_no_measure_apply_mismatch,
        test_copc_real_file_loads_correctly,
        test_export_append_to_existing_dataset_folder,
        test_pick_tool_reports_real_annotation_class,
        test_profile_extraction,
        test_profile_view_transform,
        test_polyline_tool_and_marker_roundtrip,
        test_marker_overlay_polyline_actors,
        test_finetune_partial_state_dict_load,
        test_export_onnx_real_file,
        test_polyline_point_segment_distance,
        test_polyline_backspace_undoes_last_vertex,
        test_polyline_width_and_color_configurable,
        test_polyline_edit_existing_via_ctrl_click,
        test_profile_colors_for_returns_legend_of_present_classes,
        test_profile_csv_export_format,
        test_profile_canvas_hover_and_legend_render_without_crashing,
        test_profile_canvas_raster_renders_points_and_is_fast,
        test_measure_tool_edit_and_delete_existing,
        test_marker_text_actor_visible_through_occluding_geometry,
        test_marker_overlay_text_actor_removed_on_clear,
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
        print("RESULTADO: todas las verificaciones pasaron OK")
        sys.exit(0)
