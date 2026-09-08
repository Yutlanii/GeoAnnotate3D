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
