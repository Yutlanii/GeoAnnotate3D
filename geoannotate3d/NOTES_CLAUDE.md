# GeoAnnotate3D — Notas de análisis (para mí mismo, Claude)

> Archivo de trabajo interno. Resume la arquitectura completa del proyecto tal como
> existe hoy (2026-09-04) para no tener que re-explorar todo cada vez que retome
> el trabajo. Actualízalo si cambias algo estructural importante.

## 1. Qué es el proyecto

**GeoAnnotate3D** (v1.0.0 en `main.py`, README dice "v0.2.0 Turbo Edition") es una
app de escritorio (PyQt5 + VTK) para **anotación semántica manual/semi-automática de
nubes de puntos LiDAR geoespaciales** (edificios, vegetación, suelo, etc.), pensada
para generar datasets de entrenamiento para redes de segmentación de nubes de puntos:
**RandLA-Net, KPConv, PointNet++**. También incluye un flujo de entrenamiento e
inferencia embebido (pasos 5 y 6 del workflow).

No es un repo git todavía (`Is a git repository: false`). Sin remoto de GitHub aún.
Windows-first (dark titlebar via DWM, `.bat` de build, rutas `\`), pero el código en
sí es multiplataforma (PyQt5/VTK/numpy).

Idioma de la UI y comentarios: **español**. Mantener ese idioma en cambios de UI/UX.

## 2. Cómo arranca (`main.py`)

1. Crea `QApplication`, aplica HiDPI, fuente Segoe UI 10pt.
2. Muestra `ui/splash.py::SplashScreen` con progreso: compila extensión C si falta
   (`core/build_extension.py::build_if_needed`), importa VTK, numpy, scipy, laspy.
3. Crea `ui/main_window.py::MainWindow`, `showMaximized()`.
4. 900ms después: cierra splash, muestra `ui/welcome_dialog.py::WelcomeDialog`
   (onboarding de 6 pasos, solo primera vez) y luego el diálogo de "abrir nube"
   (`MainWindow._maybe_show_onboarding` → `_build_onboarding_dialog`).
5. Fuerza barra de título oscura en Windows vía `ctypes` + `DwmSetWindowAttribute`.
6. Si se pasa un archivo por argv, lo abre automáticamente (`window.open_file`).

## 3. Layout de la ventana principal (`ui/main_window.py`, 2300 líneas)

```
_HeaderBar (top, 38px)        logo "GA·3D" | archivo | CRS badge | modo badge
                               (Overview / Tile(col,row)) | menú "≡ Proyecto" | timer | guardado
_workflow_bar (46px, full width) — 6 pasos clicables (barra de flujo horizontal):
   1 Nube  ›  2 Pre-clasificar  ›  3 Etiquetar  ›  4 Exportar  ›  5 Entrenar  ›  6 Inferir
   + botón "⬟ Capas" a la derecha (toggle panel de overlays)
_body_stack (QStackedWidget):
   [0] _main_splitter (QSplitter horizontal):
         _left_stack (QStackedWidget, 160-480px) — cambia según paso:
            0: TilePanel      (paso 1 — Nube: grid de tiles, tamaño, transform)
            1: GeoPanel       (paso 2 — Pre-clasificar: MDT/AGL, CSF, Region Growing, reglas)
            2: ClassPanel     (paso 3 — Etiquetar: lista de clases, clase activa, progreso)
            3: OverlayPanel   (capas — accesible con botón "Capas", no es un paso)
         AnnotationCanvas (centro, VTK)
         ToolPanel (derecha, 180-400px) — herramientas + opciones + visualización + capas rápido
   [1] TrainingPanel  (paso 5, ocupa toda la ventana)
   [2] InferPanel     (paso 6, ocupa toda la ventana)
AnnotationStatusBar (status bar de Qt) — fps, conteos, balance de clases, tiempo sesión
```

Tema visual: **`ui/styles.py`** — QSS global "amber/oscuro cálido".
Paleta: fondo `#0e0c0a`/`#080604`, acento ámbar `#c07818` (hover `#d88818`),
texto primario `#c4c0b8`, secundario `#706858`, bordes `#1e1c14`/`#282418`.
Fuente monoespaciada forzada globalmente (`Cascadia Code/Consolas`, 12px) — **esto
es inusual para una UI de producto y probablemente lo primero a reconsiderar en el
rediseño** (dificulta jerarquía tipográfica, se ve "tech/hacker" no "herramienta pro").
Casi todo el estilo está **inline como strings QSS por widget**, no centralizado —
mucha duplicación de colores hardcodeados repetidos en cada archivo `ui/*.py`.

## 4. El flujo de trabajo en 6 pasos (workflow bar)

Es el eje conceptual de toda la UX. `MainWindow._on_workflow_step_clicked(step)`:

1. **Nube** — cargar/gestionar la nube, configurar grid de tiles (`TilePanel`).
2. **Pre-clasificar** — `GeoPanel`: calcular MDT (terreno), clasificar por AGL
   (altura sobre el suelo) en capas configurables, filtro CSF (Cloth Simulation,
   requiere paquete opcional `CSF`), motor de reglas (`RulesEngine`) con plantillas.
3. **Etiquetar** — `ClassPanel` + `ToolPanel` + herramientas de anotación 3D.
4. **Exportar** — abre `ExportDialog` (modal, no cambia el panel izquierdo).
5. **Entrenar** — `TrainingPanel` ocupa toda la ventana (entrena RandLA-Net/
   PointNet++/KPConv sobre el dataset exportado).
6. **Inferir** — `InferPanel` ocupa toda la ventana (aplica un modelo entrenado
   sobre la nube activa y vuelca las predicciones como labels).

El estado visual de la barra (`_set_workflow_step`) marca pasos completados/activos/
pendientes con colores (verde=hecho, ámbar=activo, gris=pendiente). Los pasos 1-3
solo cambian `_left_stack`; los pasos 4/5/6 tienen comportamiento especial.

## 5. Carga de nubes — pipeline de datos (`core/pointcloud.py`, `core/heavy_cloud.py`)

Formatos soportados: `.las .laz .e57 .ply .pcd .xyz .txt .csv .asc .pts .npy .ga3d_bin`.

- **`PointCloud`**: contenedor simple — `xyz` (puede ser `np.memmap`), `intensity`,
  `rgb`, `classification`, `return_num`, `offset` (traslación al centro para evitar
  floats grandes UTM), `crs`, `bounds`, `octree`.
- **`CloudPipeline(QThread)`**: hilo de carga. Señales: `progress`, `cloud_ready`
  (emite pc con LOD parcial ~30% del octree listo, para mostrar algo rápido),
  `octree_ready` (LOD completo), `heavy_needed` (nube "pesada", ver abajo), `error`.
- **Out-of-core / mmap**: si la nube es muy grande vs RAM disponible
  (`OUT_OF_CORE_THRESHOLD = 500M pts`, o >40% RAM), xyz se escribe a un cache
  `.f32` en `~/.cache/geoannotate3d/` y se abre como `np.memmap`. Igual para
  atributos (intensity/rgb/classification) vía `_make_attr_mmap`.
- **Heavy cloud (`core/heavy_cloud.py`)**: por encima de `HEAVY_THRESHOLD` puntos,
  se ofrece convertir UNA VEZ a un formato binario propio `.ga3d_bin` (análogo al
  `.bin` de Leica) junto al archivo original — aperturas posteriores son instantáneas.
  `MainWindow._on_heavy_needed` orquesta esta decisión (diálogos Sí/No,
  `HeavyCloudConverter(QThread)`, `load_ga3d_bin`).
- **LAS/LAZ**: streaming por chunks de 2M pts (`laspy.chunk_iterator`), detecta
  backend lazrs/laszip, RGB 16-bit→8-bit, normaliza intensidad.
- **E57**: usa `pye57`, soporta cartesiano y esférico, concatena multi-scan.

Tras cargar: `MainWindow._on_cloud_ready` → inicializa/repara `project.labels`,
conecta `LabelStore` con autosave (`.geoa3d_autosave` cada 5 min), detecta si la
nube ya trae un campo `classification` con clases válidas y ofrece importarlo
mapeándolo al schema del proyecto (`_check_and_import_classification`, diálogo con
auto-match por keywords ASPRS).

## 6. Proyectos `.geoa3d` (`core/project.py`)

ZIP con `meta.json` (metadata, CRS, offset, schema de clases, stats, transform del
grid de tiles) + `labels.npy` (uint8 por punto). `Project.save/load`, hash MD5 del
archivo fuente para detectar si cambió (`verify_source`). `SemanticClass(id, name,
color)`. Schema por defecto con 13 clases (geomática urbana + agricultura),
id 0 = "sin etiquetar" (#444444).

`.geoa3d_autosave` — autosave periódico solo de labels (no todo el proyecto),
path derivado del archivo fuente.

## 7. Octree / LOD (`core/octree.py`, `render/lod_worker.py`, `core/_fastcore.c`)

- Construye niveles de LOD precomputados en C (`build_lod_levels_c`) al cargar —
  O(1) para `get_coarse_view()`. Límite duro de 150M pts para evitar crash de GPU
  (`_OctreeBuilder` en `main_window.py`, y en `CloudPipeline`).
- BFS octree tradicional para refinamiento por frustum (`iter_refinement`,
  `_build_octree_bfs`).
- `cKDTree` de scipy para `sphere_query_kdtree` (usado por BrushTool/SphereSelect) —
  400-4000x más rápido que fuerza bruta.
- Extensión C nativa opcional (`core/_fastcore.c`, compilada con
  `core/build_extension.py` via gcc `-O3 -march=native -ffast-math`): LUT de
  anotación, voxel subsample, LOD, color por elevación, sphere query, frustum cull.
  Con fallback numpy puro en `core/_fast.py` si no compila (p.ej. sin gcc en Windows).

## 8. Render 3D (`render/canvas.py`, 1836 líneas) — el corazón visual

`AnnotationCanvas(QWidget)` embebe un `vtkRenderWindow` (VTK 9.2+, OpenGL). Usa un
**buffer GPU persistente** (`vtkPolyData` no se reconstruye si N no cambia — solo
sube colores, 4 bytes/pt). Interactor style custom `_AnnotationStyle` (subclase de
`vtkInteractorStyleTrackballCamera`) enruta mouse/teclado a la herramienta activa.

Conceptos clave:
- **Modo overview vs modo tile** (`is_tile_mode`): la nube completa se ve "sparse"
  a través de tiles (grid), o se puede "entrar" a un tile individual para anotar
  con detalle completo (`enter_tile_mode`/`exit_tile_mode`). Grid de tiles editable
  con drag (mover/rotar, `set_grid_edit_mode`, `_TileGridOverlay`).
- **Modos de color**: Anotación (LUT por clase), Elevación, RGB, Intensidad,
  Clasificación (LAS), Color único (`set_color_mode`).
- **Modo vuelo (fly mode, tecla F)**: cámara WASD tipo FPS (`toggle_fly_mode`,
  `_fly_tick`, `_fly_look`).
- **Overlays**: raster (ortomosaicos) y vectorial (shp/geojson/etc.) vía
  `core/overlay_manager.py`, actores VTK adicionales.
- LOD dinámico: `_lod_tick`/`_req_worker` piden refinamiento incremental en un
  worker (`render/lod_worker.py`) según frustum y presupuesto de puntos.
- Cursor de pincel proyectado en 3D (`_update_brush_cursor`), snapping a punto más
  cercano (`_snap_to_point`), picking con proyección MVP cacheada (`_get_mvp`).

## 9. Herramientas de anotación (`annotation/tools.py`, 1221 líneas)

Todas heredan de `BaseTool`. Registro global `ALL_TOOLS`, `TOOL_BY_NAME`,
`TOOL_BY_KEY`. Atributos comunes: `label_store`, `active_class_id`, `erase_mode`
(pinta clase 0), `only_unlabeled` (no sobrescribe puntos ya etiquetados).

| Clase | Nombre UI | Tecla | Resumen |
|---|---|---|---|
| `BrushTool` | Pincel | B | Click+drag, pinta esfera 3D vía cKDTree; radio, densidad de trazo, grosor de disco (5%=disco fino, 100%=esfera), sigue la normal de superficie (`_estimate_normal`) con Ctrl+drag |
| `PolygonTool` | Polígono | L | Lazo en pantalla, Ctrl+clic añade vértice, Enter aplica (`_ray_cast_poly`), selecciona TODO a cualquier profundidad |
| `BoxSelectTool` | Caja | X | Ctrl+drag rectángulo pantalla → columna 3D completa (AABB) |
| `SphereSelectTool` | Esfera | R (¿o "Radio"?) | Ctrl+clic, radio configurable |
| `SliceTool` | Corte Z | P | Ctrl+clic define plano base, drag altura; modos: entre planos / encima / debajo (tecla T cicla) |
| `FloodFillTool` | Relleno | — | Region growing por BFS desde semilla (Ctrl+clic), paso, tolerancia Z, máx. puntos |
| `PickTool` | Pick | I | Ctrl+clic inspecciona: clase, coords UTM E/N/Z |
| `MeasureTool` | Medir | M | Clic A + clic B → distancia 3D/2D/dZ |
| `DiscTool` | — | — | Variante de disco alineado a normal local |

Nota: el README documenta teclas B/L/X/P/R/I/M pero el código real tiene más
herramientas (`FloodFillTool`/Relleno, `DiscTool`, Region Growing mencionado en
tool_panel.py como opción de contexto) — **el README está desactualizado**, uno de
los primeros candidatos de limpieza.

`LabelStore` (`annotation/label_store.py`) — stack de undo/redo sobre el array de
labels, autosave periódico, señales `stats_changed`/`labels_changed`.

## 10. Paneles de la izquierda (uno por paso del workflow)

- **`ui/tile_panel.py`** (643 líneas) — `TileGridWidget` (grid visual 2D de tiles
  con progreso por clase coloreado) + `TilePanel`: tamaño de tile, transform
  (offset X/Y, rotación), modo de vista (Sparse/Full/Volar), hover/click de tile,
  botones Mover/Rotar (drag en el canvas 3D).
- **`ui/geo_panel.py`** (419 líneas) — `_AGLBar` (barra de gradiente terreno→copa
  con selección de rango) + `GeoPanel`: Calcular MDT, clasificación AGL automática
  por capas de altura, CSF (Cloth Simulation Filter, opcional), plantillas de
  reglas (`RulesEngine`), selección manual por rango AGL.
- **`ui/class_panel.py`** (507 líneas) — chip de clase activa, lista de clases con
  atajos 1-9, doble-clic edita color/nombre, barra de cobertura total, botón
  "Gestionar clases" → `ui/class_manager.py::ClassManagerDialog`.
- **`ui/overlay_panel.py`** (310 líneas) — gestor de capas raster/vectoriales
  (carga, visibilidad, opacidad, z-offset), vista superior rápida.

## 11. Panel derecho — `ui/tool_panel.py` (846 líneas)

Grid 4×2 de botones de herramienta (iconos dibujados a mano con `QPainter`, sin
assets externos — `_draw_tool_icon`). Sección "Modo de anotación" (borrar [E],
solo sin etiquetar). Sección de contexto dinámico por herramienta activa
(`_ContextSection.update_for_tool`, reconstruye el layout completo cada vez que
cambia de herramienta). Sección Visualización (tamaño de punto, ver sin etiquetar,
grilla, combo de modo de color). Sección "Capas superpuestas" duplicada (también
existe en `OverlayPanel` — **posible redundancia a resolver en el rediseño**).

## 12. Exportación de dataset (`annotation/exporter.py`, 1178 líneas)

`ExportDialog` (modal) → elegir arquitectura (RandLA-Net / PointNet++ / KPConv,
cada una con su formato implícito) → `ExportWorker(QThread)`:
- Split espacial en bloques configurable (`compute_spatial_split`, default 50×50m)
  para evitar data leakage geográfico train/val/test.
- Pesos de clase por frecuencia inversa normalizada.
- Genera no solo los datos sino **scripts de entrenamiento standalone** por
  arquitectura (`_write_randlanet_train_py`, `_write_pointnetpp_train_py`,
  `_write_kpconv_train_py`) + `dataset.py` + `README.md` + `requirements.txt` por
  arquitectura — es decir, el export es un mini-repo de entrenamiento autocontenido.
- También puede exportar un `.las` reclasificado (`_export_classified_las`).

## 13. Entrenamiento e inferencia embebidos

- **`ui/training_panel.py`** (1126 líneas) — panel de pantalla completa (paso 5):
  `_LossChart` (gráfico de pérdida custom con QPainter), `TrainingWorker(QThread)`
  que probablemente invoca los scripts generados por el exporter o entrena in-process.
  Señal `training_finished(model_path)`.
- **`ui/infer_panel.py`** (430 líneas) + **`infer.py`** (716 líneas, standalone/CLI
  también) — `InferWorker(QThread)`, carga un checkpoint, reconstruye la arquitectura
  (funciones `knn`, `farthest_point_sample`, `ball_query`, etc. reimplementadas en
  numpy/torch puro dentro de `infer.py` — sin depender de los repos originales de
  RandLA-Net/PointNet++/KPConv), corre inferencia por batches, guarda resultado.
  `MainWindow._on_inference_done` vuelca las predicciones al `LabelStore` como si
  fueran anotaciones manuales.

## 14. Terreno / geoprocesamiento

- **`core/terrain.py`** — `TerrainModel`: MDT (modelo digital de terreno) por
  percentil de altura en grid, `compute_auto`, `agl()` (altura sobre el suelo por
  punto), `agl_range_mask`.
- **`core/rules_engine.py`** — `Rule` + `RulesEngine`: reglas simples de
  clasificación automática (condiciones sobre AGL/intensidad/RGB/etc., aplicables
  en batch desde `GeoPanel`).
- **CSF (Cloth Simulation Filter)** — dependencia opcional (`pip install
  cloth-simulation-filter`), detecta suelo simulando una tela cayendo sobre la nube.

## 15. Overlays geoespaciales (`core/overlay_manager.py`, 462 líneas)

`OverlayLayer` (raster u vector), `load_vector` (shp/geojson/gpkg/kml/dxf vía
geopandas probablemente), `load_raster` (tif/png/jpg georreferenciado), actores VTK
añadidos al canvas. `ui/layer_properties.py` (205 líneas) — diálogo flotante de
propiedades por capa (color, grosor/estilo de línea, opacidad, offset Z, brillo).

## 16. Tiles / grid espacial (`core/tile_manager.py`, 565 líneas)

`TileInfo` (col, row, bounds, n_points, progreso de etiquetado), `TileManager`
(construye grid sobre la nube con tamaño/offset/rotación configurables, índice
espacial para extracción rápida — `build_tile_index`, `get_tile_indices_fast`),
`TileExtractionWorker(QThread)` (extracción lenta cuando el índice aún no existe).
Sirve para trabajar la nube "por parcelas" en vez de cargarla completa en el canvas.

## 17. Utilidades

- **`utils/geo.py`** — conversión offset↔UTM absoluto, bbox geo, formato de
  coordenadas para UI, estimación de zona UTM (parcialmente sin implementar).
- **`utils/spatial.py`** — frustum culling, sphere query (con fallback numpy si
  no hay extensión C), decimación por densidad screen-space, ray casting.

## 18. Otros archivos de soporte

- `ui/splash.py` — pantalla de carga con progreso.
- `ui/welcome_dialog.py` — onboarding de 6 pasos, checkbox "no volver a mostrar"
  (persistido en `.geoannotate_prefs.json` junto al ejecutable).
- `ui/status_bar.py` — fps, info de render, stats por clase, tiempo de sesión.
- `ui/class_manager.py` — CRUD completo del schema de clases (añadir/eliminar/
  reordenar, importar/exportar schema probablemente).
- `build_exe.bat`, `build_installer_script.py` — empaquetado a `.exe`/instalador
  (probablemente PyInstaller + Inno Setup o similar — no leído en detalle aún).
- `.geoannotate_recent.json` — lista de archivos recientes (menú "Abrir reciente").

## 19. Puntos débiles / candidatos a mejorar (para cuando pida cambios)

- **README desactualizado** respecto al código real (herramientas, versión).
- **Estilo QSS totalmente disperso e inline** en cada widget en vez de centralizado
  en `ui/styles.py` — dificulta un rediseño de tema consistente.
- **Fuente monoespaciada global** — decisión estética discutible para una app pro.
- Posible **redundancia del panel de capas** (existe tanto en `OverlayPanel` como
  duplicado dentro de `ToolPanel`).
- `main_window.py` es un god-object de 2300 líneas con muchísima lógica de UI y de
  dominio mezclada (candidato a refactor si se piden cambios grandes).
- Manejo de errores muy defensivo con `except Exception: pass` disperso por todo
  el código — puede ocultar bugs reales al depurar.
- No hay tests (`grep` no encontró carpeta `tests/`) ni CI — a considerar antes/
  durante la subida a GitHub.
- No es repo git aún — antes de subir a GitHub: `git init`, `.gitignore` (excluir
  `__pycache__/`, `*.pyd` compilado si aplica, `.geoannotate_recent.json`,
  `.geoannotate_prefs.json`, cachés `.f32`/`.ga3d_bin`, `venv/`), decidir si el
  binario `.pyd` compilado (`core/_fastcore.cp312-win_amd64.pyd`) se versiona o se
  compila en CI/instalación.

## 20. Cambios de rendimiento aplicados (2026-09-04)

Pedido del usuario: nubes densas se congelaban al usar herramientas de dibujo
(Pincel sobre todo). Diagnóstico + fix, sin tocar el pipeline de carga aún.

**3 bugs de rendimiento reales encontrados y corregidos:**

1. **`BrushTool._paint_at` ([annotation/tools.py](annotation/tools.py)) volvía a
   copiar el tile activo completo (`c.pc.xyz[c._tile_indices]`, hasta 20-30M pts)
   en CADA evento de mouse-move**, cuando `c._cur_xyz` ya contenía exactamente
   esos mismos datos (confirmado: `_TileLoader` en `render/canvas.py` carga el
   tile sin decimar y `_tile_indices is _cur_idx`). → Eliminado; ahora se usa
   directamente lo ya cargado.
2. **La búsqueda de puntos dentro del radio del pincel era fuerza bruta O(N)**
   sobre el tile entero, ignorando el índice espacial en C que YA existe
   (`core/_fastcore.c`: `grid_build`/`grid_sphere_query`, construido una vez
   por nube en `pc.octree`, ~cientos de puntos examinados en vez de millones).
   Añadido `BaseTool._octree_sphere_query()` (usa `pc.octree` vía
   `sphere_query_octree`, con fallback al escaneo directo sobre `_cur_xyz`
   solo mientras el octree aún no terminó de construirse tras cargar la nube).
   Aplicado a `BrushTool`, `SphereSelectTool` y `DiscTool` (mismo patrón en
   los tres).
3. **`AnnotationCanvas.refresh_colors()` en [render/canvas.py](render/canvas.py)
   forzaba SIEMPRE el camino lento (recompute + reupload completo del tile) en
   modo tile** — literalmente decía "tile mode: slow path siempre" en el
   comentario — en vez de la ruta rápida en C (`FC.refresh_colors_partial`,
   actualiza solo los puntos afectados) que YA se usaba en modo overview.
   Unificado: ahora ambos modos intentan primero la vía rápida.

Efecto esperado: cada "dab" del pincel pasa de escanear+recomputar millones
de puntos en Python/numpy a examinar unos cientos y actualizar solo esos
píxeles de color en C — debería eliminar el freeze al pintar sobre tiles
densos. **Pendiente**: probar en una nube densa real (no pude ejecutar la app
en este entorno) y medir con el contador de FPS ya integrado en la status bar.

**Caveat conocido (pre-existente, no corregido, fuera de alcance):** para
nubes >150M pts cargadas por el camino "heavy"/`.ga3d_bin`
(`MainWindow._OctreeBuilder`), el octree se construye sobre un submuestreo
estrided de `pc.xyz`, así que sus índices NO corresponden 1:1 a `pc.xyz`. Esto
ya afectaba a `SphereSelectTool` antes de este cambio (mismo
`sphere_query_octree`); ahora también aplica a `BrushTool`/`DiscTool`. Si se
reporta que pintar en nubes extremadamente grandes (>150M) selecciona los
puntos "equivocados", el fix real es hacer que `_OctreeBuilder` devuelva
también el mapeo `step`/subsample para traducir índices, o evitar el cap.

**Pendiente de analizar (pedido pero no llegué a tocar código todavía):**
- Velocidad de carga inicial (`core/pointcloud.py`, `CloudPipeline`): ya usa
  streaming por chunks + mmap out-of-core + LOD en C con emisión progresiva
  (`cloud_ready` al 30%). Es una arquitectura ya bastante optimizada; no
  encontré un "quick win" obvio sin perfilar en una máquina real. Candidatos
  a explorar si se pide profundizar: paralelizar la decodificación de chunks
  LAZ (hoy es secuencial), permitir configurar `chunk_size` según SSD/HDD,
  revisar si `_available_ram_gb()`/psutil se llama redundantemente.
- Navegación (rotar/pan la cámara) ya tiene manejo adaptativo de budget por
  FPS (`_fps_tick`, `_lod_tick`, cancelación del worker durante `_interacting`)
  — arquitectura sólida. No apliqué cambios aquí todavía; si sigue sintiéndose
  lento habría que perfilar con la nube real del usuario (constantes ajustables:
  `FPS_LOW`, `FPS_HIGH`, `BUDGET_FLOOR`, `SPARSE_CAP` — no las he localizado
  con número de línea exacto todavía, están cerca del inicio de canvas.py).

## 20bis. Fixes de seguimiento (mismo día) — regresión + lentitud restante + tiles densos

El usuario probó los cambios de la sección 20 en una nube real de 80M pts
(tile de 30M) y reportó: (a) el Pincel dejó de seleccionar nada, (b) seguía
algo lento, (c) pidió explícitamente algo tipo LOD progresivo para tiles densos
(estilo Potree/LAStools — "cargar millones sin problema").

**(a) Regresión del Pincel — bug real encontrado en [utils/spatial.py](utils/spatial.py)
`sphere_query_octree()`** (preexistente, no lo introduje yo, pero mi cambio de la
sección 20 lo puso en el camino PRINCIPAL en vez de ser un fallback raro):
la función comprobaba `_kdtree_ready` para decidir si usar el índice rápido,
pero `Octree.build()` apaga `_kdtree_ready` a propósito cuando prefiere el
grid en C (nubes >30M pts). El siguiente `if hasattr(octree,
'_sphere_query_octree_bfs')` es SIEMPRE verdadero (el método existe aunque no
haya BFS construido — BFS se salta para nubes >=50M, ver
`BFS_SKIP_THRESHOLD`), así que la función nunca llegaba ni al grid ni a la
fuerza bruta real: para cualquier nube >=30M pts (grid preferido) Y >=50M pts
(sin BFS) devolvía SIEMPRE vacío. Exactamente el caso del usuario (80M pts).
Corregido: ahora prueba en orden grid (C) → cKDTree → BFS (solo si
`root is not None`) → fuerza bruta genuina.

**(b) Lentitud restante — doble escaneo completo de labels en cada trazo**:
`LabelStore.annotate()` emitía `stats_changed` en cada llamada, y
`MainWindow._on_stats_changed` hace `per_class_counts()` (`FC.per_class_counts`,
O(N) en C) Y `n_labeled` (`np.count_nonzero`, O(N)) sobre el array de labels
COMPLETO (80M elementos) — dos escaneos síncronos por cada dab del pincel.
Corregido con debounce (mismo patrón que ya usaba `refresh_colors`):
`LabelStore._request_stats_update()` coalesce llamadas seguidas en un QTimer
de 150ms (`_stats_pending`/`_stats_timer`), así que mientras se pinta rápido
se hacen como mucho ~6-7 actualizaciones de stats por segundo en vez de una
por evento de mouse. `undo()`/`redo()` se dejaron sin debounce (frecuencia
baja, un keypress a la vez).

**(c) Tiles densos siempre a densidad máxima → vista previa progresiva**:
`_TileLoader` (en `render/canvas.py`) esperaba a tener el tile COMPLETO leído
+ coloreado antes de mostrar nada — para un tile de 20-30M+ pts eso son
varios segundos de pantalla congelada al entrar. Añadido: si
`n_tile > TILE_COARSE_THRESHOLD` (3M), se emite primero `coarse_ready` con
una vista decimada a `TILE_COARSE_TARGET_PTS` (~1.2M pts, rápida de leer y
subir a GPU) para que la navegación empiece casi al instante, y el tile
completo llega después por `ready` (como antes) y reemplaza la vista previa.
Nueva señal `_TileLoader.coarse_ready` + handler
`AnnotationCanvas._on_tile_coarse_ready` (idéntico a `_on_tile_loaded` pero
deja `_render_state="TILE_REFINING"` en vez de `"DONE"`). Importante:
`_tile_indices` (el conjunto COMPLETO de índices del tile) nunca se toca —
solo `_cur_idx`/`_cur_xyz` (lo que está renderizado ahora mismo) pasan por la
vista decimada transitoriamente. Esto es seguro porque, tras los cambios de
la sección 20, Pincel/Esfera/Disco ya NO dependen de `_cur_xyz`/`_cur_idx`
para su precisión (usan `pc.octree` global) — pueden anotar a resolución
completa aunque en pantalla se vea la vista previa decimada. Las herramientas
que SÍ dependen de lo visible (`PolygonTool`, `BoxSelectTool`, `SliceTool`,
`_project_visible()`) actuarán sobre el subconjunto decimado solo durante esa
ventana transitoria (típicamente <1-2s) — igual que en software profesional
tipo Potree, donde primero ves/trabajas sobre una vista gruesa y se refina.

**Esto es un LOD de UN SOLO escalón (grueso → completo), no continuo como el
sistema de la vista overview** (`LODWorker`/`Octree.iter_lod_progression`,
que crece el budget progresivamente según FPS). Si un tile sigue sintiéndose
pesado incluso con la vista previa (ej. tiles de 100M+ pts, o GPU limitada),
el siguiente paso natural sería portar ese mismo sistema de refinamiento
continuo + frustum culling dentro de tile mode en vez de un solo salto
grueso→fino. No lo hice todavía por alcance/tiempo — decírmelo si hace falta.

## 20ter. Arnés de pruebas headless + 2 bugs nuevos corregidos (mismo día, ronda 3)

**Arnés de pruebas — `tests/verify_fixes.py`.** El usuario pidió poder
confiar en que un cambio ya fue comprobado antes de entregarlo. Confirmé que
este entorno SÍ tiene PyQt5 5.15 + VTK 9.6 instalados y que Qt corre en modo
`QT_QPA_PLATFORM=offscreen` sin problema — PERO `AnnotationCanvas` (que crea
un `QVTKRenderWindowInteractor` con ventana OpenGL nativa Win32) hace
**SEGFAULT** offscreen aquí (`vtkWin32OpenGLRenderWindow: failed to get valid
pixel format` — no hay GPU/driver real en este sandbox). Por tanto:
- SÍ puedo ejecutar y verificar automáticamente: `LabelStore`, `TileManager`,
  `sphere_query_octree`/`Octree`, `compute_colors_u8`, `ToolPanel` (widgets
  Qt normales, sin VTK) — es decir, toda la lógica de datos.
- NO puedo verificar el render 3D real (colores en pantalla, FPS real,
  comportamiento del pincel visualmente) — eso requiere que el usuario lo
  confirme en su máquina con GPU. Lo dejo explícito en cada entrega para no
  sobre-prometer.

Correr el arnés: `QT_QPA_PLATFORM=offscreen python tests/verify_fixes.py`
(en Windows con consola cp1252 puede hacer falta
`PYTHONIOENCODING=utf-8` también, por los acentos/símbolos de la salida).
Son 5 tests / 17 verificaciones, todas en verde tras los fixes de abajo.
Extenderlo con cada bug nuevo que se corrija de aquí en adelante.

**Bug nuevo #1 — grilla movida abre el tile de ANTES de moverla
([core/tile_manager.py](core/tile_manager.py) `_tile_index_path()`).**
Causa raíz: el nombre del archivo de caché del índice punto→tile en disco se
calculaba a partir de `project.tile_rotation_deg` / `project.tile_origin_x`
/ `project.tile_origin_y` — campos que existen en `Project` pero que
**ningún lugar del código actualiza jamás** (quedan en 0.0 para siempre; los
campos que SÍ se mantienen al día son `grid_offset_x/grid_offset_y/
grid_rotation`, unos completamente distintos). Resultado: mover la grilla
nunca cambiaba el nombre del archivo de caché, así que
`TileManager.build_tile_index()` recargaba el índice viejo (calculado con la
grilla en su posición ANTERIOR) para CUALQUIER posición nueva de la grilla.
Corregido: `_tile_index_path()` ahora usa `self.tile_size_m` /
`self.rotation_deg` / `self.offset_x` / `self.offset_y` — los parámetros
VIVOS de la instancia de `TileManager`, que sí reflejan la grilla actual.
Verificado en `tests/verify_fixes.py::test_tile_manager_cache_path_changes_with_transform`
(dos `TileManager` con transforms distintos → rutas de caché distintas →
cada uno construye su propio índice → los puntos del tile caen dentro de
SUS bounds correctos, no los del otro transform).

**Bug nuevo #2 — pintar con el pincel no cambiaba el color en la nube 3D**
(aunque el panel de clases/progreso sí se actualizaba). Causa raíz: el combo
de modo de color en `ToolPanel` ([ui/tool_panel.py](ui/tool_panel.py))
nunca se sincronizaba cuando `main_window.py` elegía un modo de color
automáticamente (al cargar una nube: RGB si tiene color, si no Elevación;
o al importar clasificación / restaurar anotaciones / aplicar inferencia)
— el combo se quedaba SIEMPRE en "Anotación" (primer ítem por defecto),
aunque el canvas ya estuviera renderizando en modo RGB/Elevación de verdad.
El usuario veía "Anotación" en la UI, pintaba, los labels SÍ se escribían
(por eso el panel de clases se actualizaba), pero el color 3D no cambiaba
porque el modo REAL del canvas no era "Anotación" — por diseño, cada modo
de color es exclusivo (RGB no mezcla con colores de anotación). Esto
también explica la queja de "a veces se ve como si estuviera en modo
Anotación pero se ve RGB": eran el mismo bug, dos síntomas.
Corregido: nuevo método `ToolPanel.set_color_mode(mode)` (sincroniza el
combo con `blockSignals` para no re-emitir `color_mode_changed` y evitar
eco) llamado desde los 4 sitios de `main_window.py` donde el modo se
cambiaba automáticamente (`_on_cloud_ready`, `_activate_annotation_color_if_labeled`,
`_on_inference_done`, `_check_and_import_classification`). Verificado en
`tests/verify_fixes.py::test_tool_panel_color_mode_sync`.

**Nota sobre el modo inicial al cargar una nube:** ya estaba bien
implementado en `_on_cloud_ready` (RGB si `pc.rgb is not None`, si no
Elevación, o Anotación si ya hay etiquetas) — el problema NUNCA fue la
lógica de selección, sino que el combo de la UI no reflejaba lo elegido.
No hacía falta tocar esa lógica.

## 22. Rediseño de UI implementado en el código (2026-09-05)

Tras aprobar la Opción B (riel de navegación) con paleta clara clásica en el
canvas de diseño, se implementó de verdad en el código real (no solo mockup).

**Archivos nuevos:**
- `ui/theme.py` — ÚNICA fuente de verdad de la paleta (antes ~150 hex sueltos
  repetidos por todo `ui/*.py`). Cualquier cambio de tema futuro va aquí.
- `ui/icons.py` — carga y recolorea SVG reales (sustituye `currentColor` por
  el hex pedido antes de pasarlo a `QSvgRenderer`; cachea por
  (nombre, color, tamaño)). `pixmap()`/`icon()`.
- `ui/icons/*.svg` (41 archivos) — iconos reales de **Bootstrap Icons**
  (MIT, ver `ui/icons/LICENSE.txt`), descargados de jsdelivr, NO dibujados
  a mano. Reemplazan `_draw_tool_icon` en `tool_panel.py` (que además tenía
  un bug: Disco y Region Growing se quedaban sin icono, ya corregido).
- `tests/verify_fixes.py` — extendido con `test_redesign_icons_and_rail`.

**`ui/main_window.py` — chrome reestructurado:**
- `_HeaderBar` reescrita: paleta clara, icono real de proyecto (folder2).
- NUEVO `_StepBreadcrumb`: franja fina "Paso 3 de 6 — Etiquetar — ..." bajo
  la cabecera — dónde estás, siempre visible, sin depender del riel.
- NUEVO `_StepRail` (reemplaza la barra horizontal de 6 pasos + botón
  "⬟ Capas"): riel vertical de 176px, ICONO + TEXTO por paso (no solo
  icono — más claro, menos "adivina qué es esto"), estados
  pending/active/done, siempre visible incluso en los pasos 5/6
  (Entrenar/Inferir, que ocupan toda la pantalla) porque vive FUERA del
  `_body_stack`, no dentro.
- "Capas de referencia" es una fila aparte en el riel, separada por un
  divisor — visualmente ya no se confunde con un séptimo paso.
- `_set_workflow_step`/`_on_layers_btn_clicked` reescritos para el riel
  nuevo; toda la lógica de negocio (`_on_workflow_step_clicked`, cambio de
  `_left_stack`/`_body_stack`) queda intacta, solo cambió el widget.

**Duplicación del panel de capas — ELIMINADA:**
`ToolPanel` tenía su propia mini-sección "CAPAS SUPERPUESTAS" (añadir
vector/raster + lista) que duplicaba a `OverlayPanel` casi al 100%.
Se quitó de `ToolPanel` (señales `layer_add_requested/layer_toggle_requested/
layer_remove_requested` y los métodos asociados) y las conexiones
correspondientes en `main_window.py`. Ahora la gestión de capas vive
SOLO en `OverlayPanel`, accesible desde "Capas de referencia" en el riel
(mismo mecanismo de antes: `_left_stack.setCurrentIndex(3)`).

**`ui/tool_panel.py` reescrito completo:** iconos reales en el grid de 9
herramientas, paleta clara, botón "Modo borrar" con icono real de goma en
vez de "⌫". Se decidió NO usar pestañas (Herramientas/Vista/Capas) como en
el mockup — al quitar la sección de capas duplicada, el panel ya no es tan
largo como para justificar pestañas; un solo panel vertical es más simple
y más "clásico, clara de entender".

**`ui/class_panel.py`** — retocado a mano (botón "Gestionar clases" con
icono real de lápiz en vez de "✎", y corregido un borde casi invisible).

**Resto de paneles (`class_manager.py`, `export_dialog.py`, `geo_panel.py`,
`infer_panel.py`, `training_panel.py`, `tile_panel.py`, `overlay_panel.py`,
`welcome_dialog.py`, `layer_properties.py`, `status_bar.py`) — recoloreados
por script, NO reescritos a mano.** Con ~150 hex distintos repartidos en
esos archivos, reescribir cada uno a mano no era viable en una sesión.
Proceso (documentado para poder repetirlo o auditarlo):
  1. Clasificar cada hex único por HSL (tono/saturación/luminosidad) a uno
     de ~17 tokens semánticos de `ui/theme.py` (bg/surface/texto/acento/
     ok/warn/...). Los 6 colores del gradiente AGL en `geo_panel.py`
     (`_AGLBar`, terreno→dosel) se protegieron explícitamente — son datos,
     no tema, igual que los colores de clase.
  2. Sustitución automática en los 12 archivos (616 reemplazos).
  3. **Pasada de corrección de contraste** (encontró bugs reales del paso
     1): 98 casos de texto que quedó mapeado a un tono casi-blanco
     (ilegible sobre fondo blanco) por errores de umbral en los buckets
     de luminosidad — corregidos a `text_dim`. Luego una segunda pasada
     corrigió bordes casi invisibles (`border:` mapeado a blanco/casi-blanco).
  4. Verificado: sintaxis válida + los 10 paneles/diálogos se instancian
     sin excepción en modo headless (`QT_QPA_PLATFORM=offscreen`).

**Qué NO se verificó (limitación ya conocida, ver sección 20ter):** el
render visual real (¿se ve bien de verdad, hay algún color feo que se me
escapó de la pasada de contraste, se ve bien con la nube real cargada?)
sigue sin poder probarse aquí — VTK hace SEGFAULT offscreen en este
entorno. Además, la pasada de contraste solo puede detectar los dos
patrones de bug que sabía buscar (texto ilegible, bordes invisibles) — es
posible que queden combinaciones de color sueltas que no se vean geniales
sin ser técnicamente "ilegibles". **Pedirle al usuario que abra la app y
reporte cualquier color que se vea raro es el siguiente paso obligatorio.**

## 23. Rediseño de PANELES (contenido, no solo chrome) — en progreso (2026-09-05)

El usuario pidió, después del rediseño de navegación (sección 22), que el
CONTENIDO de cada panel también se rediseñe (no solo se recoloree) —
estilo barra de herramientas de SIG (QGIS, Cyclone 3DR). También pidió
que la verificación sea con CAPTURAS DE PANTALLA reales, no un link a un
mockup HTML.

**Hallazgo importante de infraestructura — capturas offscreen sin fuentes:**
`QT_QPA_PLATFORM=offscreen` en este entorno no trae NINGUNA fuente
registrada (`QFontDatabase().families()` devuelve `[]`), así que todo
texto salía en blanco en las primeras capturas (parecía un bug de la UI,
no lo era). Solución: `QFontDatabase.addApplicationFont()` con archivos
reales de `C:/Windows/Fonts/` (segoeui.ttf, arial.ttf, consola.ttf) antes
de capturar. Quedó en
`.../scratchpad/screenshot_helper.py` (session-local, no es parte del
repo) — reusar este patrón para cualquier captura futura.

**Bug real de Qt encontrado durante la primera captura (no relacionado con
fuentes):** un `widget.setStyleSheet("background:X;border:Y;border-radius:Z;")`
SIN selector (una lista de declaraciones "pelada") se filtra a TODOS los
widgets hijos que no overrideen esas propiedades — es un gotcha documentado
de Qt Style Sheets, no until ahora había mordido porque el código viejo
dibujaba todo con QPainter manual (sin hijos reales) en vez de QLabels
hijos. Fix aplicado: cualquier contenedor "tarjeta" con hijos reales usa
ahora `setObjectName("card")` + `QFrame#card{...}` (selector con ID) en
vez de un stylesheet pelado — ver `ToolPanel._card()` en tool_panel.py,
patrón a seguir en cualquier panel nuevo.

**Completado y verificado con captura real:**
- **`ui/tool_panel.py`** — el selector de herramientas pasó de una
  cuadrícula 4×2 de tarjetas grandes (icono+nombre+atajo apilados, 56px
  de alto cada una) a una BARRA DE HERRAMIENTAS real de iconos de 34×34
  (nombre+atajo ahora en el tooltip), agrupada en una franja gris con
  fondo propio — exactamente el patrón QGIS/Cyclone 3DR pedido. Las
  secciones de abajo (Modo de anotación / Opciones / Visualización) ahora
  son tarjetas con borde+icono de encabezado en vez de bloques separados
  por líneas divisorias sueltas.
- **`ui/tile_panel.py`** — causa real de por qué "seguía igual": el
  `TileGridWidget` (el elemento MÁS visible del panel) pintaba con
  `QColor(r,g,b)` en enteros hardcodeados del tema oscuro — la sustitución
  automática de la sesión anterior solo buscaba strings `"#hex"`, así que
  nunca lo tocó. Recoloreado a la paleta nueva (grid claro, activo=acento,
  completo=verde). Además: los botones de vista/cámara/transform del grid
  (que eran texto+símbolo unicode: "⊤ Cenital", "⇔ Mover grid", "↺ Rotar")
  ahora son botones de icono real agrupados en tarjetas de "barra de
  herramientas", igual que ToolPanel.
- **`ui/splash.py`** — la barra de progreso y el texto de estado (antes
  dorado/ámbar) pasaron a azul acero. **Pendiente de decisión**: la imagen
  de fondo (`ui/splash_bg.png`, arte de marca real, no generado por mí)
  es un wireframe/partículas ámbar sobre negro — visualmente es EXACTO al
  estilo "cyberpunk" que el usuario dijo no querer. No la toqué (es un
  asset de imagen, no CSS/código) — preguntar si se reemplaza o se
  rediseña.

**Pendiente (alcance real de "todos los paneles" es demasiado grande para
una sola sesión sin sacrificar calidad/verificación):**
- `ui/class_panel.py` — reestructurar filas de clases estilo tabla/capas
  de QGIS, barra de herramientas (añadir/editar) en vez del botón
  "Gestionar clases" suelto al final.
- `ui/geo_panel.py` — agrupar MDT/AGL/CSF/Reglas en tarjetas colapsables
  en vez de secciones apiladas.
- `ui/training_panel.py` / `ui/infer_panel.py` — son pantallas completas
  grandes, no alcancé a rediseñarlas esta sesión.
- `ui/overlay_panel.py` / `ui/class_manager.py` / `ui/export_dialog.py` /
  `ui/layer_properties.py` — solo recoloreados (sección 22), no
  reestructurados.

## 24. TERCER cambio de paleta + rediseño de ClassPanel (2026-09-05, misma tarde)

El usuario pidió recolorear TODO desde cero otra vez (ya iba la segunda
vez) y rediseñar estructuralmente cada panel, con referencia a "software
famosos". Antes de adivinar mal una tercera vez, usé `AskUserQuestion` con
4 direcciones concretas (con swatches de hex en la preview) basadas en
software reales: Autodesk/CAD gris neutro, Figma/Linear blanco+índigo,
Blender gris oscuro+naranja, Lightroom/QGIS azul-gris+verde. El usuario
eligió **"Gris neutro + acento cian apagado (Autodesk/CAD clásico)"**.

**Paleta actual (ui/theme.py) — la única fuente de verdad:**
BG `#e8e9eb` (gris medio, nunca blanco puro) · SURFACE `#f4f5f6` ·
ACCENT `#0e7c86` (cian apagado) · TEXT `#2b2d30` · RADIUS bajado a 4/3px
(estética plana, casi sin esquinas redondeadas — a propósito, para que
se sienta "CAD clásico" y no "SaaS moderno").

**Migración de paleta — método (2do OLD_HEX→NEW_HEX 1:1, no HSL):**
A diferencia de la primera migración (café oscuro → blanco, que requirió
clasificar ~150 colores por HSL en buckets, con 98 errores de contraste
encontrados y corregidos), esta migración fue mucho más segura: como ya
existía un set pequeño de ~18 tokens semánticos en `ui/theme.py`, bastó
un diccionario 1:1 (viejo hex de cada token → nuevo hex del mismo rol) —
580 sustituciones en 16 archivos, cero errores de clasificación posibles
porque los ROLES no cambiaron, solo los valores. Verificado con el mismo
detector de "texto casi invisible" de la vez pasada: sin problemas reales
(1 falso positivo, un botón con fondo oscuro que necesita texto claro).

**Bug de entorno encontrado en el camino:** `python` en este shell empezó
a resolver a Python 3.14 (sin numpy/PyQt5) en vez del 3.12 del proyecto —
conviene usar la ruta completa `/c/Program Files/Python312/python.exe` de
ahora en adelante en este entorno si `python` falla con `ModuleNotFoundError`.

**`ui/class_panel.py` — rediseño estructural (antes solo recoloreado):**
Ahora sigue el lenguaje visual del panel de capas de Photoshop/Illustrator:
franjas alternadas fila sí/fila no, barra de acento vertical a la
izquierda en la fila activa (en vez de solo cambiar el fondo), badge tipo
"pill" para la tecla de atajo, conteos en monoespaciada alineados a la
derecha. El botón "Gestionar clases" pasó de ser un botón de texto grande
y suelto al final del panel a un icono pequeño (lápiz) junto al título
de la sección "CLASES" — el mismo patrón de esos paneles famosos, donde
las acciones del panel viven como iconos en su propia barra de
herramientas, no como botones sueltos.

**Todo verificado con capturas de pantalla reales** (`ToolPanel`,
`TilePanel`, `ClassPanel`) vía `grab()` offscreen + regresión completa
(`tests/verify_fixes.py`, 14/14 paneles instancian sin error).

**Pendiente (mismo alcance que sección 23, ahora con la paleta nueva):**
`TrainingPanel`/`InferPanel` (pantallas completas), y decidir qué hacer
con `ui/splash_bg.png` (arte de marca ámbar/negro, estilo "cyberpunk"
que el usuario rechazó — no es CSS, es una imagen, pendiente de
decisión explícita).

## 25. GeoPanel — reescritura estructural (no solo recolor)

`ui/geo_panel.py` tenía dos problemas de recorte que el simple recolor
de la sección 23-24 no tocaba (mismo bug de fondo que QLabel/QPushButton
en otros paneles, aplicado aquí a `QGroupBox`):

1. `QGroupBox` con título de texto plano **no envuelve** — con textos
   largos como "PASO 1  Detectar el terreno" o "Clasificar Suelo — CSF
   (Cloth Simulation Filter)" el título se recortaba en seco al ancho
   del panel (~200-210px). Reemplazado por el mismo patrón `_card()` de
   `tool_panel.py`/`tile_panel.py`: `QFrame` con `setObjectName` + QSS
   con ID-selector, encabezado en `QLabel` con `setWordWrap(True)`.
2. Cada fila de capa AGL era **una sola línea con 3 columnas fijas**
   (spinbox 85px + spinbox 85px + combo) — no cabía en el ancho del
   panel, se veía "Desde (m AGL) Hasta (m AGL) Clas...". Reescrito a
   **dos líneas por capa**: fila de rango (`sp_lo → sp_hi`) arriba,
   fila de clase + botón `×` eliminar abajo, dentro de una tarjeta
   `QFrame#aglRow` propia por capa.

Se preservó `_AGLBar` sin cambios (gradiente terreno→dosel es color de
DATO, no de tema — igual que en sección 23/24) y las firmas públicas
exactas (`set_schema`, `_rebuild_layers`, `_check_overlaps`,
`get_agl_layers`, `get_active_rules`, `is_flat_mode`, `get_csf_params`,
`get_agl_class_id`, `update_terrain`, `_on_select_click`) para no
romper `main_window.py`.

Verificado: `ast.parse` OK, `tests/verify_fixes.py` 100% OK (6/6
grupos), captura `grab()` con schema poblado (7 clases + terreno con
stats) a 260×1500 — sin recortes en ninguna de las 3 tarjetas (Paso 1
terreno, Paso 2 capas AGL con 6 capas por defecto + botón agregar,
CSF). Nota: el `Write` inicial fue rechazado una vez por "file modified
since read" (mi propio script de bump de fuentes había tocado el
archivo tras mi último Read) — se resolvió con un Read fresco antes de
reintentar.

## 26. TrainingPanel / InferPanel — mismo tratamiento de tarjetas

`ui/training_panel.py` (pantalla de Entrenar) y `ui/infer_panel.py`
(pantalla de Inferir) tenían el mismo patrón de `QGroupBox` + hex
sueltos que el resto de paneles antes del rediseño. **No se tocó**
`TrainingWorker` (los 3 modelos inline RandLA-Net/PointNet++/KPConv +
bucle de entrenamiento) ni `InferWorker` — esa es lógica de ML, no UI.

Cambios de UI en ambos:
- `QGroupBox` → mismo `_card(title, icon_name)` (QFrame ID-scoped +
  QLabel con word-wrap) que tool_panel/tile_panel/geo_panel.
- Todos los hex sueltos (`#e8e9eb`, `#0e7c86`, etc. — que ya eran los
  valores correctos de la paleta gris/cian, solo no importados desde
  `theme.py`) reemplazados por imports de `ui.theme`.
- **Bug de contraste real encontrado**: el botón "Iniciar
  entrenamiento"/"Ejecutar inferencia" tenía `background:{ACCENT}`
  (teal) con `color:{TEXT_DIM}` (gris oscuro) — texto difícil de leer
  sobre fondo de acento. Corregido a `color:#ffffff` + icono blanco
  (`play-fill`), igual que el patrón correcto ya usado en otros botones
  de acento de la app.
- Iconos nuevos descargados: `play-fill`, `stop-fill`, `graph-up`,
  `cpu-fill` (mismo método `curl` a jsdelivr, MIT, ver sección de
  iconos).
- `_LossChart` (gráfica de loss/mIoU dibujada a mano con QPainter): el
  fondo/ejes/texto "sin datos" ahora usan tokens de tema; las curvas
  en sí (train/val/mIoU) siguen siendo colores semánticos fijos
  (ACCENT/ACCENT_STRONG/OK), no recolor arbitrario.

**Lección de testing importante:** al capturar `InferPanel` a una
altura de ventana pequeña (560px) el cuadro verde de "info del modelo"
(3 líneas) se veía recortado en la captura. Investigar con
`widget.geometry()`/`sizeHint()` mostró que el layout comprimía la
etiqueta a 65px cuando pedía 79px — es decir, **la columna izquierda
completa (header + 2 tarjetas + nota + botones) necesita ~580px y a
560px de ventana simplemente no cabe** — no es un bug de diseño, es
que la captura de prueba usaba una ventana más baja que cualquier uso
real de la app. Confirmado repitiendo la captura a 820px: cero
recorte. Regla general para futuras capturas de paneles a pantalla
completa (Training/Infer, no los paneles laterales angostos): probar
con altura ≥800px antes de diagnosticar un "recorte" como bug real.

Verificado: `ast.parse` OK en ambos archivos, `tests/verify_fixes.py`
100% OK, capturas `grab()` de `TrainingPanel` e `InferPanel` a 980×820
sin recortes.

## 27. OverlayPanel / class_manager / export_dialog / layer_properties

Mismo tratamiento estructural que el resto: `_card()` ID-scoped,
iconos reales, tokens de `theme.py`.

- **`ui/overlay_panel.py`**: `_LayerRow` era una sola línea horizontal
  con hasta 7 controles (icono+nombre+estado+checkbox+slider+slider-Z+
  borrar) — recortado en el ancho real del panel (~200-230px).
  Reescrito como tarjeta de 2-3 líneas (nombre+estado+borrar /
  visibilidad+opacidad / altura-Z si es raster). Glifos ⬟/⬛/⚠/ℹ/⬆
  reemplazados por iconos reales (`diagram-3`, `layers`,
  `exclamation-triangle`, `info-circle`, `eye`/`eye-slash`, `trash`,
  `arrow-bar-up`).
- **`ui/class_manager.py`**: **bug de contraste real** — el
  `QLineEdit` de nombre editable de cada clase usaba
  `color:{TEXT_FAINT}` (`#a8acb0`, pensado para texto apenas visible)
  en el texto PRINCIPAL editable, dificultando la lectura. Corregido a
  `TEXT`. "×"/"✕" reemplazados por iconos `trash`/`x-lg`/`plus-lg`.
- **`ui/export_dialog.py`**: **dos bugs reales encontrados**:
  1. Contraste: `_ArchButton.set_selected()` pintaba el subtítulo con
     `#c9cbce`/`#d6d8da` — tonos de BORDE (`BORDER`/`BORDER_SOFT` en
     `theme.py`), casi invisibles como texto sobre los fondos claros
     del botón. Corregido a `TEXT_MUTE`/`TEXT_DIM`.
  2. **Bug de cascada de QSS sin selector** (el mismo patrón de
     `tool_panel.py`/`_RailItem` de la sección de rediseño de chrome):
     `_ArchButton.setStyleSheet(f"background:...;border:...")` sin
     selector alguno — Qt cascadea esa propiedad `border` a los
     QLabel hijos que no la sobrescriben, dibujando una caja dentro de
     otra alrededor de cada línea de texto envuelta (visible en
     captura, reproducido también en aislamiento sin la QSS global de
     la app → confirmado que es un bug real de la app, no artefacto
     del arnés de pruebas). Mismo bug en `_SplitInfo`. Corregido en
     ambos con `setObjectName()` + selector `QWidget#id{...}`.
  Lección: cuando una captura muestra algo raro, reproducir en un
  script aislado SIN la QSS global de la app — si el bug persiste, es
  real; si desaparece, es interacción con la QSS global (ver bug de
  fuentes ya documentado). Este caso fue lo primero (real).
- **`ui/layer_properties.py`**: solo recolor a tokens de tema + icono
  real en el título en vez de ⬟/⬛; sin bugs estructurales, el diseño
  original ya era razonable (ventana flotante angosta, filas
  etiqueta+slider+valor de una sola línea que sí cabían).

Verificado: `ast.parse` OK en los 4 archivos, `tests/verify_fixes.py`
100% OK, capturas de los 4 sin recortes ni bugs de contraste/cascada.
Nota aparte: `QMessageBox.warning/information` sigue provocando
segfault bajo `QT_QPA_PLATFORM=offscreen` (limitación pre-existente
del entorno de pruebas, no de la app — se reproduce igual con el
código original) — evitar disparar esas rutas en scripts de captura.

## 28. status_bar.py / splash.py / welcome_dialog.py

Con esto quedan TODOS los archivos de `ui/*.py` con paneles/diálogos
visibles pasados por el rediseño de tarjetas + tokens de tema + iconos
reales (excepto `main_window.py`, que es el shell/chrome ya rediseñado
en la sección de rediseño de chrome, y `styles.py`/`theme.py`/`icons.py`,
que son la infraestructura del tema, no paneles).

- **`ui/status_bar.py`**: **bug de contraste real** — el conteo
  principal de puntos (`_pts_main`) usaba `color:{TEXT_FAINT}`
  (`#a8acb0`) como color base del dato más importante de toda la barra
  de estado. Corregido a `TEXT_DIM`. Resto de hex sueltos migrados a
  tokens de tema (ya eran los valores correctos, solo no importados).
- **`ui/splash.py`**: el degradado de la barra de progreso quedó de
  una iteración de paleta anterior ("azul acero", comentario propio en
  el código lo delataba) — no coincidía con el ACCENT gris-cian actual.
  Recoloreado a tonos cian (versión "clara sobre fondo oscuro" del
  mismo ACCENT — el overlay oscuro sobre `splash_bg.png` es
  intencional para legibilidad, igual que el viewport 3D, así que NO
  se migró a tokens de tema plano ahí). `ui/splash_bg.png` en sí
  (arte de marca ámbar/negro) sigue sin decisión del usuario.
- **`ui/welcome_dialog.py`**: **dos bugs reales**:
  1. Contraste: botón "Comenzar" con `background:{ACCENT}` +
     `color:{TEXT_DIM}` (mismo patrón ya visto en training_panel/
     infer_panel) — corregido a texto blanco + icono blanco.
  2. Cascada de QSS: `_StepCard.setStyleSheet(f"QFrame{{...}}")` usaba
     el SELECTOR DE CLASE "QFrame" (no una regla bare, pero igual de
     amplio) — cascadeaba a CUALQUIER `QFrame` hijo que no lo
     sobrescribiera, incluyendo el propio separador `QFrame` (VLine)
     de la tarjeta. Corregido con `setObjectName("stepCard")` +
     selector `QFrame#stepCard{...}`.
  También: los 6 emoji de paso (🗃️🌍✏️📦🧠🔮) reemplazados por iconos
  reales (`cloud-arrow-up`, `layers-half`, `pencil-square`,
  `box-arrow-up`, `cpu-fill`, `magic`) — coherente con "iconos reales,
  no dibujos" pedido desde el principio del rediseño.

Verificado: `ast.parse` OK en los 3 archivos, `tests/verify_fixes.py`
100% OK, capturas de `WelcomeDialog` (800×900, todas las tarjetas
limpias, sin cascada de bordes) y `AnnotationStatusBar` (1000×80, texto
legible, barra de balance con colores reales de clase). Nota: el
glifo "✓" en "OCT ✓" se ve como tofu/caja en la captura offscreen —
limitación de fuente del arnés de pruebas (ver sección de fuentes),
no un bug de la app.

Con este batch se completa el pedido "quiero redisñar todos los
paneles y subpaneles de todo el software" — cobertura actual:
`main_window.py` (chrome), `tool_panel.py`, `tile_panel.py`,
`class_panel.py`, `geo_panel.py`, `training_panel.py`,
`infer_panel.py`, `overlay_panel.py`, `class_manager.py`,
`export_dialog.py`, `layer_properties.py`, `status_bar.py`,
`splash.py`, `welcome_dialog.py`. `ui/splash_bg.png`: el usuario
confirmó que la reemplazará él mismo más adelante — no es una tarea
pendiente nuestra, dejar de mencionarla como pendiente.

## 29. Feedback de uso real — 3 correcciones puntuales (2026-09-05)

Tras usar la app ya rediseñada, el usuario señaló 3 problemas
concretos (no eran solo "más pulido", eran bugs/UX reales):

**(a) Chips del header (`_HeaderBar`) mal ajustados.** Captura del
usuario mostraba "Overview"/CRS/"Proyecto" como cajas visualmente
desalineadas ("se ve mal hecho ahí de golpe"). Causa: los `QLabel` de
esos badges no tenían `setFixedHeight` ni `Qt.AlignVCenter` al
agregarlos al `QHBoxLayout` de 44px — Qt estira un `QLabel` sin
política de tamaño fija al alto completo disponible de la fila, así
que el fondo/borde del badge quedaba como una caja alta y desproporcionada
alrededor de un texto pequeño, en vez de una píldora ajustada. Fix:
helper `_chip()` con `setFixedHeight(22)` + `border-radius:11px`
(mitad de la altura → píldora real) + `lay.addWidget(w, 0,
Qt.AlignVCenter)` en cada elemento de la barra (separador, file_lbl,
chips, proj_btn, timer, save_lbl). Mismo fix aplicado a `_proj_btn`
(altura fija 26px, radio 13px).

**(b) `GeoPanel` — "PASO 1 Detectar terreno" no hacía nada real.**
Confirmado con grep: `is_flat_mode()` siempre devuelve `True`, así que
en `_on_agl_auto_classify` (main_window.py) la rama `else` que usaba
`self._terrain` (el MDT calculado por ese botón) NUNCA se ejecutaba —
el cálculo de AGL real siempre usa `z - z.min()` (modo plano). El
botón "Detectar terreno" literalmente no afectaba el resultado de
nada. Eliminado por completo: señal `terrain_compute_requested`,
`_calc_btn`, `_mdt_status`, método `update_terrain()` en
`ui/geo_panel.py`; en `ui/main_window.py` se quitó el `connect()`
correspondiente, el método `_on_compute_terrain` completo, y se
simplificó `_on_agl_auto_classify` a solo la rama modo-plano (se
quitó el `if/else` muerto y las 2 llamadas restantes a
`update_terrain`). `self._terrain`/`TerrainModel` se dejaron intactos
(otro código como `_on_agl_select`, ya inalcanzable de por sí, los
sigue referenciando; no vale la pena tocar más superficie por una ruta
ya muerta). Los 2 pasos restantes (AGL, CSF) se renumeraron PASO 1/2.

**(c) Rediseño UX de rangos AGL y de la lista de clases — pedido
explícito de "más entendible", con referencia a otros softwares:**

- **AGL (`geo_panel.py`)**: antes cada rango mostraba
  `[spinbox] → [spinbox]` en una fila y el combo de clase en la fila
  de abajo — no quedaba claro cuál número era mínimo/máximo ni qué
  franja de color representaba. Ahora: la clase va PRIMERO (combo +
  botón eliminar) y su color tiñe el borde izquierdo de toda la
  tarjeta (mismo lenguaje visual que las filas de clase); el rango de
  altura usa etiquetas explícitas "DESDE"/"HASTA" sobre cada spinbox
  en vez de una flecha ambigua. "Agregar capa" → "Agregar rango"
  (evitar la jerga interna "capa", que se confunde con las capas de
  `OverlayPanel`).
- **Clases (`class_panel.py`)**: v2 (Photoshop/Illustrator layers
  panel) tenía un swatch minúsculo + badge de tecla separados y un
  acento genérico (mismo teal) para "fila activa" sin importar la
  clase — difícil asociar número↔color↔clase de un vistazo. v3:
  badge circular grande (24px) que fusiona número de tecla + color
  real de la clase (metáfora tipo paleta, CVAT/Labelbox/Procreate);
  la fila activa se tiñe con EL COLOR DE ESA CLASE (no un acento
  genérico) vía helper `_tint()` (mezcla con blanco) — función nueva
  `_readable_on()` calcula luminancia para decidir texto blanco/gris
  oscuro sobre el badge según qué tan claro/oscuro sea el color de la
  clase. El chip superior "CLASE ACTIVA" (etiqueta técnica) pasó a
  "PINTANDO CON" con icono de pincel — metáfora de "color de primer
  plano" de Photoshop/Procreate. Filas más altas (30→36px) para un
  objetivo de clic más amigable.

Verificado: `ast.parse` OK en `geo_panel.py`/`main_window.py`/
`class_panel.py`, `tests/verify_fixes.py` 100% OK, capturas de
`_HeaderBar`+`_StepBreadcrumb` (900×74, chips ahora como píldoras bien
ajustadas), `GeoPanel` (260×1500, tarjetas AGL con borde de color por
clase y DESDE/HASTA claros) y `ClassPanel` (236×560, badges circulares
coloreados + fila activa teñida con el color de su propia clase).

## 30. Feedback de uso real — ronda 2 (2026-09-05)

**(a) Error real al cargar un .laz** ("read length must be non-negative
or -1", `core/pointcloud.py::_las`). Investigado a fondo — **dos bugs
reales, no solo mensaje de error mejorable**:
  1. El primer intento (`laspy.open(self.path)` para leer solo el
     encabezado) fallaba con ese mensaje —que en realidad indica un
     EVLR con longitud corrupta al FINAL del archivo, nada que ver con
     qué backend de descompresión está instalado. El código de
     fallback (`_read_laz_file`) descartaba inmediatamente cualquier
     error cuyo texto no contuviera "backend" ni "laz" (`if "backend"
     not in str(e).lower()...: raise`), así que este error concreto
     SIEMPRE se relanzaba sin probar nada más, y el mensaje final
     seguía sugiriendo "pip install laspy[lazrs]" — instalar lazrs no
     arregla un EVLR corrupto.
  2. Bug adicional encontrado en el propio código de fallback: buscaba
     `laspy.LazBackend.lazrs` / `.laszip` (minúsculas) — los miembros
     reales del enum en laspy 2.7 son `LazrsParallel`/`Lazrs`/`Laszip`
     (mayúscula inicial). `hasattr(laspy.LazBackend, "lazrs")` era
     SIEMPRE `False`, así que la ruta de "probar un backend explícito"
     nunca se ejecutaba en la práctica — ni en este código ni, por lo
     visto, en el original tampoco.
  **Fix**: `_read_laz_file` ahora prueba todas las combinaciones
  backend (`LazrsParallel`/`Lazrs`/`Laszip`, nombres correctos) ×
  `read_evlrs` (`True`/`False`) antes de rendirse — saltarse los EVLRs
  (`read_evlrs=False`) suele bastar para leer xyz/intensity/rgb aunque
  el EVLR final esté corrupto. El mensaje final ahora distingue
  "encabezado/EVLR dañado → re-exporta con CloudCompare/PDAL/lastools"
  de "backend faltante → pip install". Además, `_las()` recuerda con
  qué `read_evlrs` se pudo abrir el encabezado (`self._laz_read_evlrs`)
  y reutiliza ese mismo valor en las 3 llamadas posteriores a
  `laspy.open()` del resto del pipeline (cálculo de offset, streaming
  de puntos, atributos) — si solo se propagaba al primer intento, el
  streaming real de puntos habría vuelto a fallar con el valor por
  defecto. Probado end-to-end con un .las sintético generado con
  laspy: carga xyz correctamente.

**(b) Tooltips con texto casi invisible.** `QToolTip` en `ui/styles.py`
usaba fondo oscuro (`{TEXT}`) + texto casi blanco (`{SURFACE}`). En
Windows, `QToolTip` a veces solo honra la propiedad `color` de la QSS
y no `background` (el fondo nativo del tooltip del SO se cuela) — con
ese esquema, el resultado era texto claro sobre un fondo claro nativo,
casi ilegible (reportado y confirmado por captura del usuario). Fix:
invertido a fondo claro (`{SURFACE}`) + texto oscuro (`{TEXT}`,
básicamente negro) — seguro en ambos casos, se aplique o no el fondo
de la QSS.

**(c) Chips del header seguían sin encajar** (ronda 1 solo arregló el
estiramiento vertical; el usuario pidió ir más allá: "quita ese fondo
... o haz que converja mejor"). Quitado el fondo/borde por completo de
los badges de CRS y modo (Overview/Tile) — ahora son texto plano con
un icono como pista visual (`globe2` para CRS, `arrows-fullscreen`
para modo), sobre el mismo fondo de la barra, sin ninguna caja que
pueda desentonar. `_proj_btn` pasó de píldora con borde visible
siempre a botón plano (fondo/borde solo al hover) — mismo patrón que
los botones de icono del resto de la app. Helper `_chip()` (creado en
la ronda 1) quedó sin uso y se eliminó.

**(d) Dos líneas verticales de acento redundantes en ClassPanel.**
Tanto `_ActiveChip` ("PINTANDO CON") como la fila activa en la lista de
`_ClassRow` pintaban un `border-left: 3px solid {color}` — el usuario
pidió quitar AMBAS explícitamente ("no le pongas esas líneas
verticales de color"). Quitadas; el fondo teñido con el color de la
clase (`_tint()`) ya comunica el estado activo sin la barra vertical.

**(e) `ClassPanel` no podía redimensionarse en absoluto.**
`self.setFixedWidth(236)` en el constructor — a diferencia de
`ToolPanel`/`TilePanel`/`GeoPanel` (que ya usaban min/max o ninguna
restricción y sí se adaptan), arrastrar el borde del dock/splitter no
cambiaba nada porque el panel se negaba a ser otro ancho que 236px.
Cambiado a `setMinimumWidth(190)` + `setMaximumWidth(480)` (mismo
rango que el contenedor `_left_stack` en `main_window.py`). Verificado
con capturas a 190px y 440px: el layout (badge circular fijo + nombre
expandible + conteo) se adapta limpio en ambos extremos sin recortes.
Barrido `grep setFixedWidth` confirmó que ningún otro panel de
contenido (solo diálogos flotantes e iconos/spinboxes individuales,
que sí deben quedarse fijos) tenía este problema — el único caso real
era `ClassPanel`. `_StepRail` (176px fijo) se dejó intacto a propósito:
es un riel de navegación tipo barra de iconos (como la Activity Bar de
VSCode), no un panel de contenido — su ancho fijo es diseño
intencional, no un bug.

Verificado: `ast.parse` OK en `pointcloud.py`/`styles.py`/
`main_window.py`/`class_panel.py`, `tests/verify_fixes.py` 100% OK,
prueba end-to-end de carga LAS sintética, capturas de `_HeaderBar`
(sin cajas mal ajustadas) y `ClassPanel` a 190px/440px (sin líneas
verticales, sin recortes).

## 31. Editor de rangos AGL → diálogo ancho tipo tabla (2026-09-05)

El usuario pasó una imagen de referencia ("Configurar rangos": tabla
Clase | Desde (m) | Hasta (m), fila por clase con punto de color,
separadores finos, nota informativa, botones Cancelar/Guardar) y pidió
seguir ese diseño, con el "Hasta" de una clase ligado automáticamente
al "Desde" de la siguiente (encadenado, sin huecos).

Ese formato de tabla ancha no cabe de forma realista en el panel
lateral (~190-480px tras la sección 30 — y aun con su máximo, sigue
siendo un panel angosto para 3 columnas + acciones). Solución: nuevo
diálogo modal **`ui/agl_ranges_dialog.py::AGLRangesDialog`** (mín.
680px, mismo patrón que `ClassManagerDialog`/`ExportDialog` — copia
mutable de los datos, Cancelar descarta), y `ui/geo_panel.py` pasó de
construir filas editables inline a solo un **resumen de solo lectura**
(punto de color + nombre + rango en una línea por clase) + un botón
"Configurar rangos…" que abre el diálogo.

Cambios de modelo de datos en `GeoPanel`: la lista de capas dejó de
vivir como widgets vivos (`_layer_combos` con spinboxes/combos) y pasó
a ser datos simples `self._agl_data: List[dict]` (`{class_id, lo,
hi}`) — el diálogo edita una copia y solo se aplica a
`self._agl_data` si el usuario pulsa "Guardar clasificador". API
pública sin cambios (`get_agl_layers()` sigue devolviendo `[(class_id,
lo, hi)]`), así que `main_window.py` no necesitó tocarse para esto.

**Encadenado de rangos** (`AGLRangesDialog`): cada spinbox "Hasta"
está conectado a `on_hi_changed`, que empuja ese mismo valor al
"Desde" de la fila siguiente (con `blockSignals` para evitar recursión
infinita); cada "Desde" empuja al "Hasta" de la fila anterior de la
misma forma. Resultado: mover un límite mantiene los rangos continuos
automáticamente, como en la imagen de referencia. Verificado
programáticamente (cambiar `sp_hi` de la fila 0 mueve `sp_lo` de la
fila 1, y viceversa) — no solo visualmente.

Bug de construcción encontrado y corregido durante el desarrollo:
`_build_ui` llamaba a `_rebuild_rows()` (que a su vez llama a
`_check_overlaps()`, que usa `self._overlap_lbl`) ANTES de crear
`self._overlap_lbl` más abajo en el mismo método →
`AttributeError` al instanciar el diálogo. Corregido reordenando:
`_rebuild_rows()` se llama después de que `_overlap_lbl` ya existe.

## 32. Cabecera/breadcrumb: quitar fondos de acento propios

El usuario pidió que la franja de "Paso X de 6 — nombre — descripción"
(`_StepBreadcrumb`) use el mismo fondo neutro del resto de la UI en
vez de su propio tinte de acento (`ACCENT_SOFT`, un teal claro que
desentonaba con el gris neutro de `_HeaderBar` justo arriba).
Cambiado a `background:{SURFACE}` (igual que `_HeaderBar`) +
`border-bottom:1px solid {BORDER}` — misma paleta que el resto del
chrome, sin franja de color propia. `_HeaderBar` en sí ya usaba
`SURFACE` (neutro) — no necesitaba cambio, el problema era solo el
breadcrumb.

Verificado: `ast.parse` OK en `agl_ranges_dialog.py`/`geo_panel.py`/
`main_window.py`, `tests/verify_fixes.py` 100% OK, captura de
`AGLRangesDialog` (720×620, coincide de cerca con la imagen de
referencia del usuario), captura de `GeoPanel` con el nuevo resumen
compacto, captura de `_HeaderBar`+`_StepBreadcrumb` sin franja de
acento propia.

## 33. Feedback de uso real — ronda 3 (2026-09-05)

**(a) Eliminar una clase intermedia en "Configurar rangos" dejaba un
hueco.** Ejemplo exacto del usuario: 3 clases 1-2, 2-3, 3-4 → borrar
la clase intermedia (2-3) dejaba 1-2, 3-4 (con hueco entre 2 y 3) en
vez de reacomodarse a 1-2, 2-3. Corregido en
`AGLRangesDialog._remove_row`: tras quitar la fila, se re-encadenan
todas las filas restantes desde el índice 1 en adelante — cada una
desliza su "Desde" para que coincida con el "Hasta" de la fila
anterior, **preservando su propio ancho** (hi−lo) original, así que
3-4 (ancho 1) se convierte en 2-3 (ancho 1, no 2-4). La primera fila
queda fija como ancla. Verificado programáticamente con el ejemplo
exacto del usuario (3 clases, borrar la de en medio → resultado
`[(1, 1-2), (3, 2-3)]`, exactamente lo pedido).

**(b) Fondo con degradado de acento en la cabecera de
`WelcomeDialog`.** El título "GeoAnnotate3D" del diálogo de bienvenida
seguía con un degradado teñido (`ACCENT_SOFT→SURFACE_2`) de una
iteración de paleta anterior — no coincidía con el fondo neutro
(`SURFACE`) que ya usa `_HeaderBar` en la ventana principal. Corregido
a `background:{SURFACE}` liso, mismo tono que el resto del chrome de
la app.

**(c) Doble borde en el ítem activo del riel de navegación
(`_RailItem`, panel de la izquierda con Nube/Pre-clasificar/Etiquetar/
etc.).** Mismo bug de cascada de QSS sin selector visto antes en
`_ArchButton` (export_dialog.py) y `_StepCard` (welcome_dialog.py):
`_RailItem.setStyleSheet(f"background:...;border:...")` sin selector
cascadeaba el `border` al `_text_lbl` hijo (que solo sobrescribía
`background:transparent`, no `border:none`), dibujando un segundo
borde alrededor del texto — de ahí el efecto "doble caja" reportado.
Corregido con `setObjectName("railItem")` + selector `QWidget#railItem
{...}`, mismo patrón ya establecido. Verificado con captura: ahora una
sola caja limpia, mismos colores que antes.

Verificado: `ast.parse` OK en `agl_ranges_dialog.py`/`welcome_dialog.py`/
`main_window.py`, `tests/verify_fixes.py` 100% OK, prueba programática
del caso exacto de reacomodo de rangos, capturas de `_RailItem` (una
sola caja) y `WelcomeDialog` (cabecera neutra).

## 34. Feedback de uso real — ronda 4 (2026-09-05)

**(a) `_StepRail`: quitar la línea vertical delgada a la derecha.**
`self.setStyleSheet(f"background:{SURFACE};border-right:1px solid
{BORDER};")` — quitado el `border-right`; la diferencia de tono entre
`SURFACE` (riel) y `SURFACE_2` (panel de contenido) ya distingue las
zonas sin necesidad de una línea divisoria.

**(b) `_HeaderBar`/`_StepBreadcrumb`: fondo "blanco" propio.** Ambos
usaban `SURFACE` (`#f4f5f6`, pensado para tarjetas/superficies
elevadas — se percibe como un blanco/gris muy claro) en vez de `BG`
(`#e8e9eb`, el fondo base real de toda la app). El usuario pidió
explícitamente que la cabecera (icono nube + "GeoAnnotate3D") no
tuviera ese fondo propio, solo "el color de la UI original" — cambiado
a `BG` en ambas franjas.

**(c) Mejora de diseño añadida (no pedida explícitamente):**
`_RailItem` no daba ningún feedback visual al pasar el mouse sobre los
pasos NO activos (Pre-clasificar, Etiquetar, etc. no comunicaban ser
clicables) — agregado `QWidget#railItem:hover{background:...}` (gris
sutil `SURFACE_2` en pasos pendientes/completados).

**Sobre "optimización general del software"**: revisado
`TileGridWidget.paintEvent` (redibuja los ~64 tiles en cada movimiento
de mouse) como candidato — descartado explícitamente: 64
fillRect+drawRect+drawText por repintado es trivial para Qt, no es un
cuello de botella real, y "optimizarlo" sin evidencia de que sea lento
sería especulativo. Las optimizaciones de rendimiento reales ya se
hicieron al principio de la sesión (query por octree/grid en el
pincel, carga progresiva de tiles, cache mmap para nubes grandes — ver
secciones 1-4 de este archivo). Si el usuario nota una interacción
lenta concreta ahora, mejor señalarla puntualmente que adivinar dónde
optimizar sin perfilar.

Verificado: `ast.parse` OK, `tests/verify_fixes.py` 100% OK, capturas
de `_StepRail` completo (sin línea vertical) y `_HeaderBar` (fondo BG).

## 35. Preparación para v1.0 pública + repo de GitHub (2026-09-05)

El usuario dio luz verde para preparar el lanzamiento público y conectó
su repo: `https://github.com/Yutlanii/GeoAnnotate3D.git`.

**Repo**: el remoto YA tenía `LICENSE` (GPLv3 real), `README.md` (en
inglés, pulido, con placeholders `YOUR_USERNAME` y un badge que decía
"AGPL v3" pese a que el archivo LICENSE es GPLv3 puro — corregido en
ambos lugares) y `docs/images/banner.png`, todo en la carpeta raíz del
repo, con el proyecto en sí dentro de una subcarpeta `geoannotate3d/`.
Se inicializó git en `D:\...\GeoAnnotate3D_v10_cargar_toda_la_nube_siempre\`
(un nivel arriba del proyecto, para que la carpeta local `geoannotate3d\`
coincida exactamente con esa subcarpeta del repo), remote `origin`
apuntando al URL de arriba, identity `user.name=Yutlanii` /
`user.email=jaimeyutlany@gmail.com`. Se copiaron `LICENSE`, `README.md`
y `docs/` del remoto a la raíz local, y se creó `.gitignore`
(`__pycache__`, `.geoannotate_recent.json`, `*.zip`, `build/`, `dist/`).
El código local (`geoannotate3d/`) es la versión NUEVA (post-rediseño
completo) — el remoto tenía una versión más antigua que ni siquiera
coincidía en estructura de archivos exactamente (algunos módulos como
`render/edl.py`/`core/octree.py`/`annotation/region_growing.py` SÍ
existen en ambos, solo que yo no los había listado antes — no hay
pérdida de funcionalidad al sincronizar).

**9 mejoras concretas pedidas explícitamente, todas implementadas y
verificadas:**

1. **Tamaños de letra uniformemente pequeños**: script que sube todo
   `font-size` menor a 10.5px a un piso de 10.5px en los 15 archivos
   de `ui/*.py` que lo tenían (excepto `icons.py`/`theme.py`). Re-
   verificado con capturas de `ClassPanel`, `GeoPanel` y `ToolPanel`
   tras el cambio — sin recortes nuevos.

2. **Referencia de atajos in-app** (`ui/shortcuts_dialog.py`, nuevo,
   accesible desde el nuevo menú "Ayuda" en `_HeaderBar`, F1). La lista
   se construyó LEYENDO el código (`keyPressEvent` + `annotation/
   tools.py`), no adivinada — y al hacerlo se encontraron **2 bugs
   reales**: `DiscTool` (tecla "V") y `SphereSelectTool` (tecla "R")
   colisionaban con los atajos de cámara (V=vista cenital, R=reset,
   comprobados ANTES que el mapa de herramientas en `keyPressEvent`) —
   esas 2 herramientas NUNCA respondían a su tecla. Reasignadas a "D"
   (Disco) y "H" (Esfera). El menú "Ayuda" también por fin hace cierto
   el texto de `WelcomeDialog` ("Ayuda → Bienvenida") que antes no
   tenía nada real detrás (no existía ningún menú de ayuda en la app).

3. **Persistencia de layout de ventana**: la app no usa `QDockWidget`
   (todo es `QSplitter`/`QStackedWidget` a mano), así que
   `QMainWindow.saveState()` no aplica — en su lugar,
   `_save_window_state()`/`_restore_window_state()` usan `QSettings` +
   `saveGeometry()/restoreGeometry()` (tamaño/posición/maximizado) +
   `_main_splitter.saveState()/restoreState()` (ancho de paneles).
   Guardado en `closeEvent`, restaurado al final de `__init__`.

4. **Códigos de clasificación ASPRS en la exportación .las**
   (`annotation/exporter.py`): antes el .las clasificado escribía los
   IDs internos del schema (arbitrarios) directo en `classification` —
   abrir eso en CloudCompare/QGIS/ArcGIS mostraba clases sin sentido.
   Nuevo `ExportConfig.asprs_codes` (default `False`, checkbox nuevo en
   `ExportDialog`) + `_asprs_code_for(class_id, name)`: heurística de
   palabras clave (mismo patrón que CSF/AGL en geo_panel.py) mapea a
   códigos ASPRS estándar (2=suelo, 3/4/5=veg baja/media/alta,
   6=edificio, 9=agua, 10=riel, 11=carretera, 17=puente); lo no
   reconocido cae en 64+class_id (rango reservado por la especificación
   para uso del usuario, nunca colisiona con un código estándar).
   Remapeo vectorizado vía LUT de 256 entradas (no por punto — nubes de
   cientos de millones de puntos). Verificado con la heurística +
   remapeo LUT completo.

5. **Reanudar entrenamiento desde checkpoint** (`ui/training_panel.py`):
   antes el checkpoint solo guardaba `model_state` — no había forma de
   continuar un entrenamiento largo tras cerrar la app. Ahora el
   checkpoint también guarda `optimizer_state`/`scheduler_state`/
   `epoch`; nuevo campo "Reanudar" en el panel (selector de archivo
   `.pth`) que, si se completa, restaura los 3 y ajusta el rango del
   bucle de epochs (`range(start_epoch, epochs+1)`) y el cálculo de ETA
   (que debía dividir por epochs corridos ESTA sesión, no por el
   número absoluto de epoch, o subestimaría el tiempo restante tras
   reanudar). Verificado con un round-trip real de PyTorch (modelo +
   Adam + StepLR): pesos, optimizer y posición del scheduler se
   restauran exactamente.

6. **Reporte de validación por clase** (no solo mIoU agregado):
   `_train_with_model` ahora acumula TP/FP/FN reales (no solo el IoU
   promediado por lote que ya existía) durante la validación del MEJOR
   epoch, y al terminar el entrenamiento escribe `class_report.json` +
   `class_report.txt` (IoU/precision/recall/soporte por clase, más una
   sección "clases con IoU < 0.50" para saber qué clase necesita más
   datos). Verificado con aritmética a mano (TP=90,FP=10,FN=5 →
   IoU=0.8571 ✓).

7. **Inferencia por lote** (`ui/infer_panel.py`): nueva clase
   `BatchInferWorker` + tarjeta "Inferencia por lote (carpeta)" —
   reutiliza `infer.py::load_cloud/load_model/infer_cloud/save_result`
   (ya existían como CLI standalone, no se reinventó nada) para
   procesar todos los archivos soportados (.las/.laz/.ply/.npy/.txt/
   .csv/.xyz/.pts/.asc) de una carpeta con el modelo cargado UNA sola
   vez, escribiendo `<stem>_classified.las` por archivo en
   `<carpeta>/batch_output/`. Verificado el filtro de extensiones con
   archivos sintéticos.

8. **Reporte de errores / crash con ruta de reporte real**
   (`ui/main_window.py::_setup_crash_handler`): antes `faulthandler` y
   el `excepthook` solo escribían a stderr — invisible en el .exe
   empaquetado (ventana sin consola), el programa "simplemente se
   cerraba" sin dejar rastro. Ahora ambos también escriben a
   `~/.geoannotate3d/logs/crash.log`, más un marcador de sesión
   (`.session_running`, creado al iniciar, borrado en `closeEvent`
   limpio) que permite detectar si la sesión ANTERIOR murió sin pasar
   por el cierre normal — si es así, un aviso post-arranque ofrece
   abrir la carpeta de logs. Nuevo menú "Ayuda → Abrir carpeta de
   registros". Lógica de detección verificada con una simulación de 4
   sesiones consecutivas (limpia/crash/detecta/no-repite).

9. **README actualizado**: badges y links `YOUR_USERNAME`→`Yutlanii`,
   AGPL→GPL (para que coincida con el LICENSE real), nuevas viñetas de
   feature para los 6 puntos de arriba que tienen impacto de cara al
   usuario (ASPRS, resume, reporte por clase, batch, atajos, layout
   persistente, reporte de errores).

Verificado en conjunto: `ast.parse` en TODOS los `ui/*.py` +
`annotation/*.py` + `core/*.py` tocados, `tests/verify_fixes.py` 100%
OK, capturas de `ShortcutsDialog`/`TrainingPanel`/`InferPanel` sin
recortes.

**Pendiente / fuera de alcance de esta ronda** (documentado para no
perderlo, no bloqueante para un primer push): páginas de
`docs/quickstart.md`/`user-guide.md`/`tutorial.md`/`faq.md` y
capturas `docs/images/screenshot-*.png` que el README ya enlaza son
aspiracionales (no se pidieron explícitamente esta ronda); `CONTRIBUTING.md`
referenciado por el README tampoco existe todavía.

## 21. Estado de este análisis

Leí completos: `main.py`, `README.md`, `ui/main_window.py` (2300/2300),
`ui/styles.py`, `ui/tool_panel.py`, `ui/class_panel.py`, `core/pointcloud.py`,
`core/project.py`, `utils/geo.py`, `utils/spatial.py`. Para el resto (canvas.py,
tools.py, octree.py, exporter.py, infer.py, tile_manager.py, heavy_cloud.py,
overlay_manager.py, terrain.py, rules_engine.py, geo_panel.py, tile_panel.py,
infer_panel.py, training_panel.py, export_dialog.py) solo mapeé firmas de
clases/funciones (`grep ^class|^def`) + fragmentos — suficiente para entender la
arquitectura pero **antes de tocar código en esos archivos, releer el archivo
completo primero**, no fiarse solo de este resumen.
