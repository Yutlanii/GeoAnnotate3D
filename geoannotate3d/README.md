# GeoAnnotate3D v0.2.0 — Turbo Edition

Herramienta de anotación semántica para nubes de puntos LiDAR geoespaciales.
Orientada a crear datasets de entrenamiento para RandLA-Net, KPConv y PointNet++.

## Instalación rápida

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt

# Compilar extensión C (20x más rápido en anotación, 3-8x en LOD)
python core/build_extension.py

python main.py
python main.py ruta/a/nube.laz
```

## Dependencias

| Paquete | Uso |
|---|---|
| PyQt5 | Interfaz gráfica |
| vtk>=9.2 | Motor de render 3D C++ OpenGL |
| numpy | Operaciones sobre arrays |
| scipy | cKDTree para sphere_query (400-4000x speedup vs brute force) |
| laspy[lazrs] | Leer archivos LAS/LAZ |
| matplotlib | Colormaps |

```bash
pip install PyQt5 vtk numpy scipy "laspy[lazrs]" matplotlib
pip install h5py open3d   # opcionales
```

## Optimizaciones de rendimiento (v0.2.0)

### Extensión C nativa (`core/_fastcore.c`)
Compilada con `gcc -O3 -march=native -ffast-math`. Provee:
- `apply_lut_u8` — modo Anotación: **20x más rápido** que numpy puro
- `voxel_subsample_c` — decimación voxel-grid: **5-10x más rápido**
- `build_lod_levels_c` — 4 niveles LOD precomputados: **3-8x más rápido**, sin GIL
- `compute_z_colors` — elevación → color en C: **15x más rápido**
- `sphere_query_c` — búsqueda radial bruta en C
- `frustum_cull_points` — culling por punto antes de GPU

### scipy cKDTree para sphere_query
- BrushTool y RadiusTool usan cKDTree del octree
- **400-4000x más rápido** que búsqueda bruta O(N)
- Para nube de 20M pts con radio 5m: 0.1ms vs 200ms

### LOD precomputado O(1)
- 4 niveles LOD calculados en C al cargar
- `get_coarse_view()` es O(1) — solo un lookup de array
- La vista inicial aparece apenas termina el LOD C (~1s para 10M pts)
  antes de que el octree BFS termine (~40s)

### Buffer GPU persistente (VTK)
- `_update_vtk_all` no reconstruye `vtkPolyData` si N no cambia
- `_upload_colors_only` solo sube 4 bytes/punto a GPU durante anotación

### Pipeline de carga
- `cloud_ready` se emite con vista LOD l0 antes de que el octree BFS termine
- Atributos LAS liberados de RAM explícitamente tras conversión
- Coordenadas operadas in-place (evita array temporal doble)

## Herramientas de anotación

| Tecla | Herramienta | Uso |
|---|---|---|
| B | Pincel | Click+drag — pinta esfera 3D (cKDTree) |
| L | Lazo | Polígono en pantalla → Enter para aplicar |
| X | Caja | Click+drag define AABB 3D |
| P | Plano | Click define nivel Z — etiqueta arriba/abajo |
| R | Radio | Click individual en objeto |
| I | Pick | Inspeccionar punto — coordenadas UTM |
| M | Medir | Click A + Click B — distancia 3D |

## Atajos de teclado

| Tecla | Acción |
|---|---|
| 1–9 | Cambiar clase activa |
| B/L/X/P/R/I/M | Cambiar herramienta |
| Ctrl+Z | Deshacer |
| Ctrl+Shift+Z | Rehacer |
| Ctrl+S | Guardar proyecto |
| Space | Resetear vista |

## Arquitectura del código

```
geoannotate3d/
  main.py
  requirements.txt
  core/
    _fastcore.c          ← NUEVO: extensión C (gcc -O3 -march=native)
    _fastcore.so         ← compilado automático
    build_extension.py   ← compilador automático
    _fast.py             ← interfaz Python → C con fallback numpy
    octree.py            ← v8.0: LOD C + cKDTree + BFS
    pointcloud.py        ← loaders LAS/PLY/PCD/XYZ/NPY
    project.py           ← modelo .geoa3d (ZIP)
    terrain.py           ← MDT + AGL
    rules_engine.py      ← clasificación automática por reglas
  render/
    canvas.py            ← v2.0: VTK C++ + buffer GPU persistente
    colors.py            ← v1.0: rutas C para Anotación y Elevación
    lod_worker.py        ← v2.0: coarse O(1) + refinamiento BFS
    lod.py               ← stub de compatibilidad
  annotation/
    tools.py             ← v0.3.0: sphere_query via cKDTree
    label_store.py       ← undo/redo stack
    exporter.py          ← RandLA-Net / KPConv / PointNet++
  ui/
    main_window.py       ← ventana principal
    class_panel.py       ← panel de clases
    tool_panel.py        ← panel de herramientas
    status_bar.py        ← stats + balance
    export_dialog.py     ← modal de exportación
    styles.py            ← tema amber/oscuro
  utils/
    spatial.py           ← v2.0: frustum culling C, sphere_query C
    geo.py               ← CRS, UTM, bbox
```

## Exportación del dataset

El botón **EXPORTAR DATASET** genera:
- **Arquitectura**: RandLA-Net / KPConv / PointNet++
- **Split espacial**: bloques 50m × 50m sin data leakage geográfico
- **class_weights**: frecuencia inversa normalizada
- **Formatos**: `.npy+json` / `.pkl` / `.h5` / `.las etiquetado`

## Formato de proyecto .geoa3d

ZIP con:
- `meta.json` — metadatos, schema de clases, CRS
- `labels.npy` — array uint8 (N,) con etiqueta por punto
