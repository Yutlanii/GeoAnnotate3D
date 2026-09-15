# GeoAnnotate3D — Complete User Manual

GeoAnnotate3D organizes work as a 6-step workflow, shown as a vertical rail on the left of the window: **Point Cloud → Preclassify → Label → Export → Train → Infer** (in the app: `Nube → Pre-clasificar → Etiquetar → Exportar → Entrenar → Inferir`). Below **Infer** (`Inferir`) the rail also has a **Reference Layers** (`Capas de referencia`) entry, which is not a workflow step but an auxiliary view (overlay vector/raster layers) available at any point.

The header bar (top) shows the app name, the loaded file and point count, its CRS, whether you're viewing the whole cloud (Overview) or a single tile, a **Project** (`Proyecto`) menu (new/open/save/recent), and a **Help** (`Ayuda`) menu (shortcuts reference and this documentation's entry points).

---

## 1. Point Cloud (`Nube`)

**Loading.** `Proyecto → Nuevo proyecto` (Project → New Project, `Ctrl+N`) or drag-and-drop a file onto the window. Supported formats: `.las`, `.laz`, `.e57`, `.ply`, `.pcd`, `.xyz`, `.txt`, `.csv`, `.asc`, `.pts`, `.npy`, plus GeoAnnotate3D's own `.ga3d_bin` cache format for very large clouds.

- If the file has RGB, it opens in **RGB** color mode; otherwise it opens colored by **elevation**. You can switch color mode from the tool panel's **Visualization** (`VISUALIZACIÓN`) section at any time (RGB / Elevation / Intensity / Annotation).
- `.laz` files need a LAZ decompression backend (`lazrs` or `laszip`, both installable via `pip install "laspy[lazrs]"`). If a file fails to open with an error mentioning EVLR or header length, the file's header itself may be non-standard — try re-exporting it with CloudCompare, PDAL, or `lastools` (`lasinfo`/`las2las`) and reload. When `lazrs` is available, decompression explicitly uses its multi-threaded backend for faster loading.
- **COPC** (`.copc.laz`, Cloud Optimized Point Cloud) files are detected and loaded via a dedicated code path (their internal point storage isn't compatible with a regular LAZ streaming reader) — the file info shows "COPC" for these. GeoAnnotate3D loads the full point set at once; it does not yet use COPC's internal spatial index for partial/streamed loading of very large files.

**Tiles.** For clouds too dense to comfortably navigate as a whole, the **Overview** panel lets you configure a tile grid:

- **Tile size** (`Tamaño de tile`) — width/height of each tile in meters.
- The grid preview shows one cell per tile; percentages indicate how much of that tile is already labeled.
- **View mode** (`MODO DE VISTA`) — labeled buttons **Sparse** (`Disperso`, fast, low density, good for huge clouds), **Max density** (`Densidad máx.`, loads progressively up to the finest detail available) and **Fly** (`Volar`, free WASD camera, key `F`).
- **Camera** (`CÁMARA`) — **Top-down** (`Cenital`, key `V`), **Side** (`Lateral`, key `Y`), **Fit view** (`Encuadrar`, fit the whole cloud in view, key `R`).
- **Transform** (`TRANSFORM`) — three labeled mouse modes decide what dragging on the canvas does: **Navigate** (`Navegar`, normal camera controls — the default), **Move grid** (`Mover grid`) and **Rotate grid** (`Rotar grid`, drag to reposition/rotate the tile grid itself instead of the camera). Use the Offset X/Y and **Rotation** (`Rotación`) fields for precise numeric alignment, or **Reset** to zero them out.
- Click a tile to enter **tile mode** (loads just that tile at full density); `Escape` or the **← Overview** (`← Vista global`) button returns to Overview.
- Filters below the grid — **Unlabeled / Partial / Complete** (`Sin anotar / Parcial / Completo`) — let you jump straight to tiles that still need work.
- Hover any tile (even at small grid-cell sizes, where inline text doesn't fit) to see a tooltip with its coordinates, labeled percentage, and whether it's the currently active tile.

### What is `.ga3d_bin`, and why does it matter?

`.ga3d_bin` ("GA3D-Bin") is GeoAnnotate3D's own binary point-cloud format — not a competing standard, but a fast **cache** the app creates automatically next to a `.las`/`.laz`/`.e57` file once it crosses **200 million points**.

The difference from LAS/LAZ/E57 is structural, not just "a different extension":

- **LAS/LAZ/E57** are designed for interchange between many tools, so every read has to parse a structured record format (and, for `.laz`, decompress it first) before you get usable coordinates.
- **GA3D-Bin** is a flat binary layout — a small 512-byte header followed by raw `float32` XYZ (and other attributes) with no compression and no quantization. That means it can be opened with **memory-mapping (mmap)** instead of reading the whole file into RAM: the OS pages data in from disk on demand, so you can work with a cloud far larger than your available memory, and reopening it is essentially instant because there's no decompression or parsing step to redo.

You don't need to manage this yourself: the first time you open a huge `.las`/`.laz`/`.e57`, GeoAnnotate3D offers to convert it once; from then on, opening that *same source file* again detects the existing `.ga3d_bin` and asks whether to use it (much faster than reconverting or re-parsing the original). You can also open a `.ga3d_bin` file directly. It lives alongside the original file (same name, `.ga3d_bin` extension) and is safe to delete — GeoAnnotate3D will just offer to regenerate it from the source the next time you open that file.

---

## 2. Preclassify (`Pre-clasificar`)

Two independent, automatic methods to reduce manual labeling. Neither depends on the other.

**Classify by height / AGL** (`Clasificar por altura (AGL)`) — classifies points by height above the lowest point of the cloud (a "flat" AGL model; it doesn't need a terrain surface). Click **Configure ranges…** (`Configurar rangos…`) to open the range editor: each row is a class with a **From (m)** / **To (m)** (`Desde (m)` / `Hasta (m)`) range. Editing one boundary automatically slides the adjacent range's boundary to match, so ranges stay continuous with no gaps — and deleting a class re-closes the gap it leaves, keeping the remaining ranges adjacent. Click **Classify by AGL layers** (`Clasificar por capas AGL`) to apply.

**Ground by cloth simulation / CSF** (`Suelo por simulación de tela (CSF)`) — Cloth Simulation Filter, the same ground-detection algorithm used by CloudCompare: simulates a cloth falling onto the inverted point cloud to find the ground surface. Requires `pip install cloth-simulation-filter`. Pick the target class and cloth resolution (smaller = more detail, slower), then **Detect ground / CSF** (`Detectar suelo (CSF)`).

**Clean up noise / SOR** (`Limpiar ruido (SOR)`) — Statistical Outlier Removal, the same noise-detection algorithm used by CloudCompare/PCL: for each point, compares its mean distance to its `k` nearest neighbors against the cloud's overall average — points that are abnormally isolated are flagged as sensor noise. Adjust **Neighbors (k)** (`Vecinos (k)`) and **Sensitivity** (`Sensibilidad`, lower = more aggressive), run **Detect noise / SOR** (`Detectar ruido (SOR)`), then confirm to delete the flagged points (undoable with `Ctrl+Z`, same as any other annotation).

**Annotation QA** (`QA de anotación`) — two more diagnostic tools once you've labeled at least part of the cloud:

- **Smooth labels** (`Suavizar etiquetas`) — each labeled point recalculates its class as the majority vote among its `k` nearest neighbors, cleaning up noisy boundaries between classes (e.g. a stray brush stroke that bled one class into another). It never touches unlabeled points, never touches deleted points, and never uses them as votes either — a point with no real neighbors of a different class simply keeps its current label.
- **Detect isolated clusters** (`Detectar clusters aislados`) — for each class, groups its points by spatial connectivity and reports clusters smaller than **Max points/cluster** (`Máx. puntos/cluster`) — candidates for an accidental click or other annotation error. Purely diagnostic: nothing is changed automatically, you review the report and fix flagged spots by hand (Brush/Sphere/delete points).

---

## 3. Label (`Etiquetar`)

**Classes** (left panel): each class is a numbered, colored circle — press `1`-`9` or click a row to make it the active ("painting with") class. Double-click a row to rename it or change its color. The pencil icon next to **Classes** (`CLASES`) opens the full class manager (add/remove classes, bulk edit, visibility filter). The bottom of the panel shows overall labeled coverage.

**Tools** (right panel), each with a single-key shortcut and its own options section:

| Tool | Key | Behavior |
|---|---|---|
| Brush (`Pincel`) | `B` | Spherical brush; `Ctrl`+drag paints a continuous stroke. Radius, stroke density, and disc thickness are adjustable; scroll wheel resizes it live. |
| Disc (`Disco`) | `D` | A 3D disc oriented to the local surface; `Ctrl`+click places it. |
| Region Growing | `G` | `Ctrl`+click a seed point; grows outward through connected, similarly-colored geometry. |
| Fit plane (`Ajustar plano`) | `P` | `Ctrl`+click fits the dominant plane inside a search radius (RANSAC — same idea as CloudCompare's "RANSAC Shape Detection") and labels only the points that turn out to be flat; non-planar points in the same radius (a chimney, a railing) are left untouched. A live circle shows the search radius as you move the mouse; after fitting, a cyan outline shows exactly what was detected as planar. Radius and tolerance (how far a point can be from the plane and still count) are adjustable in the tool's options. |
| Polygon (`Polígono`) | `L` | Draw a 2D/3D polygon; everything inside gets the active class. |
| Box (`Caja`) | `X` | Box selection. |
| Sphere (`Esfera`) | `H` | `Ctrl`+click for an instant spherical selection (vs. Brush's continuous stroke). |
| Z-slice (`Corte Z`) | `C` | Slice/cross-section selection by height. |
| Pick | `I` | Sample the class of the point under the cursor (to check what's already labeled). |
| Measure (`Medir`) | `M` | Measure distances in the scene — click point A, click point B. The measurement stays anchored in the scene (see **Persistent markers** below). `Ctrl`+click an *existing* measurement to edit its color/line width or delete it. |
| 3D Label (`Etiqueta 3D`) | `N` | `Ctrl`+click to place a short text label anchored at that point — a note that stays visible and is saved with the project. `Ctrl`+click an *existing* label instead of empty space to edit its text, font size, and color, or delete it. |
| Polyline (`Polilínea`) | `K` | `Ctrl`+click adds a vertex, `Enter` finishes and saves it, `Backspace` removes the last vertex, `Escape` cancels without saving. `Ctrl`+click an *existing* polyline instead of empty space to edit its color/line width or delete it. Digitizes an open, multi-vertex line — a road/utility centerline, a talus edge — something a single 2-point measurement or a closed polygon can't represent. |
| Profile (`Perfil`) | `O` | `Ctrl`+click A, click B — opens a 2D cross-section view of the strip of cloud around that line (distance along the line vs. elevation), with adjustable strip width and vertical exaggeration. Useful for checking slopes, power lines, or road sections without rotating the 3D camera to an awkward side angle. |

`E` toggles **erase mode** on the active tool — clears the class back to unlabeled (0), it does not remove the point. `Supr` (Delete) toggles **delete mode** instead — mutually exclusive with erase mode — which removes the selected points from the cloud entirely (useful for cleaning up sensor noise or stray points); this works with *any* of the selection tools above, not just one dedicated tool. Deleted points are excluded from every export and are undoable like any other annotation (`Ctrl+Z`). `Ctrl+Z` / `Ctrl+Shift+Z` undo/redo. `Ctrl`+click on the canvas is the common "commit" gesture for the click-based tools.

**Persistent markers (3D labels, measurements, and polylines).** Unlike the selection tools above, Measure, 3D Label, and Polyline don't change any point's class — they leave a permanent visual marker (a small sphere + floating text, a line between two points for a measurement, or a connected multi-vertex line for a polyline) anchored in 3D space, saved with the project (`.geoa3d`) so they're still there next time you open it. `Ctrl`+click any existing marker (instead of empty space) to edit or delete it — every marker kind supports this. The floating text is always rendered on top of the point cloud so it can't get hidden behind it as the camera moves. Use them for field notes, QA remarks, digitizing a centerline, or documenting a distance for a report, independent of the classification itself.

**Profile / vertical cross-section (`Perfil`).** Trace a line A→B with the **Profile** tool and a separate window opens showing the strip of cloud around that line unrolled into 2D: horizontal axis is distance along the line, vertical axis is elevation, colored by classification when available (with a legend of only the classes actually present in that strip) or by height otherwise. Drag to pan, scroll to zoom, hover to read the exact distance/elevation of the nearest point, and adjust vertical exaggeration to see subtle elevation changes over a long horizontal run — the strip width is adjustable in the tool's options section. **Export CSV** (`Exportar CSV`) saves the extracted points (distance, elevation, and RGB if colored by class) to a file for further analysis outside the app. Doesn't select or change anything in the 3D scene; it's purely an inspection view, and reopens the same window for each new A→B line you trace instead of stacking windows.

**Clip box** (`Caja de recorte`). In the tool panel's **Visualization** (`VISUALIZACIÓN`) section, the **Clip box** toggle overlays a draggable 3D box — drag its faces to isolate a volume, everything outside it is temporarily hidden. It's purely visual (nothing is deleted or reclassified): turn it off to see the full cloud again exactly as before. Useful for inspecting the inside of a structure or working in a crowded scene without other tools' selections reaching hidden geometry — the annotation tools still only ever act on the visible/queried points as usual.

**Reference layers** (rail entry below Infer, or accessible while labeling): overlay georeferenced orthomosaics (`.tif`) or vector layers (`.shp`, `.geojson`, `.gpkg`, `.kml`, `.dxf`) with real-world coordinates to guide labeling. Lower the point cloud's own opacity from this panel to see an underlying raster more clearly, and use **Top-down view** (`Vista cenital`) to align the camera top-down.

---

## 4. Export (`Exportar`)

Pick one or more target architectures — **RandLA-Net**, **PointNet++**, **KPConv** — the export format for each is chosen automatically (you don't pick a file format separately). Options:

- **Also export universal format** (`También exportar formato universal`) — a `.npy`+`json` export usable by any architecture without conversion.
- **Export classified cloud (.las)** (`Exportar nube clasificada (.las)`) — a classified `.las` for viewing the result in CloudCompare/QGIS/ArcGIS. Check **Use standard ASPRS classification codes** (`Usar códigos de clasificación ASPRS estándar en el .las`) to remap your custom class IDs to the standard ASPRS codes (ground=2, low/medium/high vegetation=3/4/5, building=6, water=9, rail=10, road=11, bridge=17, unclassified=1) via keyword matching on class names; anything unrecognized is written into the 64-255 range the LAS 1.4 spec reserves for user-defined codes, so it never collides with a real standard code.
- **Labeled points only** (`Solo puntos etiquetados`) — excludes unlabeled points from the dataset.

The split between train/val/test is spatial (separate geographic blocks) to avoid data leakage between splits.

**Growing one dataset from several point clouds**: point **Output folder** (`Carpeta de salida`) at the *same* folder you already exported to, from a different (or newly annotated) point cloud, and GeoAnnotate3D adds to that dataset instead of overwriting it. Every tile file is named with a short tag derived from its source cloud, so tiles from different clouds never collide, and the classified `.las` gets a disambiguated name too if needed. `dataset.json` is merged rather than replaced: `n_tiles_total`, the train/val/test tile lists, `per_class_counts` and `class_weights` all accumulate across every export made into that folder, and `source_files` lists every cloud that contributed to it. This only makes sense across exports that share the same class schema (the same GeoAnnotate3D project) — mixing schemas writes a `schema_warning` field into `dataset.json` so you notice, but doesn't stop the export.

---

## 5. Train (`Entrenar`)

Three architectures are implemented directly in the app (no external training framework needed): **RandLA-Net** (fast, large clouds), **PointNet++** (multi-scale accuracy), **KPConv** (fine edge detail). Point **Dataset** at the exported data folder, set epochs/batch size/points-per-sample/learning rate, and click **Start training** (`Iniciar entrenamiento`). Loss and mIoU update live in the chart.

**Resuming a training run.** Use the **Resume** (`Reanudar`) row to pick a previously saved checkpoint (`.pth`) — training continues from that epoch with the optimizer and learning-rate scheduler state restored, not just the model weights, so it behaves as if it had never stopped. Safe to close the app mid-run and resume later.

**Fine-tuning from an external checkpoint.** The **Fine-tuning** row is different from **Resume** (`Reanudar`): it starts a *new* training run (epoch 1, fresh optimizer/scheduler) but loads its starting weights from another checkpoint instead of random initialization — layer-by-layer, matching by name *and* shape, skipping (and randomly re-initializing) whatever doesn't match. In practice that means the classification head gets reset when the checkpoint came from a project with a different number of classes, while the rest of the network keeps what it already learned. This is meant for continuing to learn from a model this same app already trained on a different project/dataset; a checkpoint from the original third-party RandLA-Net/PointNet++/KPConv repositories won't have matching layer names (these are from-scratch re-implementations) so it would load 0 tensors — the log tells you exactly how many tensors were reused vs. reset either way, instead of silently pretending it helped.

**Per-class report.** When training finishes, alongside `best_model.pth` you get `class_report.json` and `class_report.txt` in the output folder: precision, recall, and IoU for *each* class individually (not just the aggregate mIoU), plus how many points supported each class — useful for spotting which classes need more labeled data.

**Export to ONNX** (`Exportar a ONNX`). Converts an already-trained checkpoint to ONNX format, to run inference outside this app (another pipeline, an edge device, a server without PyTorch). Independent of any training in progress — just pick a `.pth` and where to save the `.onnx`; architecture and class count come from the panel's current settings, which must match the checkpoint you're exporting (a mismatch is reported, not silently ignored). RandLA-Net's forward pass uses a random point subsample that ONNX has no equivalent operator for — its exported graph uses a fixed subsample instead, noted in the result dialog; PointNet++ and KPConv export without that caveat. Requires the `onnx` Python package installed in addition to PyTorch.

---

## 6. Infer (`Inferir`)

Load a trained `.pth`, adjust patch size/overlap/batch/device, and click **Run inference** (`Ejecutar inferencia`) to classify the currently loaded cloud — results apply directly to the canvas as annotations you can review and correct with the labeling tools. You can also pick a single point-cloud file directly from this panel (no need to have it already open in the canvas), independent of batch mode.

**Batch inference.** Pick an input folder; every supported point cloud file inside it is processed with the same model and settings, unattended, and written to an output folder — useful for production runs over many files without repeating the single-file flow each time.

---

## Usability features

- **Keyboard shortcuts reference** — `Ayuda → Atajos de teclado` (Help → Keyboard Shortcuts, `F1`) in the header, or see the table above and in [the FAQ](faq.md).
- **Window layout persistence** — window size/position and the main panel split are remembered between sessions.
- **Crash / error logs** — written to `~/.geoannotate3d/logs/crash.log`. If the app closed unexpectedly last time, you'll see a notice on the next launch with a button to open that folder — attach `crash.log` when reporting a bug.
- **Project files** (`.geoa3d`) — a ZIP containing `meta.json` (class schema, stats, CRS) and `labels.npy` (per-point labels), so a project is fully portable as a single file (it does not embed the point cloud itself — keep the original `.las`/`.laz` alongside it).

See also: [Quick Start](quickstart.md) · [Tutorial](tutorial.md) · [FAQ](faq.md)
