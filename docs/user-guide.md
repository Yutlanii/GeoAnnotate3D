# GeoAnnotate3D — Complete User Manual

GeoAnnotate3D organizes work as a 6-step workflow, shown as a vertical rail on the left of the window: **Nube → Pre-clasificar → Etiquetar → Exportar → Entrenar → Inferir**. Below **Inferir** the rail also has a **Capas de referencia** entry, which is not a workflow step but an auxiliary view (overlay vector/raster layers) available at any point.

The header bar (top) shows the app name, the loaded file and point count, its CRS, whether you're viewing the whole cloud (Overview) or a single tile, a **Proyecto** menu (new/open/save/recent), and an **Ayuda** menu (shortcuts reference and this documentation's entry points).

---

## 1. Nube (Point Cloud)

**Loading.** `Proyecto → Nuevo proyecto` (`Ctrl+N`) or drag-and-drop a file onto the window. Supported formats: `.las`, `.laz`, `.e57`, `.ply`, `.pcd`, `.xyz`, `.txt`, `.csv`, `.asc`, `.pts`, `.npy`, plus GeoAnnotate3D's own `.ga3d_bin` cache format for very large clouds.

- If the file has RGB, it opens in **RGB** color mode; otherwise it opens colored by **elevation**. You can switch color mode from the tool panel's **VISUALIZACIÓN** section at any time (RGB / Elevation / Intensity / Annotation).
- `.laz` files need a LAZ decompression backend (`lazrs` or `laszip`, both installable via `pip install "laspy[lazrs]"`). If a file fails to open with an error mentioning EVLR or header length, the file's header itself may be non-standard — try re-exporting it with CloudCompare, PDAL, or `lastools` (`lasinfo`/`las2las`) and reload.

**Tiles.** For clouds too dense to comfortably navigate as a whole, the **Overview** panel lets you configure a tile grid:

- **Tamaño de tile** — width/height of each tile in meters.
- The grid preview shows one cell per tile; percentages indicate how much of that tile is already labeled.
- **TRANSFORM** — drag to move the grid, or use the Offset X/Y and Rotación fields to align it precisely with the cloud's footprint (useful when the cloud isn't axis-aligned).
- Click a tile to enter **tile mode** (loads just that tile at full density); `Escape` or the **← Vista global** button returns to Overview.
- Filters below the grid (**Sin anotar / Parcial / Completo**) let you jump straight to tiles that still need work.
- Hover any tile (even at small grid-cell sizes, where inline text doesn't fit) to see a tooltip with its coordinates, labeled percentage, and whether it's the currently active tile.

### What is `.ga3d_bin`, and why does it matter?

`.ga3d_bin` ("GA3D-Bin") is GeoAnnotate3D's own binary point-cloud format — not a competing standard, but a fast **cache** the app creates automatically next to a `.las`/`.laz`/`.e57` file once it crosses **200 million points**.

The difference from LAS/LAZ/E57 is structural, not just "a different extension":

- **LAS/LAZ/E57** are designed for interchange between many tools, so every read has to parse a structured record format (and, for `.laz`, decompress it first) before you get usable coordinates.
- **GA3D-Bin** is a flat binary layout — a small 512-byte header followed by raw `float32` XYZ (and other attributes) with no compression and no quantization. That means it can be opened with **memory-mapping (mmap)** instead of reading the whole file into RAM: the OS pages data in from disk on demand, so you can work with a cloud far larger than your available memory, and reopening it is essentially instant because there's no decompression or parsing step to redo.

You don't need to manage this yourself: the first time you open a huge `.las`/`.laz`/`.e57`, GeoAnnotate3D offers to convert it once; from then on, opening that *same source file* again detects the existing `.ga3d_bin` and asks whether to use it (much faster than reconverting or re-parsing the original). You can also open a `.ga3d_bin` file directly. It lives alongside the original file (same name, `.ga3d_bin` extension) and is safe to delete — GeoAnnotate3D will just offer to regenerate it from the source the next time you open that file.

---

## 2. Pre-clasificar

Two independent, automatic methods to reduce manual labeling. Neither depends on the other.

**Clasificar por altura (AGL)** — classifies points by height above the lowest point of the cloud (a "flat" AGL model; it doesn't need a terrain surface). Click **Configurar rangos…** to open the range editor: each row is a class with a **Desde (m)** / **Hasta (m)** range. Editing one boundary automatically slides the adjacent range's boundary to match, so ranges stay continuous with no gaps — and deleting a class re-closes the gap it leaves, keeping the remaining ranges adjacent. Click **Clasificar por capas AGL** to apply.

**Suelo por simulación de tela (CSF)** — Cloth Simulation Filter, the same ground-detection algorithm used by CloudCompare: simulates a cloth falling onto the inverted point cloud to find the ground surface. Requires `pip install cloth-simulation-filter`. Pick the target class and cloth resolution (smaller = more detail, slower), then **Detectar suelo (CSF)**.

---

## 3. Etiquetar (Labeling)

**Classes** (left panel): each class is a numbered, colored circle — press `1`-`9` or click a row to make it the active ("painting with") class. Double-click a row to rename it or change its color. The pencil icon next to **CLASES** opens the full class manager (add/remove classes, bulk edit, visibility filter). The bottom of the panel shows overall labeled coverage.

**Tools** (right panel), each with a single-key shortcut and its own options section:

| Tool | Key | Behavior |
|---|---|---|
| Pincel | `B` | Spherical brush; `Ctrl`+drag paints a continuous stroke. Radius, stroke density, and disc thickness are adjustable; scroll wheel resizes it live. |
| Disco | `D` | A 3D disc oriented to the local surface; `Ctrl`+click places it. |
| Region Growing | `G` | `Ctrl`+click a seed point; grows outward through connected, similarly-colored geometry. |
| Polígono | `L` | Draw a 2D/3D polygon; everything inside gets the active class. |
| Caja | `X` | Box selection. |
| Esfera | `H` | `Ctrl`+click for an instant spherical selection (vs. Pincel's continuous stroke). |
| Corte Z | `C` | Slice/cross-section selection by height. |
| Pick | `I` | Sample the class of the point under the cursor (to check what's already labeled). |
| Medir | `M` | Measure distances in the scene. |

`E` toggles **erase mode** on the active tool — clears the class back to unlabeled (0), it does not remove the point. `Supr` (Delete) toggles **delete mode** instead — mutually exclusive with erase mode — which removes the selected points from the cloud entirely (useful for cleaning up sensor noise or stray points); this works with *any* of the selection tools above, not just one dedicated tool. Deleted points are excluded from every export and are undoable like any other annotation (`Ctrl+Z`). `Ctrl+Z` / `Ctrl+Shift+Z` undo/redo. `Ctrl+clic` on the canvas is the common "commit" gesture for the click-based tools.

**Reference layers** (rail entry below Inferir, or accessible while labeling): overlay georeferenced orthomosaics (`.tif`) or vector layers (`.shp`, `.geojson`, `.gpkg`, `.kml`, `.dxf`) with real-world coordinates to guide labeling. Lower the point cloud's own opacity from this panel to see an underlying raster more clearly, and use **Vista cenital** to align the camera top-down.

---

## 4. Exportar

Pick one or more target architectures — **RandLA-Net**, **PointNet++**, **KPConv** — the export format for each is chosen automatically (you don't pick a file format separately). Options:

- **También exportar formato universal** — a `.npy`+`json` export usable by any architecture without conversion.
- **Exportar nube clasificada (.las)** — a classified `.las` for viewing the result in CloudCompare/QGIS/ArcGIS. Check **Usar códigos de clasificación ASPRS estándar en el .las** to remap your custom class IDs to the standard ASPRS codes (ground=2, low/medium/high vegetation=3/4/5, building=6, water=9, rail=10, road=11, bridge=17, unclassified=1) via keyword matching on class names; anything unrecognized is written into the 64-255 range the LAS 1.4 spec reserves for user-defined codes, so it never collides with a real standard code.
- **Solo puntos etiquetados** — excludes unlabeled points from the dataset.

The split between train/val/test is spatial (separate geographic blocks) to avoid data leakage between splits.

---

## 5. Entrenar

Three architectures are implemented directly in the app (no external training framework needed): **RandLA-Net** (fast, large clouds), **PointNet++** (multi-scale accuracy), **KPConv** (fine edge detail). Point **Dataset** at the exported data folder, set epochs/batch size/points-per-sample/learning rate, and **Iniciar entrenamiento**. Loss and mIoU update live in the chart.

**Resuming a training run.** Use the **Reanudar** row to pick a previously saved checkpoint (`.pth`) — training continues from that epoch with the optimizer and learning-rate scheduler state restored, not just the model weights, so it behaves as if it had never stopped. Safe to close the app mid-run and resume later.

**Per-class report.** When training finishes, alongside `best_model.pth` you get `class_report.json` and `class_report.txt` in the output folder: precision, recall, and IoU for *each* class individually (not just the aggregate mIoU), plus how many points supported each class — useful for spotting which classes need more labeled data.

---

## 6. Inferir

Load a trained `.pth`, adjust patch size/overlap/batch/device, and **Ejecutar inferencia** to classify the currently loaded cloud — results apply directly to the canvas as annotations you can review and correct with the labeling tools.

**Batch inference.** Pick an input folder; every supported point cloud file inside it is processed with the same model and settings, unattended, and written to an output folder — useful for production runs over many files without repeating the single-file flow each time.

---

## Usability features

- **Keyboard shortcuts reference** — `Ayuda → Atajos de teclado` (`F1`) in the header, or see the table above and in [the FAQ](faq.md).
- **Window layout persistence** — window size/position and the main panel split are remembered between sessions.
- **Crash / error logs** — written to `~/.geoannotate3d/logs/crash.log`. If the app closed unexpectedly last time, you'll see a notice on the next launch with a button to open that folder — attach `crash.log` when reporting a bug.
- **Project files** (`.geoa3d`) — a ZIP containing `meta.json` (class schema, stats, CRS) and `labels.npy` (per-point labels), so a project is fully portable as a single file (it does not embed the point cloud itself — keep the original `.las`/`.laz` alongside it).

See also: [Quick Start](quickstart.md) · [Tutorial](tutorial.md) · [FAQ](faq.md)
