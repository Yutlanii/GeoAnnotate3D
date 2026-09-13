# Changelog

All notable changes to GeoAnnotate3D are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **Caja de recorte interactiva (clip box)** — a draggable 3D box widget that
  isolates a volume of the point cloud for inspection or cleanup; purely
  visual (no data is modified), toggle in the tool panel's Visualización
  section.
- **Etiquetas y medidas persistentes en 3D** — text labels and measurements
  now stay anchored in the scene and are saved with the project, instead of
  disappearing as soon as you measured something else. New "Etiqueta 3D"
  tool (key `N`) places a text label; `Ctrl`+click an existing one to edit
  its text, font size, and color.
- **Ajustar plano (RANSAC)** tool (key `P`) — fits the dominant plane inside
  a search radius and labels only the points that are actually flat, leaving
  non-planar geometry in the same radius untouched. Useful for roofs, walls,
  and local ground patches in one click.
- **Polilínea** tool (key `K`) — digitizes an open, multi-vertex line
  (`Ctrl`+click adds a vertex, `Enter` saves it, `Backspace` removes the last
  vertex, `Escape` cancels) for centerlines — roads, power lines, talus edges
  — that a 2-point measurement or a closed polygon can't represent. Color and
  line width are adjustable before creating one, and `Ctrl`+click an existing
  polyline to edit its color/width or delete it. Persistent, saved with the
  project like the other 3D markers.
- **Perfil** tool (key `O`) — traces a line and opens a 2D vertical
  cross-section of the strip of cloud around it (distance along the line
  vs. elevation, colored by classification when available, with a legend of
  the classes actually present), with adjustable strip width, vertical
  exaggeration, a hover readout of the nearest point's distance/elevation,
  and a CSV export button. For inspecting slopes, power lines, or road
  sections without rotating the 3D camera to an awkward angle.
- **Suavizado de etiquetas por mayoría de vecinos** — post-processing pass
  (Pre-clasificar panel) that cleans up noisy label boundaries by having each
  labeled point adopt the majority class among its k nearest neighbors.
  Never touches unlabeled or deleted points.
- **Detección de clusters aislados (QA)** — reports small, spatially
  disconnected groups of a class that are likely annotation mistakes (e.g. a
  stray click), without modifying anything.
- **Soporte real de formato COPC** (Cloud Optimized Point Cloud) —
  `.copc.laz` files are now recognized and labeled as such in the file
  info, and load correctly end-to-end. (An earlier attempt assumed a COPC
  file could be read through the same code path as a regular LAZ file
  since it's technically valid LAZ 1.4 — verified false against a real
  COPC sample: neither `laspy`'s streaming reader nor a full `laspy.read()`
  can decode a COPC file's point data with any backend, since COPC
  organizes points by octree rather than sequentially. Fixed by routing
  COPC files through `laspy.CopcReader` instead.)
- **`sample_data/`** — a small (~80 MB, 10.6M points), real, manually
  classified aerial LiDAR point cloud (Autzen Stadium, CC BY 4.0, see
  `sample_data/README.md` for attribution) to try the app immediately
  after cloning, without needing your own LiDAR data. Also doubles as a
  real COPC file for testing.
- **`sample_data/isprs_ground_filter/`** — 15 more small, real, labeled
  scenes (384,955 points total, binary ground/non-ground classification)
  from the classic ISPRS filter-comparison benchmark, CC BY 4.0. Good for
  testing the CSF ground-detection tool against a published reference,
  and as a second dataset to try the incremental/append-safe export
  feature (below) against.
- Parallel LAZ decompression: LAZ files are now decoded using `lazrs`'s
  multi-threaded backend explicitly, instead of relying on whichever backend
  laspy happens to default to.
- **"Crear todas como clases nuevas"** button in the classification-import
  dialog (shown when a loaded cloud already carries a classification) — adds
  every detected class code as a new project class in one action, instead of
  mapping each one by hand via the per-row "++ Crear clase nueva" option.
  Useful for files with many vendor-specific codes (e.g. ASPRS's reserved
  64-255 range) that GeoAnnotate3D can't name automatically.
- **Exportación incremental a un mismo dataset** — exporting to a folder
  that already holds a previous export (from a different point cloud) now
  adds to it instead of overwriting it: tile files are tagged with their
  source cloud so they never collide, and `dataset.json` merges tile counts,
  per-class counts, and class weights across every cloud exported into that
  folder, tracking all of their source files. Lets you build one larger
  training dataset by annotating and exporting several point clouds one at
  a time into the same output folder.
- **Fine-tuning desde checkpoint externo** (Paso 5 — Entrenar) — starts a
  new training run seeded with another checkpoint's weights (matched by
  layer name and shape; whatever doesn't match, typically the final
  classification layer when the class count differs, is left randomly
  initialized), instead of always starting from scratch. Distinct from
  "Reanudar", which continues the exact same run (optimizer/scheduler state
  included) and requires an identical architecture.
- **Exportar a ONNX** (Paso 5 — Entrenar) — exports an already-trained
  checkpoint to ONNX format for inference outside the app. RandLA-Net's
  random point-subsampling (`torch.randperm`, which ONNX has no equivalent
  operator for) is swapped for a fixed subsample for the duration of the
  export; PointNet++ and KPConv don't need this. Requires the `onnx` Python
  package.

### Changed
- **Panel de herramientas (derecha)** — redesigned: tool icons are a bit
  smaller by default and now resize responsively (both icon size and column
  count) as you resize the panel, so tools never scroll out of view in a
  narrow panel or leave dead space in a wide one. The panel can now be
  widened further than before. The whole panel content also scrolls when a
  tool's options don't fit the available height, instead of the layout being
  crushed together.
- **Panel de tiles (navegación de nube)** — the view-mode, camera, and grid
  edit-mode buttons now show text labels instead of icon-only buttons, so
  it's clear at a glance how to return to the overview, switch to maximum
  density, set a top-down view, or move/rotate the tile grid.

### Fixed
- **Bug de índices en nubes "heavy" (>150M pts)** — the octree built for
  clouds loaded via `.ga3d_bin`/"cargar como normal" pre-subsampled the
  point array before indexing it, so Pincel/Esfera/Disco could paint the
  wrong points on very large clouds. The octree now always indexes the full
  array; its own internal RAM-aware limits handle huge clouds safely.
- Several Qt font-size rounding mismatches where a label's text was measured
  at one pixel size but rendered at another (fractional CSS `font-size`
  values get rounded differently than `QFont.setPixelSize` truncates),
  causing tool names to render clipped in certain panel widths.
- **Pick tool always showed "desconocido" as the class** — it re-searched
  the picked position against the point cloud with a small fixed-radius
  query, but the position shown is an average of several nearby points
  (for a smooth reading), not the exact coordinate of one real point, so
  the search almost always found nothing. Fixed by resolving the actual
  point under the cursor (and its real global index) directly, and
  reading its classification straight from the current annotation labels
  instead of re-deriving it from coordinates.
- **Perfil got slow when zooming/panning on strips with many points** —
  it drew every point with an individual `drawEllipse()` call, and the
  hover readout re-scanned every point in a Python loop on every mouse
  move. Both are now vectorized (a numpy-written raster buffer for the
  scatter, a single vectorized nearest-point search for hover), so
  zoom/pan stays smooth regardless of strip size.
- **Measurements and polylines could never be deleted once created** —
  `Escape` only cancels a measurement/polyline still in progress, not
  one already saved; there was no way at all to remove an existing one.
  `Ctrl`+click an existing measurement or polyline now opens an edit
  dialog (color/line width, or delete it) — the same capability
  "Etiqueta 3D" already had, which also gained a delete button of its own.
- **A marker's floating text could get hidden behind the point cloud**
  as the camera rotated, or behind other points from elsewhere in the
  scene that happened to project to the same screen area. Two earlier
  attempts in this same unreleased cycle didn't hold up: moving the text
  to a separate always-on-top VTK layer (verified with a real offscreen
  pixel test to not actually work — VTK preserves the depth buffer
  across layers) and, after that, drawing it as a Qt widget stacked on
  top of VTK's native OpenGL window (which broke rendering outright —
  the point cloud stopped appearing at all, and reloading a cloud could
  hang the app — overlaying a translucent Qt widget on a native child
  window isn't safe in general). The fix that actually works: the text
  is a `vtkTextActor` — a 2D actor, not 3D — with its position set in
  world coordinates; VTK reprojects it to screen space every frame using
  the active camera on its own, and 2D actors are composited without
  any depth test against 3D geometry by design, so they're always on
  top. Verified with a real offscreen render: a solid plane placed
  directly in front of a label (in Z) no longer hides it.
- **A stale color-picker row could stay visible on top of another
  tool's options** after switching away from "Etiqueta 3D" or
  "Polilínea" — their color row is added as a nested layout, and the
  cleanup that runs on every tool switch only knew how to hide direct
  child widgets, never widgets living inside a nested layout. Now
  recurses into nested layouts too.

## [1.0.0] — public release preparation

- Complete UI redesign (light theme, navigation rail, real icon set),
  performance improvements across loading/rendering/annotation tools, and
  general polish ahead of the public release.
- Added user guide, tutorial, FAQ, and contributing guide.
- Added a "delete points" tool (removes noise from the cloud, not just its
  label), redesigned onboarding, and a "Confianza" (confidence) color mode
  for reviewing inference results.
- Housekeeping: removed build artifacts and a stale nested README from
  version control; corrected a false claim about PyTorch in the README;
  added a permanent regression-test harness.
