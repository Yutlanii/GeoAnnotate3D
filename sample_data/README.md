# Sample data

Two real, labeled point clouds to try GeoAnnotate3D without needing your
own LiDAR data:

- **`autzen-classified.copc.laz`** (this folder, see below) — one large
  scene, rich 21-category classification, COPC format.
- **[`isprs_ground_filter/`](isprs_ground_filter/README.md)** — 15 small
  scenes, binary ground/non-ground classification, good for testing the
  CSF ground-filter tool against a published benchmark.

Both are CC BY 4.0 (attribution required). See each folder's own README
for full details and attribution text.

## Autzen Stadium (classified)

`autzen-classified.copc.laz` — a real, manually-labeled aerial LiDAR point
cloud of Autzen Stadium (Eugene, Oregon): **10,653,336 points**, 21
hand-classified object categories (ground, vegetation, building, water,
transmission tower, bridge deck, wire, car, truck, fence, and more), full
RGB color, LAS 1.4 point format 7, in **COPC** (Cloud Optimized Point
Cloud) form.

Use it to try GeoAnnotate3D immediately after cloning, without needing your
own LiDAR data:

1. `Proyecto → Nuevo proyecto` and pick this file (or drag it onto the window).
2. It loads with real classification already present — GeoAnnotate3D will
   offer to import it directly as labels (auto-matched to a default class
   schema by keyword), so you can see a fully-labeled cloud right away.
3. It's a real COPC file, so opening it also exercises GeoAnnotate3D's COPC
   support end-to-end.
4. It's small enough (~80 MB, ~10.6M points) to export and run a real,
   if brief, training pass with the built-in RandLA-Net/PointNet++/KPConv
   trainer (Paso 5 — Entrenar) to see the full workflow work end-to-end.

## Origin & license

- Original scan ("autzen.laz"): provided by Aaron Reyna, Watershed Sciences,
  Inc. (2010), originally distributed for libLAS testing.
- Manual classification into 21 categories: Max Sampson, Hobu, Inc. (2021).
- Converted to COPC form as part of the [PDAL/data](https://github.com/PDAL/data)
  repository, which distributes it under a **CC BY 4.0** license
  (attribution required, redistribution and reuse permitted, no
  non-commercial restriction).
- Source: <https://github.com/PDAL/data/tree/main/autzen>

If you redistribute this file elsewhere, keep this attribution.

## Classification codes actually present in this file

The full Hobu/Sampson scheme defines 21 categories; only some appear in
this particular scene. Codes 0–19 follow the standard ASPRS LAS
classification table (which GeoAnnotate3D's importer already recognizes
by name); codes ≥64 are scheme-specific extensions Hobu/Sampson defined
for this dataset (ASPRS reserves 64–255 as vendor/project-defined, so
they don't have a universal meaning — GeoAnnotate3D's importer shows
them as generic "Clase 64" etc. because it can't know their meaning for
an arbitrary file; here's what they mean *for this specific file*):

| Code | Name | Points in this file |
|---|---|---|
| 0 | Sin clasificar / unclassified | 390,708 |
| 2 | Suelo / Ground | 6,909,405 |
| 5 | Vegetación / Vegetation | 2,613,807 |
| 6 | Edificio / Building | 343,181 |
| 9 | Agua / Water | 355,339 |
| 15 | Torre de transmisión / Transmission Tower | 51 |
| 17 | Tablero de puente / Bridge Deck | 6,790 |
| 19 | Estructura elevada / Overhead Structure | 3,605 |
| 64 | Cable / Wire | 4,224 |
| 65 | Auto / Car | 18,774 |
| 66 | Camión / Truck | 92 |
| 68 | Barrera / Barrier | 1,233 |
| 73 | Cerca / Fence | 5,882 |
| 76 | Silo / Storage Tank | 221 |
| 77 | Estructura de puente / Bridge Structure | 24 |

When GeoAnnotate3D offers to import this classification, use the
**"Crear todas como clases nuevas"** button in that dialog to bring in
all of them at once as new project classes (named generically, e.g.
"Clase 64") instead of mapping each of the ~15 present codes to an
existing class by hand — then rename them using the table above if you
want the real names.

## A note on file size

At ~80 MB, this file is within GitHub's per-file limits, but it's still a
binary asset that will live forever in the repository's git history once
committed to the normal source tree. If repo size becomes a concern, the
common alternatives are to track it with **Git LFS**, or attach it as a
binary asset on a **GitHub Release** instead of committing it directly —
both keep a fresh `git clone` small. Decide based on how you want to manage
the repo; this file works identically either way once downloaded.
