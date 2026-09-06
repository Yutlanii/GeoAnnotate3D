# Quick Start Guide

Get GeoAnnotate3D running and label your first points in a few minutes.

## 1. Install

**Option A — Windows installer (recommended for most users)**

Download `GeoAnnotate3D_Setup_v1.0.0.exe` from the [Releases page](https://github.com/Yutlanii/GeoAnnotate3D/releases/latest) and run it. It bundles everything you need; skip to [step 2](#2-launch-the-app).

**Option B — From source (Windows/Linux/macOS, for developers)**

```bash
git clone https://github.com/Yutlanii/GeoAnnotate3D.git
cd GeoAnnotate3D/geoannotate3d

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt

# Optional but recommended: compile the native C extension
# (20x faster annotation, 3-8x faster LOD — the app runs fine without
# it too, just slower on very dense clouds)
python core/build_extension.py

python main.py
```

Training and inference need PyTorch, which is **not** in `requirements.txt` because the right build depends on your GPU:

```bash
# NVIDIA GPU (CUDA 12.x)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

CSF (ground detection) is also optional:

```bash
pip install cloth-simulation-filter
```

## 2. Launch the app

```bash
python main.py
# or open a cloud directly:
python main.py path/to/cloud.laz
```

On first launch you'll see a welcome/tutorial dialog describing the 6-step workflow. You can reopen it any time from **Ayuda → Bienvenida / Tutorial** in the header bar.

## 3. Load a point cloud

From the **Nube** step (the first item in the left rail): **Proyecto → Nuevo proyecto** (`Ctrl+N`), then pick a `.las`, `.laz`, `.e57`, `.ply`, or `.npy` file.

- If the cloud has RGB data, it opens in RGB color mode automatically; otherwise it opens in elevation mode.
- For clouds bigger than roughly 20-30M points, the **Overview** panel lets you configure a tile grid (drag/rotate/resize it over the footprint) so you work tile-by-tile instead of loading everything into the 3D view at once.

## 4. Pre-classify (optional but recommended)

Move to the **Pre-clasificar** step. Two independent tools help you avoid hand-labeling the obvious classes:

- **CSF (Cloth Simulation Filter)** — detects the ground automatically.
- **Clasificar por altura (AGL)** — assigns a class to each height-above-ground range (e.g. 0-0.3m → Ground, 0.3-2m → Low vegetation...). Click **Configurar rangos…** to edit those ranges in a table; adjusting one boundary automatically keeps neighboring ranges continuous.

## 5. Label manually

In the **Etiquetar** step, pick a class (`1`-`9` or click it in the **CLASES** list) and a tool from the right panel: Pincel (brush, `B`), Disco (`D`), Region Growing (`G`), Polígono (`L`), Caja (`X`), Esfera (`H`), Corte Z (`C`), Pick (`I`), Medir (`M`). `Ctrl+Z` / `Ctrl+Shift+Z` undo/redo. See **Ayuda → Atajos de teclado** (`F1`) for the full list.

## 6. Export a dataset

In **Exportar**, pick a target architecture (RandLA-Net, PointNet++, or KPConv) — the export format is chosen automatically — and click **Exportar dataset**. You get a ready-to-train folder with an automatic spatial train/val/test split, plus an optional classified `.las` (with standard ASPRS codes if you check that box) for viewing in CloudCompare/QGIS.

## 7. Train and infer

**Entrenar**: point it at the exported dataset folder, pick an architecture, and start training — it shows live loss/mIoU curves and a per-class report at the end. Training can be resumed from any saved checkpoint if interrupted.

**Inferir**: load a trained `.pth` model and run it on the current cloud, or use **batch inference** to process a whole folder of point clouds unattended.

## Where to go next

- [Complete User Manual](user-guide.md) — every panel and tool explained
- [Step-by-step Tutorial](tutorial.md) — a full worked example end to end
- [FAQ](faq.md) — common questions and troubleshooting
