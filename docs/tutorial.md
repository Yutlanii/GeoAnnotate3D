# Step-by-Step Tutorial

This walks through the complete workflow once, end to end, using a LiDAR tile of your own (any `.las`/`.laz` file with a few million points works well for a first run — very large files just mean more tiles in step 2). If you haven't installed GeoAnnotate3D yet, start with the [Quick Start Guide](quickstart.md).

## Goal

By the end you'll have: a cloud pre-classified automatically, a handful of classes hand-labeled with corrections, an exported training dataset, a small trained model, and inference results applied back onto the cloud.

## Step 1 — Load the cloud

1. Launch GeoAnnotate3D (`python main.py`, or the installed shortcut).
2. Dismiss or read the welcome dialog (you can reopen it later from **Ayuda → Bienvenida / Tutorial**).
3. `Proyecto → Nuevo proyecto` (`Ctrl+N`) and pick your `.las`/`.laz` file.
4. Watch the header bar: once loading finishes it shows the filename, point count, and detected CRS.

If the cloud is small enough to navigate comfortably as a whole, you can skip straight to Step 3. Otherwise, continue to Step 2.

## Step 2 — Set up the tile grid

1. You're on the **Nube** step, in **Overview**. Set **Tamaño de tile** to something that gives you a manageable working area — 50m × 50m is a reasonable starting point for typical LiDAR density.
2. If the grid doesn't line up with your cloud's footprint, drag it into place or use the **TRANSFORM** section's Offset X/Y and Rotación fields.
3. Click any tile in the grid to enter **tile mode** — the canvas now shows just that tile at full point density. This is where you'll do the actual work; use `Escape` or **← Vista global** to go back to Overview and pick a different tile.

## Step 3 — Pre-classify automatically

Move to **Pre-clasificar** (click it in the left rail, or use `Ctrl` + the step number if you've set that up — otherwise just click).

1. **Suelo por simulación de tela (CSF)**: pick your "ground" class in the dropdown, leave the default resolution (0.5m) for a first try, and click **Detectar suelo (CSF)**. Watch the canvas — ground points should recolor to that class.
2. **Clasificar por altura (AGL)**: click **Configurar rangos…**. You'll see default ranges already guessed from your class names (e.g. a class named "Suelo" or "Ground" gets 0-0.3m). Adjust them to match your data, or add a new range with **+ Agregar clase**. Close the dialog with **Guardar clasificador**, then click **Clasificar por capas AGL** in the main panel to apply.

At this point a good chunk of the cloud should already be classified — ground, low/medium/high vegetation, whatever your ranges captured. Check a few tiles visually before moving on.

## Step 4 — Hand-label the rest

Move to **Etiquetar**.

1. Press `1` (or click the first class row) to make it active — the "PINTANDO CON" indicator at the top of the class panel shows your current color.
2. Press `B` for the brush tool. In the right panel, set a radius that fits the feature size you're labeling (buildings need a bigger brush than power lines).
3. `Ctrl`+drag over a misclassified area to relabel it. If you overshoot, press `E` to toggle erase mode on the same tool, or `Ctrl+Z` to undo.
4. For clean geometric shapes (a building footprint, say), the **Caja** (`X`) or **Polígono** (`L`) tools are often faster than the brush.
5. If you have a georeferenced orthophoto or a vector layer (parcel boundaries, road centerlines) for this area, load it from **Capas de referencia** — seeing the labels overlaid on real imagery makes edge cases much easier to call correctly.
6. Watch the **COBERTURA TOTAL** bar at the bottom of the class panel — it's your progress indicator for this tile.

Repeat steps 3-4 across a few tiles before exporting — the more representative tiles you label, the better the model in Step 6.

## Step 5 — Export a training dataset

Move to **Exportar**.

1. Pick **RandLA-Net** (a reasonable default for a first model — fast to train, good on large outdoor scenes).
2. Leave **También exportar formato universal** checked if you might want to try a different architecture later without re-exporting.
3. Check **Exportar nube clasificada (.las)** and **Usar códigos de clasificación ASPRS estándar** if you want to open the result in CloudCompare/QGIS afterward to sanity-check it.
4. Click **Exportar dataset** and choose an output folder. You'll get a folder with `train/`, `val/`, `test/` splits, a `custom_dataset.py`, and a `README.md` with the exact PyTorch training command.

## Step 6 — Train a model

Move to **Entrenar**.

1. **Dataset**: point it at the folder from Step 5.
2. Leave **Arquitectura** on RandLA-Net to match what you exported.
3. Start with a small **Epochs** value (10-20) for your first run just to confirm the pipeline works end to end — you can always resume and train longer later.
4. Click **Iniciar entrenamiento** and watch the loss/mIoU chart. If you need to stop, the last good checkpoint is saved automatically (`best_model.pth` in your chosen output folder) — pick it up again later from the **Reanudar** field.
5. When it finishes, open `class_report.txt` next to your model — it tells you, per class, whether you have enough labeled examples (low support + low IoU together usually means "label more of this class").

## Step 7 — Run inference and correct

Move to **Inferir**.

1. Load the `.pth` you just trained.
2. Click **Ejecutar inferencia** on the currently loaded cloud (or a different one loaded via **Nube**) — predictions apply directly to the canvas.
3. Switch back to **Etiquetar** and fix whatever the model got wrong, the same way you labeled originally.
4. Re-export (Step 5) with the corrected labels and train again (Step 6, resuming from the previous checkpoint) — each cycle should need fewer manual corrections.

If you have many files to classify with an already-trained model, use **Ejecutar por lote** in the Inferir panel instead of repeating steps 1-2 per file.

## Next steps

- [Complete User Manual](user-guide.md) for every option in detail
- [FAQ](faq.md) for common issues (LAZ loading errors, missing dependencies, performance tuning)
