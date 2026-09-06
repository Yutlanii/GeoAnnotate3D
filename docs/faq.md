# Frequently Asked Questions

## Installation & dependencies

**"pip install laspy[lazrs]" and my `.las` still won't open — "read length must be non-negative or -1".**
This specific message usually means the file's header (or an EVLR block near the end of the file) is non-standard or slightly corrupted — it's not actually a missing-backend problem, even though it can look like one. GeoAnnotate3D already retries with several backend/EVLR combinations before giving up; if it still fails, try re-exporting the file with CloudCompare, PDAL, or `lastools` (`lasinfo`/`las2las`) and load the re-exported copy.

**Training/inference says PyTorch isn't installed.**
PyTorch isn't in `requirements.txt` because the correct build depends on your hardware:
```bash
# NVIDIA GPU
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**CSF (ground detection) says the library isn't installed.**
```bash
pip install cloth-simulation-filter
```

**Do I need a GPU?**
No — training and inference both run on CPU, just slower. A CUDA-capable NVIDIA GPU speeds up training substantially (roughly 10-50x depending on the architecture and cloud size) and is picked up automatically if present; the training panel shows which device it's using.

## File formats & data

**Which point cloud formats can I open?**
`.las`, `.laz`, `.e57`, `.ply`, `.pcd`, `.xyz`, `.txt`, `.csv`, `.asc`, `.pts`, `.npy`, and GeoAnnotate3D's own `.ga3d_bin` cache format.

**What does "Usar códigos de clasificación ASPRS estándar" actually do?**
Your class schema is free text (whatever names you gave your classes), so by default the exported classified `.las` just writes your internal class IDs (0, 1, 2…) into the `classification` field — meaningless to other software. With this option checked, class names are matched by keyword (e.g. "suelo"/"ground" → code 2, "edificio"/"building" → code 6) to the ASPRS standard classification codes so the file opens correctly in CloudCompare, QGIS, or ArcGIS. Anything that doesn't match a known keyword gets a code in the 64-255 range, which the LAS 1.4 spec reserves for user-defined classes, so it never collides with a real standard code.

**What's inside a `.geoa3d` project file?**
It's a ZIP containing `meta.json` (class schema, project metadata, stats) and `labels.npy` (one label per point). It does **not** embed the original point cloud — keep the source `.las`/`.laz` next to it, or reference it by path.

## Performance

**My cloud has hundreds of millions of points — will it work?**
Yes, that's the primary design target. Use the tile grid (**Nube** step) to split it into workable chunks instead of loading it all at once; very large clouds automatically switch to an out-of-core (memory-mapped) loading path. The brush and other annotation tools use an octree spatial index specifically so they stay fast on dense tiles.

**Annotation feels slow on a specific tile.**
Compile the native C extension once (`python core/build_extension.py` — needs a C compiler, see the installer instructions) — it gives roughly a 20x speedup on annotation and 3-8x on level-of-detail rendering. The app works without it, just slower.

## Troubleshooting

**The app closed without any error message.**
Check `~/.geoannotate3d/logs/crash.log` — this is where crash details are written even when no dialog is shown (some crashes happen inside native rendering code where showing a dialog would itself be unsafe). On the next launch after an unexpected close, GeoAnnotate3D shows a notice with a button to open that log folder directly. Attach `crash.log` when reporting an issue.

**A keyboard shortcut doesn't seem to do anything.**
Check **Ayuda → Atajos de teclado** (`F1`) for the current, accurate list — it's generated from the same code that handles the shortcuts, so it can't drift out of sync with reality.

**My window layout / panel sizes reset.**
This should no longer happen — window size, position, and the main panel split are saved on close and restored on the next launch. If you're on a build from before this was added, update to the latest version.

## Licensing

**What license is GeoAnnotate3D under?**
GPLv3 — see [LICENSE](../LICENSE). This choice is tied to PyQt5, which GeoAnnotate3D's interface is built on: PyQt5 is dual-licensed (GPLv3 or a paid Riverbank commercial license), so an application built on it that isn't itself GPL-compatible would need a commercial PyQt5 license instead.

**Can I use GeoAnnotate3D or datasets it produces commercially?**
The datasets and models you produce with it are yours. The application's *source code* is GPLv3, which affects redistributing modified versions of the app itself, not the outputs (labeled data, trained model weights) you create with it. If you have a specific commercial redistribution question, open an issue.

---

Still stuck? [Open an issue](https://github.com/Yutlanii/GeoAnnotate3D/issues) with your OS, Python version, the exact error, and (if relevant) `crash.log`.
