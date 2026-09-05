"""
annotation/exporter.py — Export plug-and-play para redes neuronales 3D
=======================================================================

Para cada red neuronal genera:
  - Archivos de datos en el formato EXACTO que espera la red
  - custom_dataset.py  listo para copiar al repositorio de la red
  - train.py           ejecutar directamente con "python train.py"
  - README.md          instrucciones paso a paso
  - requirements.txt   dependencias exactas

Redes soportadas:
  RandLA-Net   (QingyongHu/RandLA-Net)
  PointNet++   (yanx27/Pointnet_Pointnet2_pytorch)
  KPConv       (HuguesThomas/KPConv-PyTorch)

Cada TILE anotado = una ESCENA para la red neuronal.
Split espacial: bloques geográficos separados (sin data leakage).

Uso:
    worker = ExportWorker(pc, project, label_store, config)
    worker.progress.connect(...)
    worker.finished.connect(...)
    worker.start()
"""
from __future__ import annotations
import json, textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExportConfig:
    output_dir:         str   = ""
    architectures:      List[str] = field(default_factory=lambda: ["randlanet"])
    split_strategy:     str   = "spatial"
    train_ratio:        float = 0.70
    val_ratio:          float = 0.20
    export_las:         bool  = True    # también .las clasificado para GIS
    only_labeled:       bool  = True    # excluir puntos sin etiquetar
    asprs_codes:        bool  = False   # mapear a códigos ASPRS estándar (ver _asprs_code_for)


# ─────────────────────────────────────────────────────────────────────────────
# Mapeo a códigos de clasificación ASPRS estándar (LAS 1.4)
# ─────────────────────────────────────────────────────────────────────────────
# Antes, el .las clasificado exportado escribía directamente los IDs
# internos del schema (0,1,2,3…) en el campo `classification` — como esos
# IDs son arbitrarios (definidos por el usuario, sin relación con ningún
# estándar), abrir ese .las en CloudCompare/QGIS/ArcGIS mostraba clases
# sin sentido o que colisionaban con su propia leyenda ASPRS. Con
# `asprs_codes=True`, cada clase se mapea por nombre (heurística de
# palabras clave, igual que ya se usa en geo_panel.py para CSF/AGL) a su
# código ASPRS estándar; lo que no se reconoce cae en el rango 64-255,
# reservado por la especificación para uso definido por el usuario, en
# vez de colisionar con un código estándar real.
_ASPRS_KEYWORDS = [
    (2,  ["suelo", "ground", "terreno", "tierra", "floor"]),
    (3,  ["veg. baja", "veg baja", "low veg", "hierba", "pasto", "grass"]),
    (4,  ["veg. media", "veg media", "medium veg", "arbusto", "shrub", "mid veg"]),
    (5,  ["veg. alta", "veg alta", "high veg", "arbol", "árbol", "tree", "forest", "bosque", "canopy"]),
    (6,  ["edificio", "building", "construcc", "techo", "roof"]),
    (9,  ["agua", "water", "rio", "río", "lago", "mar", "lake", "river"]),
    (10, ["riel", "rail", "ferrocarril", "railway"]),
    (11, ["carretera", "road", "calle", "pavimento", "pavement", "asfalto", "asphalt"]),
    (17, ["puente", "bridge"]),
    (1,  ["sin clasificar", "unclassified", "no clasificar", "ninguna"]),
]


def _asprs_code_for(class_id: int, class_name: str) -> int:
    """Heurística nombre→código ASPRS; sin coincidencia → 64+class_id
    (rango 64-255, reservado para uso definido por el usuario en LAS 1.4),
    para no colisionar nunca con un código ASPRS estándar real."""
    name = (class_name or "").lower()
    for code, keywords in _ASPRS_KEYWORDS:
        if any(kw in name for kw in keywords):
            return code
    return min(255, 64 + max(0, class_id))

    # compatibilidad con código antiguo
    @property
    def format(self): return "npy"
    @property
    def architecture(self): return self.architectures[0] if self.architectures else "randlanet"
    @property
    def include_weights(self): return True


@dataclass
class SplitResult:
    train_tiles: List[int] = field(default_factory=list)  # tile IDs
    val_tiles:   List[int] = field(default_factory=list)
    test_tiles:  List[int] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Worker principal
# ─────────────────────────────────────────────────────────────────────────────

class ExportWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, pc, project, label_store, config: ExportConfig,
                 tile_manager=None, parent=None):
        super().__init__(parent)
        self._pc      = pc
        self._project = project
        self._labels  = np.array(project.labels) if project.labels is not None else None
        self._config  = config
        self._tm      = tile_manager
        self._ls      = label_store

    def run(self):
        try:
            cfg = self._config
            out = Path(cfg.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            self.progress.emit(5, "Analizando tiles anotados…")
            tiles_data = self._collect_tile_data()
            if not tiles_data:
                self.error.emit("No hay tiles con puntos etiquetados para exportar.\n"
                                "Anota al menos 3 tiles antes de exportar.")
                return

            self.progress.emit(15, "Calculando split espacial…")
            split = self._spatial_split(tiles_data)

            # Exportar para cada arquitectura seleccionada
            total_archs = len(cfg.architectures)
            for arch_i, arch in enumerate(cfg.architectures):
                base_pct = 20 + int(60 * arch_i / total_archs)
                end_pct  = 20 + int(60 * (arch_i + 1) / total_archs)
                arch_dir = out / arch
                arch_dir.mkdir(exist_ok=True)

                self.progress.emit(base_pct, f"Exportando {arch}…")

                if arch == "randlanet":
                    self._export_randlanet(tiles_data, split, arch_dir, base_pct, end_pct)
                elif arch == "pointnetpp":
                    self._export_pointnetpp(tiles_data, split, arch_dir, base_pct, end_pct)
                elif arch == "kpconv":
                    self._export_kpconv(tiles_data, split, arch_dir, base_pct, end_pct)

            if cfg.export_las:
                self.progress.emit(82, "Exportando nube clasificada .las…")
                self._export_classified_las(out)

            self.progress.emit(92, "Escribiendo metadatos…")
            self._write_global_metadata(tiles_data, split, out)

            self.progress.emit(100, f"✓ Exportación completa — {out}")
            self.finished.emit(str(out))

        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")

    # ─────────────────────────────────────────────────────────────────────────
    # Recolección de datos por tile
    # ─────────────────────────────────────────────────────────────────────────

    def _collect_tile_data(self) -> List[dict]:
        """
        Por cada tile anotado extrae:
          xyz_local  (N, 3) float32 — centrado en el tile
          features   (N, 9) float32 — x,y,z,i,r,g,b,xnorm,ynorm
          labels     (N,)   uint8
          center_xy  (2,)   float64 — para el split espacial
          tile_id    str             — "tile_col_row"
        """
        pc     = self._pc
        labels = self._labels
        tiles  = []

        if self._tm is None:
            # Sin tile manager: dividir la nube en 3 segmentos espaciales (train/val/test)
            mask = labels > 0 if self._config.only_labeled else np.ones(len(labels), bool)
            idx  = np.where(mask)[0]
            if len(idx) == 0:
                return []
            # Dividir por posición X para split espacial
            xyz  = self._pc.xyz[idx]
            order = np.argsort(xyz[:, 0])  # ordenar por X
            idx   = idx[order]
            n     = len(idx)
            n_tr  = max(1, int(n * self._config.train_ratio))
            n_val = max(1, int(n * self._config.val_ratio))
            n_tr  = min(n_tr,  n - 2)
            n_val = min(n_val, n - n_tr - 1)
            tiles.append(self._build_tile_dict(idx[:n_tr],         "tile_train", np.array([0.0, 0.0])))
            tiles.append(self._build_tile_dict(idx[n_tr:n_tr+n_val],"tile_val",  np.array([1.0, 0.0])))
            tiles.append(self._build_tile_dict(idx[n_tr+n_val:],   "tile_test",  np.array([2.0, 0.0])))
            return tiles

        for tile in self._tm.tiles:
            # Incluir tiles con al menos 5% de puntos etiquetados
            if tile.n_points == 0 or tile.labeled_pct < 5.0:
                continue
            # Obtener índices del tile
            idx = self._tm.get_tile_indices_fast(tile)
            if idx is None or len(idx) == 0:
                continue
            # Filtrar solo etiquetados si se pide
            lbl_tile = labels[idx]
            if self._config.only_labeled:
                mask = lbl_tile > 0
                idx  = idx[mask]
                if len(idx) == 0:
                    continue
            center = np.array([
                (tile.min_xr + tile.max_xr) * 0.5,
                (tile.min_yr + tile.max_yr) * 0.5,
            ], dtype=np.float64)
            tile_id = f"tile_{tile.col}_{tile.row}"
            tiles.append(self._build_tile_dict(idx, tile_id, center))

        return tiles

    def _build_tile_dict(self, idx: np.ndarray, tile_id: str,
                         center_xy: np.ndarray) -> dict:
        pc     = self._pc
        labels = self._labels
        n      = len(idx)

        # Coordenadas locales centradas en el tile
        xyz = np.array(pc.xyz[idx], dtype=np.float64)
        cx, cy = float(xyz[:, 0].mean()), float(xyz[:, 1].mean())
        cz = float(xyz[:, 2].mean())
        xyz_local = (xyz - np.array([cx, cy, cz])).astype(np.float32)

        # Intensidad
        intensity = np.zeros(n, np.float32)
        if pc.intensity is not None:
            intensity = np.array(pc.intensity[idx], np.float32)

        # RGB normalizado 0-1
        rgb = np.zeros((n, 3), np.float32)
        if pc.rgb is not None:
            rgb = np.array(pc.rgb[idx], np.float32) / 255.0

        # XY normalizado al rango del tile (0-1) para features geométricas
        xy_min = xyz_local[:, :2].min(0)
        xy_max = xyz_local[:, :2].max(0)
        xy_rng = np.maximum(xy_max - xy_min, 1e-6)
        xy_norm = (xyz_local[:, :2] - xy_min) / xy_rng  # (N, 2) en [0,1]

        # Z normalizado (0-1)
        z_min = float(xyz_local[:, 2].min())
        z_max = float(xyz_local[:, 2].max())
        z_norm = (xyz_local[:, 2] - z_min) / max(z_max - z_min, 1e-6)  # (N,)

        lbl = labels[idx].astype(np.uint8)

        return {
            "tile_id":    tile_id,
            "center_xy":  center_xy,
            "xyz_local":  xyz_local,          # (N,3)
            "intensity":  intensity,           # (N,)
            "rgb":        rgb,                 # (N,3)
            "xy_norm":    xy_norm,             # (N,2)
            "z_norm":     z_norm,              # (N,)
            "labels":     lbl,                 # (N,)
            "n_points":   n,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Split espacial (sin data leakage geográfico)
    # ─────────────────────────────────────────────────────────────────────────

    def _spatial_split(self, tiles: List[dict]) -> SplitResult:
        """
        Divide los tiles en train/val/test por posición geográfica.
        Los tiles más alejados entre sí van a splits diferentes.
        Esto previene data leakage: el modelo no puede "memorizar"
        rasgos locales del terreno entre train y test.
        """
        cfg = self._config
        n   = len(tiles)
        if n < 3:
            # Con pocos tiles, asignar manualmente para no dejar ningún split vacío
            ids = list(range(n))
            return SplitResult(
                train_tiles=[tiles[i]["tile_id"] for i in ids[:-min(2,n-1)]],
                val_tiles  =[tiles[i]["tile_id"] for i in ids[-min(2,n-1):-1]] if n>1 else [],
                test_tiles =[tiles[i]["tile_id"] for i in ids[-1:]] if n>0 else [],
            )

        # Ordenar tiles por posición (X primero, luego Y) para distribución uniforme
        rng = np.random.default_rng(42)
        shuffled = rng.permutation(n)

        n_train = max(1, int(n * cfg.train_ratio))
        n_val   = max(1, int(n * cfg.val_ratio))
        n_train = min(n_train, n - 2)
        n_val   = min(n_val,   n - n_train - 1)

        return SplitResult(
            train_tiles=[tiles[i]["tile_id"] for i in shuffled[:n_train]],
            val_tiles  =[tiles[i]["tile_id"] for i in shuffled[n_train:n_train+n_val]],
            test_tiles =[tiles[i]["tile_id"] for i in shuffled[n_train+n_val:]],
        )

    def _split_name(self, tile_id: str, split: SplitResult) -> str:
        # Tiles generados por la ruta sin TileManager tienen nombre prefijado
        if tile_id == "tile_train": return "train"
        if tile_id == "tile_val":   return "val"
        if tile_id == "tile_test":  return "test"
        if tile_id in split.train_tiles: return "train"
        if tile_id in split.val_tiles:   return "val"
        return "test"

    # ─────────────────────────────────────────────────────────────────────────
    # RandLA-Net (QingyongHu/RandLA-Net)
    # ─────────────────────────────────────────────────────────────────────────

    def _export_randlanet(self, tiles, split, out: Path, p0, p1):
        """
        Formato: un .npy por tile con shape (N, 9) float32
          cols: x, y, z, intensity, r, g, b, x_norm, y_norm
        Labels: tile_id_labels.npy shape (N,) uint8

        Estructura:
          randlanet/
            data/train/tile_0_3.npy + tile_0_3_labels.npy
            data/val/...
            data/test/...
            custom_dataset.py
            train.py
            requirements.txt
            README.md
        """
        for sub in ("train", "val", "test"):
            (out / "data" / sub).mkdir(parents=True, exist_ok=True)

        schema    = self._project.schema
        n_classes = len(schema) + 1   # +1 para "sin etiquetar"
        class_names = ["unlabeled"] + [s.name for s in schema]

        for i, t in enumerate(tiles):
            self.progress.emit(p0 + int((p1-p0)*i/len(tiles)),
                               f"RandLA-Net: {t['tile_id']}…")
            sname = self._split_name(t["tile_id"], split)
            tdir  = out / "data" / sname

            # Features (N, 9): x y z intensity r g b x_norm y_norm
            feats = np.concatenate([
                t["xyz_local"],                     # (N,3)
                t["intensity"][:, None],            # (N,1)
                t["rgb"],                           # (N,3)
                t["xy_norm"],                       # (N,2)
            ], axis=1).astype(np.float32)           # (N,9)

            np.save(str(tdir / f"{t['tile_id']}.npy"), feats)
            np.save(str(tdir / f"{t['tile_id']}_labels.npy"), t["labels"])

        # Archivos de soporte
        self._write_randlanet_dataset_py(out, schema, n_classes, class_names)
        self._write_randlanet_train_py(out, n_classes, class_names)
        self._write_randlanet_readme(out, class_names)
        self._write_requirements(out, "randlanet")

    def _write_randlanet_dataset_py(self, out, schema, n_classes, class_names):
        colors_hex = ["#808080"] + [s.color for s in schema]
        code = textwrap.dedent(f"""\
        \"\"\"
        custom_dataset.py — Dataset custom para RandLA-Net
        ====================================================
        Generado automáticamente por GeoAnnotate3D.
        COPIAR este archivo al directorio raíz del repo RandLA-Net.
        \"\"\"
        import os, glob
        import numpy as np
        import torch
        from torch.utils.data import Dataset

        # ── Configuración del dataset ─────────────────────────────────────────
        NUM_CLASSES = {n_classes}
        CLASS_NAMES = {class_names}
        CLASS_COLORS = {colors_hex}
        NUM_POINTS  = 40960   # puntos por sample (submuestreo aleatorio si más)
        FEATURES    = 9       # x,y,z,intensity,r,g,b,x_norm,y_norm

        class GeoAnnotateDataset(Dataset):
            \"\"\"
            Carga los tiles exportados por GeoAnnotate3D.
            Cada tile es una escena independiente.
            Se submuestrea aleatoriamente a NUM_POINTS si el tile tiene más.
            \"\"\"
            def __init__(self, data_dir, split='train', num_points=NUM_POINTS,
                         augment=True):
                self.num_points = num_points
                self.augment    = augment and split == 'train'
                pattern = os.path.join(data_dir, split, '*.npy')
                all_files = sorted(f for f in glob.glob(pattern)
                                   if not f.endswith('_labels.npy'))
                self.samples = []
                for f in all_files:
                    lf = f.replace('.npy', '_labels.npy')
                    if os.path.exists(lf):
                        self.samples.append((f, lf))
                print(f"[GeoAnnotate] {{split}}: {{len(self.samples)}} tiles cargados")

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                data_path, label_path = self.samples[idx]
                data   = np.load(data_path).astype(np.float32)  # (N, 9)
                labels = np.load(label_path).astype(np.int64)    # (N,)

                n = len(data)
                if n >= self.num_points:
                    choice = np.random.choice(n, self.num_points, replace=False)
                else:
                    choice = np.random.choice(n, self.num_points, replace=True)

                data   = data[choice]
                labels = labels[choice]

                if self.augment:
                    # Rotación aleatoria en Z
                    angle  = np.random.uniform(0, 2 * np.pi)
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    rot = np.array([[cos_a,-sin_a,0],[sin_a,cos_a,0],[0,0,1]], np.float32)
                    data[:, :3] = data[:, :3] @ rot.T
                    # Jitter de intensidad
                    data[:, 3] += np.random.normal(0, 0.01, len(data))
                    np.clip(data[:, 3], 0, 1, out=data[:, 3])

                return (torch.from_numpy(data).float(),
                        torch.from_numpy(labels).long())


        def get_class_weights(data_dir, split='train'):
            \"\"\"Calcula class_weights por frecuencia inversa para loss ponderado.\"\"\"
            counts = np.zeros(NUM_CLASSES, dtype=np.int64)
            for f in glob.glob(os.path.join(data_dir, split, '*_labels.npy')):
                lbl = np.load(f)
                for c in range(NUM_CLASSES):
                    counts[c] += int((lbl == c).sum())
            counts = np.maximum(counts, 1)
            weights = 1.0 / counts.astype(np.float64)
            return torch.FloatTensor(weights / weights.sum() * NUM_CLASSES)
        """)
        (out / "custom_dataset.py").write_text(code, encoding="utf-8")

    def _write_randlanet_train_py(self, out, n_classes, class_names):
        code = textwrap.dedent(f"""\
        \"\"\"
        train.py — Entrenamiento RandLA-Net con dataset GeoAnnotate3D
        ==============================================================
        Generado automáticamente por GeoAnnotate3D.

        USO:
            1. Clonar RandLA-Net:
               git clone https://github.com/QingyongHu/RandLA-Net.git
               cd RandLA-Net

            2. Copiar archivos de GeoAnnotate3D al repo:
               cp /ruta/a/export/randlanet/custom_dataset.py .
               cp /ruta/a/export/randlanet/train.py .

            3. Instalar dependencias:
               pip install -r requirements.txt

            4. Entrenar:
               python train.py --data_dir /ruta/a/export/randlanet/data

        \"\"\"
        import argparse, os, time
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from custom_dataset import (GeoAnnotateDataset, get_class_weights,
                                    NUM_CLASSES, CLASS_NAMES, NUM_POINTS)

        # ── Hiperparámetros ───────────────────────────────────────────────────
        BATCH_SIZE  = 4
        EPOCHS      = 100
        LR          = 0.01
        DECAY_STEP  = 20
        DECAY_RATE  = 0.7
        NUM_WORKERS = 4
        DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

        def train(data_dir: str, out_dir: str = "checkpoints"):
            os.makedirs(out_dir, exist_ok=True)
            print(f"Dispositivo: {{DEVICE}}")
            print(f"Clases: {{NUM_CLASSES}} — {{CLASS_NAMES}}")

            train_ds = GeoAnnotateDataset(data_dir, 'train', NUM_POINTS, augment=True)
            val_ds   = GeoAnnotateDataset(data_dir, 'val',   NUM_POINTS, augment=False)
            train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
            val_ld   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            weights = get_class_weights(data_dir).to(DEVICE)

            # Importar modelo RandLA-Net del repo
            try:
                from RandLA_Net import Network as RandLANet
                model = RandLANet(num_classes=NUM_CLASSES).to(DEVICE)
            except ImportError:
                print("ERROR: No se encontró el modelo RandLA-Net.")
                print("Asegúrate de estar en el directorio del repo clonado.")
                return

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=DECAY_STEP, gamma=DECAY_RATE)
            criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0)

            best_miou = 0.0
            for epoch in range(EPOCHS):
                # Train
                model.train()
                train_loss = 0.0
                for batch_data, batch_lbl in train_ld:
                    batch_data = batch_data.to(DEVICE)
                    batch_lbl  = batch_lbl.to(DEVICE)
                    optimizer.zero_grad()
                    pred = model(batch_data)
                    loss = criterion(pred.view(-1, NUM_CLASSES), batch_lbl.view(-1))
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                scheduler.step()

                # Validation
                model.eval()
                iou_per_class = np.zeros(NUM_CLASSES)
                counts        = np.zeros(NUM_CLASSES)
                with torch.no_grad():
                    for batch_data, batch_lbl in val_ld:
                        pred = model(batch_data.to(DEVICE)).argmax(dim=-1).cpu().numpy()
                        lbl  = batch_lbl.numpy()
                        for c in range(NUM_CLASSES):
                            inter = ((pred == c) & (lbl == c)).sum()
                            union = ((pred == c) | (lbl == c)).sum()
                            if union > 0:
                                iou_per_class[c] += inter / union
                                counts[c] += 1

                miou = (iou_per_class / np.maximum(counts, 1)).mean()
                print(f"Epoch {{epoch+1:3d}}/{{EPOCHS}} | "
                      f"Loss: {{train_loss/len(train_ld):.4f}} | mIoU val: {{miou:.4f}}")

                if miou > best_miou:
                    best_miou = miou
                    torch.save(model.state_dict(), f"{{out_dir}}/best_model.pth")
                    print(f"  -> Guardado mejor modelo (mIoU={{best_miou:.4f}})")

            print(f"\\nEntrenamiento completo. Mejor mIoU: {{best_miou:.4f}}")
            print(f"Modelo guardado en: {{out_dir}}/best_model.pth")

        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--data_dir",  default="data",
                                help="Carpeta con data/train, data/val, data/test")
            parser.add_argument("--out_dir",   default="checkpoints")
            args = parser.parse_args()
            train(args.data_dir, args.out_dir)
        """)
        (out / "train.py").write_text(code, encoding="utf-8")

    def _write_randlanet_readme(self, out, class_names):
        n = len(class_names)
        names_str = ", ".join(f'"{c}"' for c in class_names)
        md = textwrap.dedent(f"""\
        # Dataset RandLA-Net — GeoAnnotate3D
        
        Generado con **GeoAnnotate3D**. Listo para entrenar RandLA-Net
        directamente sin ninguna modificación adicional.
        
        ## Clases ({n})
        {chr(10).join(f"- `{i}`: {c}" for i,c in enumerate(class_names))}
        
        ## Estructura de archivos
        ```
        data/
          train/
            tile_0_3.npy          # (N, 9) float32: x,y,z,intensity,r,g,b,x_norm,y_norm
            tile_0_3_labels.npy   # (N,) uint8: clase de cada punto
            tile_1_2.npy
            ...
          val/
            ...
          test/
            ...
        custom_dataset.py   # ← COPIAR al repo RandLA-Net
        train.py            # ← ejecutar directamente
        requirements.txt
        ```
        
        ## Pasos para entrenar
        
        ```bash
        # 1. Clonar RandLA-Net
        git clone https://github.com/QingyongHu/RandLA-Net.git
        cd RandLA-Net
        
        # 2. Instalar dependencias
        pip install -r /ruta/export/randlanet/requirements.txt
        
        # 3. Copiar archivos del export
        cp /ruta/export/randlanet/custom_dataset.py .
        cp /ruta/export/randlanet/train.py .
        
        # 4. Entrenar
        python train.py --data_dir /ruta/export/randlanet/data
        
        # 5. El mejor modelo se guarda en: checkpoints/best_model.pth
        ```
        
        ## Formato de los datos
        
        Cada archivo `.npy` contiene un tile (escena) con shape `(N, 9)`:
        
        | Col | Feature    | Rango  |
        |-----|------------|--------|
        | 0   | x          | metros (centrado en el tile) |
        | 1   | y          | metros |
        | 2   | z          | metros |
        | 3   | intensity  | 0-1 |
        | 4   | r          | 0-1 |
        | 5   | g          | 0-1 |
        | 6   | b          | 0-1 |
        | 7   | x_norm     | 0-1 (normalizado al tile) |
        | 8   | y_norm     | 0-1 |
        
        Los labels son `uint8` con valores 0-{n-1}.
        La clase 0 es "sin etiquetar" (excluida del loss si se usa `ignore_index=0`).
        """)
        (out / "README.md").write_text(md, encoding="utf-8")

    # ─────────────────────────────────────────────────────────────────────────
    # PointNet++ (yanx27/Pointnet_Pointnet2_pytorch)
    # ─────────────────────────────────────────────────────────────────────────

    def _export_pointnetpp(self, tiles, split, out: Path, p0, p1):
        for sub in ("train", "val", "test"):
            (out / "data" / sub).mkdir(parents=True, exist_ok=True)

        schema    = self._project.schema
        n_classes = len(schema) + 1
        class_names = ["unlabeled"] + [s.name for s in schema]

        for i, t in enumerate(tiles):
            self.progress.emit(p0 + int((p1-p0)*i/len(tiles)),
                               f"PointNet++: {t['tile_id']}…")
            sname = self._split_name(t["tile_id"], split)
            tdir  = out / "data" / sname

            # PointNet++ S3DIS format: (N, 9) float32
            # cols: x, y, z, r, g, b, x_norm, y_norm, z_norm
            feats = np.concatenate([
                t["xyz_local"],                     # (N,3)
                t["rgb"],                           # (N,3)
                t["xy_norm"],                       # (N,2)
                t["z_norm"][:, None],               # (N,1)
            ], axis=1).astype(np.float32)           # (N,9)

            np.save(str(tdir / f"{t['tile_id']}.npy"), feats)
            np.save(str(tdir / f"{t['tile_id']}_labels.npy"), t["labels"])

        self._write_pointnetpp_dataset_py(out, n_classes, class_names)
        self._write_pointnetpp_train_py(out, n_classes, class_names)
        self._write_pointnetpp_readme(out, class_names)
        self._write_requirements(out, "pointnetpp")

    def _write_pointnetpp_dataset_py(self, out, n_classes, class_names):
        code = textwrap.dedent(f"""\
        \"\"\"
        custom_dataset.py — Dataset custom para PointNet++
        ====================================================
        Generado automáticamente por GeoAnnotate3D.
        COPIAR este archivo al directorio del repo PointNet++.
        Adapta el S3DISDataset del repo original.
        \"\"\"
        import os, glob
        import numpy as np
        import torch
        from torch.utils.data import Dataset

        NUM_CLASSES  = {n_classes}
        CLASS_NAMES  = {class_names}
        NUM_POINTS   = 4096
        FEATURES     = 9       # x,y,z,r,g,b,x_norm,y_norm,z_norm

        class GeoAnnotateDataset(Dataset):
            def __init__(self, data_dir, split='train', num_points=NUM_POINTS,
                         block_size=1.0, augment=True):
                self.num_points = num_points
                self.block_size = block_size
                self.augment    = augment and split == 'train'
                pattern = os.path.join(data_dir, split, '*.npy')
                all_files = sorted(f for f in glob.glob(pattern)
                                   if not f.endswith('_labels.npy'))
                self.samples = [(f, f.replace('.npy','_labels.npy'))
                                for f in all_files
                                if os.path.exists(f.replace('.npy','_labels.npy'))]
                print(f"[GeoAnnotate] {{split}}: {{len(self.samples)}} escenas")

            def __len__(self):
                return len(self.samples) * 10  # 10 samples por escena

            def __getitem__(self, idx):
                scene_idx = idx % len(self.samples)
                data_f, lbl_f = self.samples[scene_idx]
                data   = np.load(data_f).astype(np.float32)  # (N, 9)
                labels = np.load(lbl_f).astype(np.int64)      # (N,)

                n = len(data)
                if n >= self.num_points:
                    choice = np.random.choice(n, self.num_points, replace=False)
                else:
                    choice = np.random.choice(n, self.num_points, replace=True)
                data   = data[choice]
                labels = labels[choice]

                if self.augment:
                    angle  = np.random.uniform(0, 2*np.pi)
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    rot = np.array([[cos_a,-sin_a,0],[sin_a,cos_a,0],[0,0,1]], np.float32)
                    data[:, :3] = data[:, :3] @ rot.T
                    data[:, :3] += np.random.normal(0, 0.01, (len(data),3)).astype(np.float32)

                # PointNet++ espera (N, C) para puntos y (N,) para labels
                return (torch.from_numpy(data).float(),
                        torch.from_numpy(labels).long())
        """)
        (out / "custom_dataset.py").write_text(code, encoding="utf-8")

    def _write_pointnetpp_train_py(self, out, n_classes, class_names):
        code = textwrap.dedent(f"""\
        \"\"\"
        train.py — Entrenamiento PointNet++ con dataset GeoAnnotate3D
        ==============================================================
        Generado por GeoAnnotate3D.

        USO:
            1. git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
               cd Pointnet_Pointnet2_pytorch
            2. pip install -r /ruta/export/pointnetpp/requirements.txt
            3. cp /ruta/export/pointnetpp/custom_dataset.py .
               cp /ruta/export/pointnetpp/train.py .
            4. python train.py --data_dir /ruta/export/pointnetpp/data
        \"\"\"
        import argparse, os
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from custom_dataset import GeoAnnotateDataset, NUM_CLASSES, CLASS_NAMES

        BATCH_SIZE  = 16
        EPOCHS      = 128
        LR          = 0.001
        NUM_WORKERS = 4
        DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

        def train(data_dir, out_dir="log"):
            os.makedirs(out_dir, exist_ok=True)
            train_ds = GeoAnnotateDataset(data_dir, 'train', augment=True)
            val_ds   = GeoAnnotateDataset(data_dir, 'val',   augment=False)
            train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
            val_ld   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            try:
                from models.pointnet2_sem_seg import get_model, get_loss
                model    = get_model(NUM_CLASSES).to(DEVICE)
                loss_fn  = get_loss().to(DEVICE)
            except ImportError:
                print("ERROR: Ejecutar desde el directorio del repo PointNet++")
                return

            optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                          betas=(0.9, 0.999), weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.7)

            best_miou = 0.0
            for epoch in range(EPOCHS):
                model.train()
                for pts, lbl in train_ld:
                    pts = pts.transpose(2,1).to(DEVICE)  # (B, C, N)
                    lbl = lbl.to(DEVICE)
                    optimizer.zero_grad()
                    pred, trans_feat = model(pts)
                    pred = pred.contiguous().view(-1, NUM_CLASSES)
                    loss = loss_fn(pred, lbl.view(-1), trans_feat)
                    loss.backward(); optimizer.step()
                scheduler.step()

                model.eval()
                ious = []
                with torch.no_grad():
                    for pts, lbl in val_ld:
                        pred = model(pts.transpose(2,1).to(DEVICE))[0]
                        pred = pred.argmax(-1).cpu().numpy()
                        lbl  = lbl.numpy()
                        for c in range(1, NUM_CLASSES):  # skip 0
                            i = ((pred==c)&(lbl==c)).sum()
                            u = ((pred==c)|(lbl==c)).sum()
                            if u > 0: ious.append(i/u)
                miou = float(np.mean(ious)) if ious else 0
                print(f"Epoch {{epoch+1:3d}} | mIoU val: {{miou:.4f}}")
                if miou > best_miou:
                    best_miou = miou
                    torch.save(model.state_dict(), f"{{out_dir}}/best_model.pth")
                    print(f"  -> Guardado (mIoU={{best_miou:.4f}})")

        if __name__ == "__main__":
            p = argparse.ArgumentParser()
            p.add_argument("--data_dir", default="data")
            p.add_argument("--out_dir",  default="log")
            a = p.parse_args()
            train(a.data_dir, a.out_dir)
        """)
        (out / "train.py").write_text(code, encoding="utf-8")

    def _write_pointnetpp_readme(self, out, class_names):
        md = textwrap.dedent(f"""\
        # Dataset PointNet++ — GeoAnnotate3D
        
        ## Clases ({len(class_names)})
        {chr(10).join(f"- `{i}`: {c}" for i,c in enumerate(class_names))}
        
        ## Pasos para entrenar
        
        ```bash
        git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
        cd Pointnet_Pointnet2_pytorch
        pip install -r /ruta/export/pointnetpp/requirements.txt
        cp /ruta/export/pointnetpp/custom_dataset.py .
        cp /ruta/export/pointnetpp/train.py .
        python train.py --data_dir /ruta/export/pointnetpp/data
        ```
        
        ## Formato (N, 9) float32
        | Col | Feature  | Rango |
        |-----|----------|-------|
        | 0-2 | x, y, z  | metros (centrado en tile) |
        | 3-5 | r, g, b  | 0-1 |
        | 6-7 | x_norm, y_norm | 0-1 |
        | 8   | z_norm   | 0-1 |
        """)
        (out / "README.md").write_text(md, encoding="utf-8")

    # ─────────────────────────────────────────────────────────────────────────
    # KPConv (HuguesThomas/KPConv-PyTorch)
    # ─────────────────────────────────────────────────────────────────────────

    def _export_kpconv(self, tiles, split, out: Path, p0, p1):
        for sub in ("train", "val", "test"):
            (out / "data" / sub).mkdir(parents=True, exist_ok=True)

        schema      = self._project.schema
        n_classes   = len(schema) + 1
        class_names = ["unlabeled"] + [s.name for s in schema]

        for i, t in enumerate(tiles):
            self.progress.emit(p0 + int((p1-p0)*i/len(tiles)),
                               f"KPConv: {t['tile_id']}…")
            sname = self._split_name(t["tile_id"], split)
            tdir  = out / "data" / sname

            # KPConv usa PLY con campos: x,y,z + scalar_label + RGB opcional
            self._write_ply(
                tdir / f"{t['tile_id']}.ply",
                t["xyz_local"],
                t["labels"],
                t["rgb"],
            )

        self._write_kpconv_dataset_py(out, n_classes, class_names)
        self._write_kpconv_train_py(out, n_classes, class_names)
        self._write_kpconv_readme(out, class_names)
        self._write_requirements(out, "kpconv")

    def _write_ply(self, path: Path, xyz: np.ndarray,
                   labels: np.ndarray, rgb: np.ndarray):
        """Escribe un archivo PLY con xyz, scalar_label y RGB."""
        n = len(xyz)
        has_rgb = rgb is not None and rgb.max() > 0
        header  = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar scalar_label\n"
        )
        if has_rgb:
            header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        header += "end_header\n"

        with open(str(path), "wb") as f:
            f.write(header.encode("ascii"))
            if has_rgb:
                rgb_u8 = (rgb * 255).astype(np.uint8)
                for i in range(n):
                    f.write(xyz[i].astype(np.float32).tobytes())
                    f.write(labels[i:i+1].astype(np.uint8).tobytes())
                    f.write(rgb_u8[i].tobytes())
            else:
                for i in range(n):
                    f.write(xyz[i].astype(np.float32).tobytes())
                    f.write(labels[i:i+1].astype(np.uint8).tobytes())

    def _write_kpconv_dataset_py(self, out, n_classes, class_names):
        code = textwrap.dedent(f"""\
        \"\"\"
        custom_dataset.py — Dataset custom para KPConv
        ================================================
        Generado por GeoAnnotate3D.
        COPIAR al directorio del repo KPConv-PyTorch.
        \"\"\"
        import os, glob
        import numpy as np
        import torch
        from torch.utils.data import Dataset

        NUM_CLASSES  = {n_classes}
        CLASS_NAMES  = {class_names}

        def read_ply(path):
            \"\"\"Lee un .ply binario exportado por GeoAnnotate3D.\"\"\"
            with open(path, 'rb') as f:
                lines = []
                while True:
                    line = f.readline().decode('ascii').strip()
                    lines.append(line)
                    if line == 'end_header':
                        break
                n = int(next(l.split()[-1] for l in lines if l.startswith('element vertex')))
                has_rgb = any('red' in l for l in lines)
                dtype = [('x','f4'),('y','f4'),('z','f4'),('label','u1')]
                if has_rgb:
                    dtype += [('red','u1'),('green','u1'),('blue','u1')]
                data = np.frombuffer(f.read(n * np.dtype(dtype).itemsize), dtype=dtype)

            xyz    = np.stack([data['x'],data['y'],data['z']], axis=1).astype(np.float32)
            labels = data['label'].astype(np.int64)
            rgb    = None
            if has_rgb:
                rgb = np.stack([data['red'],data['green'],data['blue']],1).astype(np.float32)/255
            return xyz, labels, rgb

        class GeoAnnotateDataset(Dataset):
            def __init__(self, data_dir, split='train', num_points=8192, augment=True):
                self.num_points = num_points
                self.augment    = augment and split == 'train'
                self.files = sorted(glob.glob(os.path.join(data_dir, split, '*.ply')))
                print(f"[GeoAnnotate] {{split}}: {{len(self.files)}} escenas")

            def __len__(self): return len(self.files)

            def __getitem__(self, idx):
                xyz, labels, rgb = read_ply(self.files[idx])
                n = len(xyz)
                choice = (np.random.choice(n, self.num_points, replace=n < self.num_points))
                xyz, labels = xyz[choice], labels[choice]
                if rgb is not None: rgb = rgb[choice]

                if self.augment:
                    angle = np.random.uniform(0, 2*np.pi)
                    c, s  = np.cos(angle), np.sin(angle)
                    R = np.array([[c,-s,0],[s,c,0],[0,0,1]], np.float32)
                    xyz = xyz @ R.T

                feats = xyz.copy()
                if rgb is not None:
                    feats = np.concatenate([xyz, rgb], axis=1)

                return (torch.from_numpy(feats).float(),
                        torch.from_numpy(labels).long())
        """)
        (out / "custom_dataset.py").write_text(code, encoding="utf-8")

    def _write_kpconv_train_py(self, out, n_classes, class_names):
        code = textwrap.dedent(f"""\
        \"\"\"
        train.py — Entrenamiento KPConv con dataset GeoAnnotate3D
        ==========================================================
        Generado por GeoAnnotate3D.

        USO:
            1. git clone https://github.com/HuguesThomas/KPConv-PyTorch.git
               cd KPConv-PyTorch
            2. pip install -r /ruta/export/kpconv/requirements.txt
            3. cp /ruta/export/kpconv/custom_dataset.py .
               cp /ruta/export/kpconv/train.py .
            4. python train.py --data_dir /ruta/export/kpconv/data
        \"\"\"
        import argparse, os
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from custom_dataset import GeoAnnotateDataset, NUM_CLASSES, CLASS_NAMES

        BATCH_SIZE  = 8
        EPOCHS      = 200
        LR          = 1e-2
        NUM_WORKERS = 4
        DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

        def train(data_dir, out_dir="results"):
            os.makedirs(out_dir, exist_ok=True)
            train_ds = GeoAnnotateDataset(data_dir, 'train', augment=True)
            val_ds   = GeoAnnotateDataset(data_dir, 'val',   augment=False)
            train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
            val_ld   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            try:
                from models.architectures import KPFCNN
                config = type('cfg', (), dict(
                    num_classes=NUM_CLASSES, first_features_dim=128,
                    in_features_dim=4, architecture=['simple','resblock']*5+['nearest_upsample']*5,
                ))()
                model = KPFCNN(config).to(DEVICE)
            except ImportError:
                print("ERROR: Ejecutar desde el directorio del repo KPConv-PyTorch")
                return

            optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                         momentum=0.98, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            best_miou = 0.0

            for epoch in range(EPOCHS):
                model.train()
                for feats, lbl in train_ld:
                    optimizer.zero_grad()
                    pred = model(feats.to(DEVICE))
                    loss = criterion(pred.view(-1,NUM_CLASSES), lbl.view(-1).to(DEVICE))
                    loss.backward(); optimizer.step()
                model.eval()
                ious = []
                with torch.no_grad():
                    for feats, lbl in val_ld:
                        pred = model(feats.to(DEVICE)).argmax(-1).cpu().numpy()
                        lbl  = lbl.numpy()
                        for c in range(1, NUM_CLASSES):
                            i = ((pred==c)&(lbl==c)).sum()
                            u = ((pred==c)|(lbl==c)).sum()
                            if u: ious.append(i/u)
                miou = float(np.mean(ious)) if ious else 0
                print(f"Epoch {{epoch+1:3d}} | mIoU val: {{miou:.4f}}")
                if miou > best_miou:
                    best_miou = miou
                    torch.save(model.state_dict(), f"{{out_dir}}/best_model.pth")

        if __name__ == "__main__":
            p = argparse.ArgumentParser()
            p.add_argument("--data_dir", default="data")
            p.add_argument("--out_dir",  default="results")
            a = p.parse_args()
            train(a.data_dir, a.out_dir)
        """)
        (out / "train.py").write_text(code, encoding="utf-8")

    def _write_kpconv_readme(self, out, class_names):
        md = textwrap.dedent(f"""\
        # Dataset KPConv — GeoAnnotate3D
        
        ## Clases ({len(class_names)})
        {chr(10).join(f"- `{i}`: {c}" for i,c in enumerate(class_names))}
        
        ## Pasos para entrenar
        
        ```bash
        git clone https://github.com/HuguesThomas/KPConv-PyTorch.git
        cd KPConv-PyTorch
        pip install -r /ruta/export/kpconv/requirements.txt
        cp /ruta/export/kpconv/custom_dataset.py .
        cp /ruta/export/kpconv/train.py .
        python train.py --data_dir /ruta/export/kpconv/data
        ```
        
        ## Formato PLY
        Cada tile es un archivo `.ply` binario little-endian con campos:
        `x y z scalar_label [red green blue]`
        """)
        (out / "README.md").write_text(md, encoding="utf-8")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers compartidos
    # ─────────────────────────────────────────────────────────────────────────

    def _write_requirements(self, out: Path, arch: str):
        reqs = {
            "randlanet": "torch>=1.10\nnumpy\nscikit-learn\ntqdm\nopen3d\n",
            "pointnetpp": "torch>=1.10\nnumpy\nscikit-learn\ntqdm\nopen3d\nh5py\n",
            "kpconv":    "torch>=1.10\nnumpy\nscikit-learn\ntqdm\nopen3d\n",
        }
        (out / "requirements.txt").write_text(reqs.get(arch, "torch\nnumpy\n"),
                                               encoding="utf-8")

    def _export_classified_las(self, out: Path):
        """Nube .las con classification = labels para GIS / CloudCompare."""
        try:
            import laspy
        except ImportError:
            return
        pc  = self._pc
        lbl = self._labels
        if pc is None or pc.xyz is None or lbl is None:
            return
        try:
            offset = pc.offset
            x = pc.xyz[:, 0].astype(np.float64) + offset[0]
            y = pc.xyz[:, 1].astype(np.float64) + offset[1]
            z = pc.xyz[:, 2].astype(np.float64) + offset[2]
            hdr = laspy.LasHeader(point_format=0, version="1.4")
            hdr.offsets = np.array([x.min(), y.min(), z.min()])
            hdr.scales  = np.array([0.001, 0.001, 0.001])
            las = laspy.LasData(header=hdr)
            las.x = x; las.y = y; las.z = z
            out_lbl = lbl.astype(np.uint8)
            if getattr(self._config, "asprs_codes", False) and self._project is not None:
                # Remapear cada class_id presente a su código ASPRS (o
                # 64+id si no se reconoce el nombre) — una tabla de
                # traducción, no un bucle por punto, para no penalizar
                # nubes de cientos de millones de puntos.
                lut = np.arange(256, dtype=np.uint8)
                for sc in self._project.schema:
                    if 0 <= sc.id < 256:
                        lut[sc.id] = _asprs_code_for(sc.id, sc.name)
                out_lbl = lut[np.clip(out_lbl, 0, 255)]
            las.classification = out_lbl
            if pc.intensity is not None:
                las.intensity = (pc.intensity * 65535).astype(np.uint16)
            stem = Path(pc.filename).stem if pc.filename else "cloud"
            las.write(str(out / f"{stem}_classified.las"))
        except Exception as e:
            print(f"[export_las] {e}")

    def _write_global_metadata(self, tiles, split, out: Path):
        schema = self._project.schema
        per_class: Dict[str, int] = {}
        lbl = self._labels
        if lbl is not None:
            u, c = np.unique(lbl[lbl > 0], return_counts=True)
            per_class = {str(int(k)): int(v) for k, v in zip(u, c)}

        # Calcular class_weights
        counts = {int(k): int(v) for k, v in per_class.items()}
        if counts:
            inv = {k: 1.0/v for k,v in counts.items()}
            s   = sum(inv.values())
            cw  = {k: round(v/s, 6) for k,v in inv.items()}
        else:
            cw = {}

        meta = {
            "geoannotate3d_version": "10.0",
            "project_name":   self._project.name,
            "source_file":    self._project.source_file,
            "crs":            self._project.crs,
            "offset_xyz":     self._project.offset_xyz,
            "architectures":  self._config.architectures,
            "split_strategy": self._config.split_strategy,
            "n_tiles_total":  len(tiles),
            "splits": {
                "train": len(split.train_tiles),
                "val":   len(split.val_tiles),
                "test":  len(split.test_tiles),
            },
            "tile_ids": {
                "train": split.train_tiles,
                "val":   split.val_tiles,
                "test":  split.test_tiles,
            },
            "schema": [s.to_dict() for s in schema],
            "per_class_counts": per_class,
            "class_weights": cw,
            "features": {
                "randlanet":  ["x","y","z","intensity","r","g","b","x_norm","y_norm"],
                "pointnetpp": ["x","y","z","r","g","b","x_norm","y_norm","z_norm"],
                "kpconv":     ["x","y","z","r","g","b"],
            },
        }
        with open(str(out / "dataset.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Función standalone de split espacial (para tests)
# ─────────────────────────────────────────────────────────────────────────────

def compute_spatial_split(xyz: np.ndarray, block_size_m: float = 50.0,
                           train_ratio: float = 0.70,
                           val_ratio:   float = 0.20):
    """Split espacial por bloques — sin data leakage."""
    from annotation.exporter import SplitResult
    if len(xyz) == 0:
        e = np.zeros(0, np.int32)
        return SplitResult([], [], [])
    xy  = xyz[:, :2]
    bb  = xy.min(0)
    gx  = np.floor((xy[:,0]-bb[0])/block_size_m).astype(np.int32)
    gy  = np.floor((xy[:,1]-bb[1])/block_size_m).astype(np.int32)
    cid = gx.astype(np.int64)*100_003 + gy.astype(np.int64)
    uc  = np.unique(cid)
    np.random.default_rng(42).shuffle(uc)
    n   = len(uc)
    t1  = max(1, int(n*train_ratio))
    t2  = t1 + max(1, int(n*val_ratio))
    t2  = min(t2, n-1)
    all_idx   = np.arange(len(xyz), np.int32)
    train_m   = np.isin(cid, uc[:t1])
    val_m     = np.isin(cid, uc[t1:t2])
    return SplitResult(
        all_idx[train_m].tolist(),
        all_idx[val_m].tolist(),
        all_idx[~(train_m|val_m)].tolist(),
    )
