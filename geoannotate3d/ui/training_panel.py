"""
ui/training_panel.py — Panel de entrenamiento integrado

Permite configurar, lanzar y monitorear el entrenamiento de:
  - RandLA-Net
  - PointNet++
  - KPConv

El entrenamiento corre en un QThread para no bloquear la UI.
Muestra en tiempo real:
  - Curvas de loss (train / val)
  - mIoU por epoch
  - Log de consola
  - Barra de progreso
"""
from __future__ import annotations
import os, sys, json, time, traceback
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QProgressBar,
    QFrame, QSplitter, QGroupBox, QScrollArea, QFileDialog,
    QCheckBox, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QFont

from ui.icons import icon as qicon, pixmap as qpixmap
from ui.theme import (
    BG, SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE,
    ACCENT, ACCENT_STRONG, ACCENT_SOFT, ACCENT_BORDER,
    OK, OK_SOFT, WARN, WARN_SOFT,
)


# ── Gráfica de curvas de entrenamiento ─────────────────────────────────────────

class _LossChart(QWidget):
    """
    Gráfica simple de loss y mIoU por epoch.
    Dibuja directamente con QPainter — sin matplotlib, sin dependencias extra.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._train_loss: list = []
        self._val_loss:   list = []
        self._miou:       list = []
        self.setStyleSheet(f"background:{SURFACE};border:1px solid {BORDER};border-radius:4px;")

    def add_epoch(self, train_loss: float, val_loss: float, miou: float):
        self._train_loss.append(train_loss)
        self._val_loss.append(val_loss)
        self._miou.append(miou)
        self.update()

    def reset(self):
        self._train_loss.clear(); self._val_loss.clear(); self._miou.clear()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        pad = 32

        # Background
        p.fillRect(0, 0, W, H, QColor(SURFACE))

        if not self._train_loss:
            p.setPen(QColor(TEXT_MUTE))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(0, 0, W, H, Qt.AlignCenter, "Sin datos — inicia el entrenamiento")
            p.end(); return

        n = len(self._train_loss)
        # X axis
        def x(i): return int(pad + (W - pad * 2) * i / max(n - 1, 1))

        def draw_curve(values, color, label, offset_y=0):
            if not values: return
            mn = min(values); mx = max(values)
            span = max(mx - mn, 1e-6)
            def y(v): return int(H - pad - (H - pad * 2) * (v - mn) / span) + offset_y
            pen = QPen(QColor(color)); pen.setWidth(2)
            p.setPen(pen)
            for i in range(1, len(values)):
                p.drawLine(x(i-1), y(values[i-1]), x(i), y(values[i]))
            # Label at end
            if values:
                p.setFont(QFont("Courier", 7))
                p.drawText(x(len(values)-1)+3, y(values[-1])+4,
                           f"{label}: {values[-1]:.3f}")

        draw_curve(self._train_loss, ACCENT, "train_loss")
        draw_curve(self._val_loss,   ACCENT_STRONG, "val_loss")
        # mIoU on secondary scale — draw at top area
        if self._miou:
            mn, mx = 0.0, 1.0
            span = 1.0
            def ym(v): return int(H - pad - (H - pad * 2) * v / span)
            pen = QPen(QColor(OK)); pen.setWidth(2); pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            for i in range(1, len(self._miou)):
                p.drawLine(x(i-1), ym(self._miou[i-1]), x(i), ym(self._miou[i]))
            p.setFont(QFont("Segoe UI", 7))
            p.setPen(QColor(OK))
            p.drawText(x(len(self._miou)-1)+3, ym(self._miou[-1])+4,
                       f"mIoU: {self._miou[-1]:.3f}")

        # X axis labels
        p.setPen(QColor(TEXT_MUTE))
        p.setFont(QFont("Segoe UI", 7))
        p.drawText(pad, H-2, "ep 1")
        p.drawText(x(n-1)-12, H-2, f"ep {n}")

        # Axes
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawLine(pad, pad, pad, H-pad)
        p.drawLine(pad, H-pad, W-pad, H-pad)
        p.end()


# ── Worker de entrenamiento ────────────────────────────────────────────────────

class TrainingWorker(QThread):
    """
    Hilo de entrenamiento. Lanza el training loop de la red seleccionada
    y emite señales para actualizar la UI en tiempo real.
    """
    progress    = pyqtSignal(int, int)              # (epoch, total_epochs)
    epoch_done  = pyqtSignal(float, float, float)   # (train_loss, val_loss, miou)
    log_line    = pyqtSignal(str)
    finished_ok = pyqtSignal(str)                   # path al mejor modelo
    error       = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self._cfg  = config
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self._train()
        except Exception as e:
            self.error.emit(f"Error: {e}\n{traceback.format_exc()[:600]}")

    def _train(self):
        cfg  = self._cfg
        arch = cfg["arch"]           # "RandLA-Net" | "PointNet++" | "KPConv"
        data_dir   = cfg["data_dir"]
        out_dir    = cfg["out_dir"]
        epochs     = cfg["epochs"]
        batch      = cfg["batch"]
        lr         = cfg["lr"]
        pts        = cfg["pts"]
        num_classes = cfg["num_classes"]
        class_names = cfg["class_names"]

        self.log_line.emit(f"[Entrenamiento] Arquitectura: {arch}")
        self.log_line.emit(f"[Entrenamiento] Dataset: {data_dir}")
        self.log_line.emit(f"[Entrenamiento] Clases: {num_classes} — {class_names}")
        self.log_line.emit(f"[Entrenamiento] Epochs: {epochs}  Batch: {batch}  LR: {lr}")

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_line.emit(f"[Entrenamiento] Dispositivo: {device}" + (
                f" ({torch.cuda.get_device_name(0)})" if device=="cuda" else " (CPU — lento)"))
        except ImportError:
            self.error.emit("PyTorch no instalado.\n\npip install torch torchvision")
            return

        # Build the appropriate model and run training
        if arch == "RandLA-Net":
            self._train_randla(data_dir, out_dir, epochs, batch, lr, pts,
                               num_classes, class_names, device)
        elif arch == "PointNet++":
            self._train_pointnet(data_dir, out_dir, epochs, batch, lr, pts,
                                 num_classes, class_names, device)
        elif arch == "KPConv":
            self._train_kpconv(data_dir, out_dir, epochs, batch, lr, pts,
                               num_classes, class_names, device)

    def _train_randla(self, data_dir, out_dir, epochs, batch, lr, pts,
                      num_classes, class_names, device):
        import torch, torch.nn as nn, torch.nn.functional as F
        import numpy as np
        from torch.utils.data import DataLoader

        os.makedirs(out_dir, exist_ok=True)

        # ── Inline model (no external dependency) ────────────────────────────
        def knn(xyz, k, chunk=256):
            B, _, N = xyz.shape; k = min(k, N-1)
            xy = xyz.permute(0,2,1)
            out = torch.zeros(B,N,k,dtype=torch.long,device=xyz.device)
            for s in range(0,N,chunk):
                e=min(s+chunk,N)
                diff=xy[:,s:e].unsqueeze(2)-xy.unsqueeze(1)
                _,idx=(diff**2).sum(-1).topk(k+1,dim=-1,largest=False)
                out[:,s:e]=idx[:,:,1:]
            return out

        def gather(x,idx):
            B,D,N=x.shape;k=idx.shape[-1]
            return x.unsqueeze(-1).expand(B,D,N,N).gather(2,idx.unsqueeze(1).expand(B,D,N,k))

        class MLP(nn.Module):
            def __init__(self,i,o,act=True):
                super().__init__()
                self.net=nn.Sequential(nn.Conv1d(i,o,1,bias=False),nn.BatchNorm1d(o),
                                       nn.LeakyReLU(0.2) if act else nn.Identity())
            def forward(self,x): return self.net(x)

        class LFA(nn.Module):
            def __init__(self,in_ch,out_ch,k=16):
                super().__init__()
                self.k=k;mid=out_ch//2
                self.enc=MLP(in_ch+3,mid)
                self.att=nn.Sequential(MLP(mid,mid),nn.Conv1d(mid,1,1))
                self.out=MLP(mid,out_ch);self.sc=MLP(in_ch,out_ch,act=False)
                self.act=nn.LeakyReLU(0.2)
            def forward(self,xyz,feat):
                B,C,N=feat.shape;k=min(self.k,N-1);mid=self.enc.net[0].out_channels
                idx=knn(xyz,k);rel_pos=gather(xyz,idx)-xyz.unsqueeze(-1)
                inp=torch.cat([feat.unsqueeze(-1).expand(B,C,N,k),rel_pos],1)
                enc=self.enc(inp.view(B,C+3,N*k)).view(B,mid,N,k)
                att=torch.softmax(self.att(enc.view(B,mid,N*k)).view(B,1,N,k),-1)
                return self.act(self.out((enc*att).sum(-1))+self.sc(feat))

        class RandLA(nn.Module):
            def __init__(self,nc):
                super().__init__()
                self.fc0=MLP(9,32);self.l1=LFA(32,64);self.l2=LFA(64,128);self.l3=LFA(128,256)
                self.d2=MLP(256+128,128);self.d1=MLP(128+64,64);self.d0=MLP(64+32,32)
                self.clf=nn.Sequential(MLP(32,64),nn.Dropout(0.5),nn.Conv1d(64,nc,1))
            def forward(self,x):
                xyz=x[:,:3];N=x.shape[-1]
                f0=self.fc0(x);f1=self.l1(xyz,f0)
                n1=max(8,N//4);i1=torch.randperm(N,device=x.device)[:n1]
                f1s=f1[...,i1];xyz1=xyz[...,i1]
                f2=self.l2(xyz1,f1s)
                n2=max(4,n1//4);i2=torch.randperm(n1,device=x.device)[:n2]
                f2s=f2[...,i2];xyz2=xyz1[...,i2]
                f3=self.l3(xyz2,f2s)
                up3=F.interpolate(f3,n1,mode='nearest')
                d2=self.d2(torch.cat([up3,f2],1))
                up2=F.interpolate(d2,N,mode='nearest')
                d1=self.d1(torch.cat([up2,f1],1))
                d0=self.d0(torch.cat([d1,f0],1))
                return self.clf(d0)

        self.log_line.emit("[RandLA-Net] Cargando dataset…")

        # Buscar custom_dataset.py en data_dir o carpetas padre
        import importlib.util as _ilu
        ds_mod = None
        for sp in [data_dir, str(Path(data_dir).parent),
                   str(Path(data_dir).parent.parent)]:
            cand = Path(sp) / "custom_dataset.py"
            if cand.exists():
                spec = _ilu.spec_from_file_location("custom_dataset", cand)
                ds_mod = _ilu.module_from_spec(spec)
                spec.loader.exec_module(ds_mod)
                self.log_line.emit(f"[RandLA-Net] Módulo dataset: {cand}")
                break
        if ds_mod is None:
            self.error.emit(
                "No se encontró custom_dataset.py\n\n"
                "Pasos:\n"
                "1. Paso 4 → Exportar Dataset (elige RandLA-Net)\n"
                "2. Anota la carpeta donde se generó (contiene data/, custom_dataset.py)\n"
                "3. En Paso 5, selecciona la subcarpeta data/ como Dataset")
            return
        try:
            GeoAnnotateDataset = ds_mod.GeoAnnotateDataset
            get_class_weights   = ds_mod.get_class_weights
            tds = GeoAnnotateDataset(data_dir,"train",pts,augment=True)
            vds = GeoAnnotateDataset(data_dir,"val",  pts,augment=False)
            tl  = DataLoader(tds,batch,shuffle=True, num_workers=0,
                             pin_memory=(device=="cuda"))
            vl  = DataLoader(vds,batch,shuffle=False,num_workers=0)
            self.log_line.emit(f"[RandLA-Net] Train: {len(tds)} muestras · Val: {len(vds)} muestras")
            if len(vds) == 0:
                self.error.emit(
                    "El conjunto de validación está vacío.\n\n"
                    "Solución: exporta de nuevo el dataset con más puntos etiquetados\n"
                    "(al menos 5% de la nube) o usa más tiles.")
                return
            # Verificar que val tiene puntos etiquetados
            try:
                sample_pt, sample_lbl = vds[0]
                n_labeled_sample = int((sample_lbl > 0).sum())
                pct = 100 * n_labeled_sample / max(len(sample_lbl), 1)
                self.log_line.emit(
                    f"[RandLA-Net] Val muestra: {n_labeled_sample}/{len(sample_lbl)} pts "
                    f"etiquetados ({pct:.1f}%)")
                if pct < 1.0:
                    self.log_line.emit(
                        "[WARNING] El val set tiene <1% puntos etiquetados. "
                        "mIoU puede ser 0. Exporta más nube etiquetada.")
            except Exception: pass
        except Exception as e:
            self.error.emit(f"Error cargando dataset: {e}\n"
                           "La carpeta debe contener train/ y val/ con archivos .npy")
            return

        # Get actual num_classes from dataset (avoids mismatch)
        actual_nc = getattr(ds_mod, 'NUM_CLASSES', num_classes)
        actual_nc = getattr(ds_mod, 'NUM_CLASSES', num_classes)
        model = self._build_randlanet(actual_nc).to(device)
        self._train_with_model(model, "RandLA-Net", data_dir, out_dir, epochs, batch,
                               lr, pts, actual_nc, class_names, ds_mod, device)

    def _build_model(self, arch: str, num_classes: int):
        """Factory: construye el modelo correcto según la arquitectura."""
        import torch.nn as nn
        import torch.nn.functional as F
        import torch

        if arch == "RandLA-Net":
            return self._build_randlanet(num_classes)
        elif arch == "PointNet++":
            return self._build_pointnetpp(num_classes)
        elif arch == "KPConv":
            return self._build_kpconv(num_classes)
        else:
            self.log_line.emit(f"[WARNING] Arquitectura '{arch}' desconocida, usando RandLA-Net")
            return self._build_randlanet(num_classes)

    def _build_randlanet(self, num_classes: int):
        """RandLA-Net inline (ya implementado en _train_randla)."""
        import torch, torch.nn as nn, torch.nn.functional as F

        def knn(xyz, k, chunk=256):
            B, _, N = xyz.shape; k = min(k, N-1)
            xy = xyz.permute(0,2,1)
            out = torch.zeros(B,N,k,dtype=torch.long,device=xyz.device)
            for s in range(0,N,chunk):
                e=min(s+chunk,N)
                diff=xy[:,s:e].unsqueeze(2)-xy.unsqueeze(1)
                _,idx=(diff**2).sum(-1).topk(k+1,dim=-1,largest=False)
                out[:,s:e]=idx[:,:,1:]
            return out

        def gather(x,idx):
            B,D,N=x.shape;k=idx.shape[-1]
            return x.unsqueeze(-1).expand(B,D,N,N).gather(2,idx.unsqueeze(1).expand(B,D,N,k))

        class MLP(nn.Module):
            def __init__(self,i,o,act=True):
                super().__init__()
                self.net=nn.Sequential(nn.Conv1d(i,o,1,bias=False),nn.BatchNorm1d(o),
                                       nn.LeakyReLU(0.2) if act else nn.Identity())
            def forward(self,x): return self.net(x)

        class LFA(nn.Module):
            def __init__(self,in_ch,out_ch,k=16):
                super().__init__()
                self.k=k; mid=out_ch//2
                self.enc=MLP(in_ch+3,mid)
                self.att=nn.Sequential(MLP(mid,mid),nn.Conv1d(mid,1,1))
                self.out=MLP(mid,out_ch); self.sc=MLP(in_ch,out_ch,act=False)
                self.act=nn.LeakyReLU(0.2)
            def forward(self,xyz,feat):
                B,C,N=feat.shape; k=min(self.k,N-1); mid=self.enc.net[0].out_channels
                idx=knn(xyz,k); rel_pos=gather(xyz,idx)-xyz.unsqueeze(-1)
                inp=torch.cat([feat.unsqueeze(-1).expand(B,C,N,k),rel_pos],1)
                enc=self.enc(inp.view(B,C+3,N*k)).view(B,mid,N,k)
                att=torch.softmax(self.att(enc.view(B,mid,N*k)).view(B,1,N,k),-1)
                return self.act(self.out((enc*att).sum(-1))+self.sc(feat))

        class RandLA(nn.Module):
            def __init__(self,nc):
                super().__init__()
                self.fc0=MLP(9,32); self.l1=LFA(32,64); self.l2=LFA(64,128); self.l3=LFA(128,256)
                self.d2=MLP(256+128,128); self.d1=MLP(128+64,64); self.d0=MLP(64+32,32)
                self.clf=nn.Sequential(MLP(32,64),nn.Dropout(0.5),nn.Conv1d(64,nc,1))
            def forward(self,x):
                xyz=x[:,:3]; N=x.shape[-1]
                f0=self.fc0(x); f1=self.l1(xyz,f0)
                n1=max(8,N//4); i1=torch.randperm(N,device=x.device)[:n1]
                f1s=f1[...,i1]; xyz1=xyz[...,i1]
                f2=self.l2(xyz1,f1s)
                n2=max(4,n1//4); i2=torch.randperm(n1,device=x.device)[:n2]
                f2s=f2[...,i2]; xyz2=xyz1[...,i2]
                f3=self.l3(xyz2,f2s)
                up3=F.interpolate(f3,n1,mode='nearest')
                d2=self.d2(torch.cat([up3,f2],1))
                up2=F.interpolate(d2,N,mode='nearest')
                d1=self.d1(torch.cat([up2,f1],1))
                d0=self.d0(torch.cat([d1,f0],1))
                return self.clf(d0)
        return RandLA(num_classes)

    def _build_pointnetpp(self, num_classes: int):
        """
        PointNet++ Segmentation (MSG — Multi-Scale Grouping) inline.
        Arquitectura: encoder SA (Set Abstraction) + decoder FP (Feature Propagation).
        Sin dependencias externas.
        """
        import torch, torch.nn as nn, torch.nn.functional as F

        def sq_dist(src, dst):
            """Distancias al cuadrado entre dos conjuntos de puntos. (B,N,C),(B,M,C) → (B,N,M)"""
            return (src.unsqueeze(2) - dst.unsqueeze(1)).pow(2).sum(-1)

        def ball_query(xyz, new_xyz, radius, k):
            """Vectorized ball query — sin bucles Python, apto para GPU."""
            dist = sq_dist(new_xyz, xyz)           # (B,M,N)
            masked = dist.clone()
            masked[dist > radius*radius] = 1e10    # fuera del radio
            topk  = masked.topk(k, dim=-1, largest=False)
            idx   = topk.indices                   # (B,M,k)
            # Puntos sin vecinos válidos → usar el primer índice de cada fila
            valid  = (topk.values < 1e9)
            first  = idx[:,:,0:1].expand_as(idx)
            return torch.where(valid, idx, first)

        def farthest_point_sample(xyz, n_pts):
            """FPS: selecciona n_pts puntos lo más distribuidos posible."""
            B, N, _ = xyz.shape
            device  = xyz.device
            sel     = torch.zeros(B, n_pts, dtype=torch.long, device=device)
            dist    = torch.full((B, N), float('inf'), device=device)
            cur     = torch.zeros(B, dtype=torch.long, device=device)
            for i in range(n_pts):
                sel[:, i] = cur
                dp = (xyz - xyz[torch.arange(B),cur,:].unsqueeze(1)).pow(2).sum(-1)
                dist  = torch.min(dist, dp)
                cur   = dist.argmax(1)
            return sel

        def index_points(pts, idx):
            """Indexa pts (B,N,C) con idx (B,M) o (B,M,K) → (B,M,C) o (B,M,K,C)."""
            B = pts.shape[0]
            if idx.dim() == 2:
                return pts.gather(1, idx.unsqueeze(-1).expand(-1,-1,pts.shape[-1]))
            M, K = idx.shape[1], idx.shape[2]
            return pts.gather(1, idx.reshape(B,-1,1).expand(-1,-1,pts.shape[-1])).reshape(B,M,K,-1)

        class SAModule(nn.Module):
            """Set Abstraction: FPS + ball query + shared MLP."""
            def __init__(self, n_pts, radius, k, in_ch, mlp_chs):
                super().__init__()
                self.n_pts   = n_pts
                self.radius  = radius
                self.k       = k
                layers = []
                prev = in_ch + 3
                for out in mlp_chs:
                    layers += [nn.Conv2d(prev, out, 1, bias=False),
                               nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
                    prev = out
                self.mlp = nn.Sequential(*layers)

            def forward(self, xyz, feat):
                B, N, _ = xyz.shape
                n = min(self.n_pts, N)
                fps_idx   = farthest_point_sample(xyz, n)
                new_xyz   = index_points(xyz, fps_idx)           # (B,n,3)
                bq_idx    = ball_query(xyz, new_xyz, self.radius, self.k)  # (B,n,k)
                grouped   = index_points(xyz, bq_idx) - new_xyz.unsqueeze(2)  # (B,n,k,3)
                if feat is not None:
                    gf = index_points(feat, bq_idx)
                    grouped = torch.cat([grouped, gf], dim=-1)
                x = grouped.permute(0,3,1,2)  # (B,C,n,k)
                x = self.mlp(x).max(-1)[0]     # (B,C',n)
                return new_xyz, x.permute(0,2,1)  # (B,n,3),(B,n,C')

        class FPModule(nn.Module):
            """Feature Propagation: interpolación + MLP."""
            def __init__(self, in_ch, mlp_chs):
                super().__init__()
                layers = []
                prev = in_ch
                for out in mlp_chs:
                    layers += [nn.Conv1d(prev, out, 1, bias=False),
                               nn.BatchNorm1d(out), nn.ReLU(inplace=True)]
                    prev = out
                self.mlp = nn.Sequential(*layers)

            def forward(self, xyz1, xyz2, feat1, feat2):
                """Interpola feat2 (sub-muestreado) a densidad de xyz1."""
                B, N, _ = xyz1.shape; M = xyz2.shape[1]
                if M == 1:
                    interp = feat2.expand(B, N, -1)
                else:
                    dist = sq_dist(xyz1, xyz2) + 1e-10          # (B,N,M)
                    knn3 = dist.topk(min(3,M), dim=-1, largest=False)
                    w    = 1.0 / knn3.values; w = w / w.sum(-1, keepdim=True)
                    idx3 = knn3.indices
                    interp = (index_points(feat2, idx3) * w.unsqueeze(-1)).sum(2)
                if feat1 is not None:
                    interp = torch.cat([feat1, interp], dim=-1)
                return self.mlp(interp.permute(0,2,1))   # (B,C',N)

        class PointNetPP(nn.Module):
            def __init__(self, nc):
                super().__init__()
                # Encoder — in_ch=6 porque feat=x[:,3:] tiene 6 canales
                self.sa1 = SAModule(512,  0.2, 32,  6,    [32,  32,  64])
                self.sa2 = SAModule(128,  0.4, 64,  64,   [64,  64,  128])
                self.sa3 = SAModule(64,   0.8, 64,  128,  [128, 128, 256])
                self.sa4 = SAModule(32,   1.6, 64,  256,  [256, 256, 512])
                # Decoder
                self.fp4 = FPModule(512+256,     [256, 256])
                self.fp3 = FPModule(256+128,     [256, 128])
                self.fp2 = FPModule(128+64,      [128, 128])
                self.fp1 = FPModule(128+0,       [128, 128, 128])
                self.clf = nn.Sequential(
                    nn.Conv1d(128,128,1,bias=False), nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True), nn.Dropout(0.5),
                    nn.Conv1d(128, nc, 1))
            def forward(self, x):
                xyz  = x[:,:3].permute(0,2,1)  # (B,N,3)
                feat = x[:,3:].permute(0,2,1)  # (B,N,6)
                xyz0, f0 = xyz, feat
                xyz1, f1 = self.sa1(xyz0, f0)
                xyz2, f2 = self.sa2(xyz1, f1)
                xyz3, f3 = self.sa3(xyz2, f2)
                xyz4, f4 = self.sa4(xyz3, f3)
                f3u = self.fp4(xyz3, xyz4, f3, f4)
                f2u = self.fp3(xyz2, xyz3, f2, f3u.permute(0,2,1))
                f1u = self.fp2(xyz1, xyz2, f1, f2u.permute(0,2,1))
                f0u = self.fp1(xyz0, xyz1, None, f1u.permute(0,2,1))
                return self.clf(f0u)   # (B, nc, N)

        return PointNetPP(num_classes)

    def _build_kpconv(self, num_classes: int):
        """
        KPConv simplificado (KPCNN) inline — sin extensiones C++.
        Usa convolución con kernel de puntos sobre vecindades esféricas.
        Compatible con el mismo pipeline de entrenamiento.
        """
        import torch, torch.nn as nn, torch.nn.functional as F

        def knn_idx(xyz, k):
            """KNN (B,3,N) → (B,N,k)."""
            B,_,N = xyz.shape; k = min(k, N-1)
            xy = xyz.permute(0,2,1)
            dist = (xy.unsqueeze(2) - xy.unsqueeze(1)).pow(2).sum(-1)   # (B,N,N)
            return dist.topk(k+1, dim=-1, largest=False).indices[:,:,1:] # (B,N,k)

        class KPConvLayer(nn.Module):
            """
            Simplified KPConv: para cada punto, agrega features de sus K vecinos
            ponderados por un kernel gaussiano basado en distancia relativa.
            """
            def __init__(self, in_ch, out_ch, k=16, sigma=0.1):
                super().__init__()
                self.k     = k
                self.sigma = sigma
                self.kernel = nn.Parameter(torch.randn(k, in_ch, out_ch) * 0.01)
                self.bn     = nn.BatchNorm1d(out_ch)
                self.act    = nn.LeakyReLU(0.2)

            def forward(self, xyz, feat):
                """xyz:(B,3,N), feat:(B,C,N) → (B,C',N)"""
                B, C, N = feat.shape; k = min(self.k, N-1)
                idx    = knn_idx(xyz, k)                          # (B,N,k)
                xyz_t  = xyz.permute(0,2,1)                       # (B,N,3)
                pts_k  = xyz_t.gather(1, idx.reshape(B,-1,1).expand(-1,-1,3)).reshape(B,N,k,3)
                rel    = pts_k - xyz_t.unsqueeze(2)               # (B,N,k,3)
                dist2  = rel.pow(2).sum(-1)                       # (B,N,k)
                w      = torch.exp(-dist2 / (2*self.sigma**2))    # (B,N,k) Gaussian

                # Gather features of neighbors
                feat_t = feat.permute(0,2,1)                      # (B,N,C)
                neigh  = feat_t.gather(1, idx.reshape(B,-1,1).expand(-1,-1,C)).reshape(B,N,k,C)

                # Weighted kernel: (B,N,k,C) x (k,C,C') weighted → (B,N,C')
                # Approximate: select closest kernel point per neighbor
                k_sel  = torch.arange(k, device=xyz.device).unsqueeze(0).unsqueeze(0).expand(B,N,k)
                kern   = self.kernel[k_sel]                        # (B,N,k,C,C')
                out    = (neigh.unsqueeze(-1) * kern).sum(-2)      # (B,N,k,C')
                out    = (out * w.unsqueeze(-1)).sum(2)            # (B,N,C')

                return self.act(self.bn(out.permute(0,2,1)))       # (B,C',N)

        class KPConvNet(nn.Module):
            def __init__(self, nc):
                super().__init__()
                self.enc0  = KPConvLayer(9,  32,  k=16)
                self.enc1  = KPConvLayer(32, 64,  k=16)
                self.enc2  = KPConvLayer(64, 128, k=16)
                self.enc3  = KPConvLayer(128,256, k=8)
                self.dec3  = nn.Sequential(nn.Conv1d(256+128,128,1,bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
                self.dec2  = nn.Sequential(nn.Conv1d(128+64, 64, 1,bias=False), nn.BatchNorm1d(64),  nn.LeakyReLU(0.2))
                self.dec1  = nn.Sequential(nn.Conv1d(64+32,  32, 1,bias=False), nn.BatchNorm1d(32),  nn.LeakyReLU(0.2))
                self.clf   = nn.Sequential(nn.Conv1d(32,64,1,bias=False), nn.BatchNorm1d(64),
                                           nn.LeakyReLU(0.2), nn.Dropout(0.5), nn.Conv1d(64,nc,1))
            def forward(self, x):
                xyz = x[:,:3]         # (B,3,N)
                e0  = self.enc0(xyz, x)
                e1  = self.enc1(xyz, e0)
                e2  = self.enc2(xyz, e1)
                e3  = self.enc3(xyz, e2)
                d3  = self.dec3(torch.cat([e3, e2], 1))
                d2  = self.dec2(torch.cat([d3, e1], 1))
                d1  = self.dec1(torch.cat([d2, e0], 1))
                return self.clf(d1)   # (B, nc, N)

        return KPConvNet(num_classes)

    def _train_with_model(self, model, arch, data_dir, out_dir, epochs, batch, lr,
                          pts, actual_nc, class_names, ds_mod, device):
        """
        Bucle de entrenamiento común para todas las arquitecturas.
        El modelo ya fue construido y movido al dispositivo correcto.
        """
        import torch, torch.nn as nn
        from torch.utils.data import DataLoader

        GeoAnnotateDataset = ds_mod.GeoAnnotateDataset
        get_class_weights   = ds_mod.get_class_weights
        try:
            tds = GeoAnnotateDataset(data_dir,"train",pts,augment=True)
            vds = GeoAnnotateDataset(data_dir,"val",  pts,augment=False)
            tl  = DataLoader(tds,batch,shuffle=True, num_workers=0,
                             pin_memory=(device=="cuda"))
            vl  = DataLoader(vds,batch,shuffle=False,num_workers=0)
            self.log_line.emit(f"[{arch}] Train: {len(tds)} · Val: {len(vds)}")
            if len(vds) == 0:
                self.error.emit("El conjunto de validación está vacío.\n"
                               "Exporta más datos etiquetados primero.")
                return
        except Exception as e:
            self.error.emit(f"Error cargando dataset: {e}\n"
                           "Asegúrate de haber exportado el dataset (Paso 4).")
            return

        w = get_class_weights(data_dir)
        if w is not None and len(w) != actual_nc:
            self.log_line.emit(f"[{arch}] Weight shape mismatch ({len(w)} vs {actual_nc}), ignorando")
            w = None
        crit = nn.CrossEntropyLoss(
            weight=w.float().to(device) if w is not None else None,
            ignore_index=0, reduction='mean')
        opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sch  = torch.optim.lr_scheduler.StepLR(opt, max(1, epochs//5), 0.7)
        n_params = sum(p.numel() for p in model.parameters())/1e6
        self.log_line.emit(f"[{arch}] Parámetros: {n_params:.1f}M")

        best_miou = 0.0
        best_epoch_stats = None
        best_path = str(os.path.join(out_dir, "best_model.pth"))
        start_epoch = 1

        # ── Reanudar desde checkpoint ──────────────────────────────────────
        # Antes no existía forma de continuar un entrenamiento largo tras
        # cerrar la app o un corte — solo se guardaba model_state. Ahora el
        # checkpoint también incluye optimizer/scheduler state + epoch, y
        # si el usuario eligió un checkpoint en el panel, se restaura todo
        # (no solo los pesos) para que el LR schedule continúe donde iba.
        resume_from = self._cfg.get("resume_from")
        if resume_from:
            try:
                ckpt = torch.load(resume_from, map_location=device)
                model.load_state_dict(ckpt["model_state"])
                if "optimizer_state" in ckpt:
                    opt.load_state_dict(ckpt["optimizer_state"])
                if "scheduler_state" in ckpt:
                    sch.load_state_dict(ckpt["scheduler_state"])
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                best_miou   = float(ckpt.get("miou", 0.0))
                self.log_line.emit(
                    f"[{arch}] Reanudado desde {resume_from} "
                    f"(epoch {ckpt.get('epoch','?')}, mIoU previo {best_miou:.4f})")
            except Exception as e:
                self.log_line.emit(
                    f"[WARNING] No se pudo reanudar desde {resume_from}: {e}\n"
                    f"           Empezando desde cero.")

        import time as _time
        t_start   = _time.time()
        ep_times  = []

        if start_epoch > epochs:
            self.log_line.emit(
                f"[{arch}] El checkpoint ya alcanzó epoch {start_epoch-1} "
                f">= epochs configurados ({epochs}). Sube el número de "
                f"epochs si quieres seguir entrenando.")
        for ep in range(start_epoch, epochs+1):
            if self._stop:
                self.log_line.emit(f"[{arch}] Detenido."); return

            # ── Train ──────────────────────────────────────────────────────────
            model.train(); tsum = 0.0
            for pts_t, lbl_t in tl:
                if self._stop: return
                pts_t, lbl_t = pts_t.to(device), lbl_t.long().clamp(0, actual_nc-1).to(device)
                opt.zero_grad()
                pred = model(pts_t.permute(0,2,1))
                loss = crit(pred.permute(0,2,1).reshape(-1,actual_nc), lbl_t.reshape(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); tsum += loss.item()
            sch.step()

            # ── Val ────────────────────────────────────────────────────────────
            model.eval(); vsum = 0.0
            iS = torch.zeros(actual_nc); iC = torch.zeros(actual_nc)
            # TP/FP/FN acumulados en TODO el val set (no promediados por lote
            # como iS/iC) — para el reporte por clase final (precision/recall,
            # además del IoU). Ver _write_class_report más abajo.
            tp_sum = torch.zeros(actual_nc)
            fp_sum = torch.zeros(actual_nc)
            fn_sum = torch.zeros(actual_nc)
            with torch.no_grad():
                for pts_v, lbl_v in vl:
                    pts_v, lbl_v = pts_v.to(device), lbl_v.long().clamp(0, actual_nc-1).to(device)
                    pred = model(pts_v.permute(0,2,1))
                    flat_p = pred.permute(0,2,1).reshape(-1, actual_nc)
                    flat_l = lbl_v.reshape(-1)
                    vsum  += crit(flat_p, flat_l).item()
                    pc, lc = flat_p.argmax(1).cpu(), flat_l.cpu()
                    for c in range(1, actual_nc):
                        mp = pc==c; ml = lc==c
                        i  = (mp & ml).sum().float()
                        u  = (mp | ml).sum().float()
                        if u > 0: iS[c] += i/u; iC[c] += 1
                        tp_sum[c] += (mp & ml).sum().float()
                        fp_sum[c] += (mp & ~ml).sum().float()
                        fn_sum[c] += (~mp & ml).sum().float()

            v     = iC > 0
            miou  = float((iS[v]/iC[v]).mean()) if v.any() else 0.0
            tl_a  = tsum / max(len(tl), 1)
            vl_a  = vsum / max(len(vl), 1)

            # ETA calculation — dividir por epochs corridos EN ESTA SESIÓN
            # (ep - start_epoch + 1), no por el número absoluto de epoch
            # `ep`: al reanudar desde un checkpoint, t_start es el inicio de
            # ESTA sesión, así que dividir por `ep` (que puede ya venir alto)
            # subestimaría gravemente el tiempo promedio por epoch.
            ep_times.append(_time.time() - t_start)
            ran_this_session = ep - start_epoch + 1
            avg_ep  = ep_times[-1] / max(1, ran_this_session)
            remaining = avg_ep * (epochs - ep)
            h, r  = divmod(int(remaining), 3600)
            m, s  = divmod(r, 60)
            eta   = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
            elapsed = _time.time() - t_start
            eh, er = divmod(int(elapsed), 3600)
            em, es = divmod(er, 60)
            ela   = f"{eh:02d}:{em:02d}:{es:02d}" if eh > 0 else f"{em:02d}:{es:02d}"

            self.log_line.emit(
                f"Ep {ep:3d}/{epochs}  train={tl_a:.4f}  val={vl_a:.4f}  "
                f"mIoU={miou:.4f}  lr={opt.param_groups[0]['lr']:.5f}  "
                f"[{ela} / ETA {eta}]")
            self.epoch_done.emit(tl_a, vl_a, miou)
            self.progress.emit(ep, epochs)

            if miou > best_miou:
                best_miou = miou
                best_epoch_stats = {"tp": tp_sum.clone(), "fp": fp_sum.clone(), "fn": fn_sum.clone()}
                import torch as _t
                _t.save({"epoch":ep,"model_state":model.state_dict(),
                         "optimizer_state":opt.state_dict(),
                         "scheduler_state":sch.state_dict(),
                         "miou":miou,"num_classes":actual_nc,
                         "class_names":class_names,"arch":arch}, best_path)
                self.log_line.emit(f"  → Guardado mIoU={miou:.4f}: {best_path}")

        self.log_line.emit(f"\n[{arch}] Entrenamiento completo. Mejor mIoU: {best_miou:.4f}")
        if best_epoch_stats is not None:
            try:
                report_path = self._write_class_report(
                    out_dir, class_names, best_epoch_stats, arch, best_miou)
                self.log_line.emit(f"  → Reporte por clase: {report_path}")
            except Exception as e:
                self.log_line.emit(f"  [WARNING] No se pudo escribir el reporte por clase: {e}")
        self.finished_ok.emit(best_path)

    def _write_class_report(self, out_dir, class_names, stats, arch, best_miou):
        """
        Escribe un reporte de precision/recall/IoU por clase del MEJOR
        epoch (no solo el mIoU agregado) — antes el usuario no tenía
        forma de saber QUÉ clase necesitaba más datos etiquetados,
        solo un número global. Escribe .json (para procesar) y .txt
        (para leer directamente).
        """
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        rows = []
        # class 0 (sin clasificar / ignore_index) se excluye, igual que
        # en el cálculo de mIoU de arriba (range(1, actual_nc))
        for c in range(1, len(tp)):
            tp_c, fp_c, fn_c = float(tp[c]), float(fp[c]), float(fn[c])
            denom_iou = tp_c + fp_c + fn_c
            iou  = tp_c / denom_iou if denom_iou > 0 else None
            prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else None
            rec  = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else None
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            rows.append({"class_id": c, "name": name, "iou": iou,
                        "precision": prec, "recall": rec,
                        "support_points": int(tp_c + fn_c)})

        data = {"arch": arch, "best_miou": best_miou, "per_class": rows}
        json_path = os.path.join(out_dir, "class_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        txt_path = os.path.join(out_dir, "class_report.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Reporte por clase — {arch}  (mIoU global: {best_miou:.4f})\n")
            f.write("=" * 70 + "\n")
            f.write(f"{'Clase':<22}{'IoU':>8}{'Precision':>12}{'Recall':>10}{'Puntos':>12}\n")
            for r in rows:
                def fmt(v): return f"{v:.4f}" if v is not None else "  n/a"
                f.write(f"{r['name']:<22}{fmt(r['iou']):>8}{fmt(r['precision']):>12}"
                       f"{fmt(r['recall']):>10}{r['support_points']:>12,}\n")
            low = [r for r in rows if r['iou'] is not None and r['iou'] < 0.5]
            if low:
                f.write("\nClases con IoU < 0.50 (probablemente necesitan más datos etiquetados):\n")
                for r in low:
                    f.write(f"  - {r['name']}: IoU={r['iou']:.4f}\n")
        return txt_path

    def _train_pointnet(self, data_dir, out_dir, epochs, batch, lr, pts,
                        num_classes, class_names, device):
        import torch, importlib.util as _ilu
        from pathlib import Path
        ds_mod = self._load_ds_module(data_dir)
        if ds_mod is None: return
        actual_nc = getattr(ds_mod, 'NUM_CLASSES', num_classes)
        self.log_line.emit(f"[PointNet++] Construyendo modelo ({actual_nc} clases)…")
        model = self._build_pointnetpp(actual_nc).to(device)
        self._train_with_model(model, "PointNet++", data_dir, out_dir, epochs, batch,
                               lr, pts, actual_nc, class_names, ds_mod, device)

    def _train_kpconv(self, data_dir, out_dir, epochs, batch, lr, pts,
                      num_classes, class_names, device):
        import torch, importlib.util as _ilu
        from pathlib import Path
        ds_mod = self._load_ds_module(data_dir)
        if ds_mod is None: return
        actual_nc = getattr(ds_mod, 'NUM_CLASSES', num_classes)
        self.log_line.emit(f"[KPConv] Construyendo modelo ({actual_nc} clases)…")
        model = self._build_kpconv(actual_nc).to(device)
        self._train_with_model(model, "KPConv", data_dir, out_dir, epochs, batch,
                               lr, pts, actual_nc, class_names, ds_mod, device)

    def _load_ds_module(self, data_dir):
        """Helper: carga custom_dataset.py desde data_dir o sus padres."""
        from pathlib import Path
        import importlib.util as _ilu
        for sp in [data_dir, str(Path(data_dir).parent), str(Path(data_dir).parent.parent)]:
            cand = Path(sp) / "custom_dataset.py"
            if cand.exists():
                spec = _ilu.spec_from_file_location("custom_dataset", cand)
                mod  = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self.log_line.emit(f"[Dataset] Módulo cargado: {cand}")
                return mod
        self.error.emit("No se encontró custom_dataset.py\n\n"
                       "Exporta el dataset primero (Paso 4).")
        return None



# ── Panel principal ────────────────────────────────────────────────────────────

class TrainingPanel(QWidget):
    """
    Widget que reemplaza el canvas durante el entrenamiento.
    Dividido en dos zonas:
      Izquierda: configuración + log
      Derecha:   gráfica de curvas + métricas
    """
    training_finished = pyqtSignal(str)   # path al mejor modelo
    training_stopped  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[TrainingWorker] = None
        self._start_time = 0.0
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_elapsed)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setStyleSheet(f"background:{SURFACE_2};")

        # Header
        hdr = QWidget()
        hdr.setStyleSheet(f"background:{SURFACE};border-bottom:1px solid {BORDER};")
        hdr.setFixedHeight(38)
        hl = QHBoxLayout(hdr); hl.setContentsMargins(14, 0, 14, 0); hl.setSpacing(8)
        ic = QLabel(); ic.setPixmap(qpixmap("cpu-fill", ACCENT_STRONG, 15))
        hl.addWidget(ic)
        title = QLabel("Entrenamiento de red neuronal")
        title.setStyleSheet(f"color:{ACCENT_STRONG};font-size:11.5px;font-weight:700;")
        hl.addWidget(title); hl.addStretch()
        self._elapsed_lbl = QLabel("00:00:00")
        self._elapsed_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;font-family:'Consolas';")
        hl.addWidget(self._elapsed_lbl)
        root.addWidget(hdr)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(f"QSplitter::handle{{background:{BORDER_SOFT};}}")

        # Left: config + log
        left = QWidget(); left.setStyleSheet(f"background:{SURFACE_2};")
        left.setMinimumWidth(300)
        ll = QVBoxLayout(left); ll.setContentsMargins(12,12,12,12); ll.setSpacing(10)

        # Config card
        cfg_gb, cgl = self._card("Configuración", "sliders2")

        # Architecture
        cgl.addLayout(self._row("Arquitectura", self._make_arch_combo()))
        # Data dir
        self._data_dir_lbl = QLabel("(no seleccionado)")
        self._data_dir_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;")
        self._data_dir_lbl.setWordWrap(True)
        data_row = QHBoxLayout(); data_row.setSpacing(6)
        dset_lbl = QLabel("Dataset"); dset_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;min-width:66px;")
        data_row.addWidget(dset_lbl)
        data_row.addWidget(self._data_dir_lbl, 1)
        browse_btn = self._icon_btn("folder2")
        browse_btn.clicked.connect(self._browse_data)
        data_row.addWidget(browse_btn)
        cgl.addLayout(data_row)

        # Out dir
        self._out_dir_lbl = QLabel("(junto al dataset)")
        self._out_dir_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;")
        self._out_dir_lbl.setWordWrap(True)
        out_row = QHBoxLayout(); out_row.setSpacing(6)
        out_lbl = QLabel("Salida"); out_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;min-width:66px;")
        out_row.addWidget(out_lbl)
        out_row.addWidget(self._out_dir_lbl, 1)
        out_btn = self._icon_btn("folder2")
        out_btn.clicked.connect(self._browse_out)
        out_row.addWidget(out_btn)
        cgl.addLayout(out_row)

        cgl.addLayout(self._row("Epochs", self._spin(50, 1, 1000)))
        self._epochs_spin = cgl.itemAt(cgl.count()-1).layout().itemAt(1).widget()
        cgl.addLayout(self._row("Batch size", self._spin(2, 1, 32)))
        self._batch_spin = cgl.itemAt(cgl.count()-1).layout().itemAt(1).widget()
        cgl.addLayout(self._row("Pts/muestra", self._spin(8192, 512, 65536, step=512)))
        self._pts_spin = cgl.itemAt(cgl.count()-1).layout().itemAt(1).widget()
        cgl.addLayout(self._row("LR inicial", self._dspin(0.01, 1e-5, 0.1)))
        self._lr_spin = cgl.itemAt(cgl.count()-1).layout().itemAt(1).widget()

        # Reanudar desde checkpoint — antes no existía forma de continuar
        # un entrenamiento largo tras cerrar la app o un corte de luz;
        # todo el progreso se perdía salvo por el último best_model.pth,
        # que solo servía como punto de partida de inferencia, no para
        # seguir entrenando (sin optimizer/scheduler state ni epoch).
        self._resume_path = None
        resume_row = QHBoxLayout(); resume_row.setSpacing(6)
        resume_lbl = QLabel("Reanudar"); resume_lbl.setStyleSheet(
            f"color:{TEXT_MUTE};font-size:10.5px;min-width:74px;")
        resume_row.addWidget(resume_lbl)
        self._resume_lbl = QLabel("(entrenamiento nuevo)")
        self._resume_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;")
        self._resume_lbl.setWordWrap(True)
        resume_row.addWidget(self._resume_lbl, 1)
        resume_btn = QPushButton()
        resume_btn.setIcon(qicon("folder2", TEXT_DIM)); resume_btn.setFixedSize(26, 24)
        resume_btn.setToolTip("Elegir un checkpoint (.pth) para continuar su entrenamiento")
        resume_btn.setStyleSheet(
            f"QPushButton{{background:{SURFACE_2};border:1px solid {BORDER};"
            f"border-radius:3px;}}"
            f"QPushButton:hover{{border-color:{ACCENT};background:{ACCENT_SOFT};}}")
        resume_btn.clicked.connect(self._browse_resume_checkpoint)
        resume_row.addWidget(resume_btn)
        cgl.addLayout(resume_row)

        ll.addWidget(cfg_gb)

        # Buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(6)
        self._start_btn = QPushButton("  Iniciar entrenamiento")
        self._start_btn.setIcon(qicon("play-fill", "#ffffff"))
        self._start_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:#ffffff;border:none;"
            f"border-radius:4px;padding:9px;font-size:10.5px;font-weight:600;text-align:left;}}"
            f"QPushButton:hover{{background:{ACCENT_STRONG};}}"
            f"QPushButton:disabled{{background:{ACCENT_SOFT};color:{TEXT_MUTE};}}")
        self._start_btn.clicked.connect(self._on_start)

        self._stop_btn = QPushButton("  Detener")
        self._stop_btn.setIcon(qicon("stop-fill", WARN))
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            f"QPushButton{{background:{WARN_SOFT};border:1px solid {WARN_SOFT};"
            f"border-radius:4px;padding:9px;font-size:10.5px;color:{WARN};font-weight:600;}}"
            f"QPushButton:hover{{border-color:{WARN};}}"
            f"QPushButton:disabled{{color:{TEXT_MUTE};background:{SURFACE_2};border-color:{SURFACE_2};}}")
        self._stop_btn.clicked.connect(self._on_stop)

        btn_row.addWidget(self._start_btn, 1)
        btn_row.addWidget(self._stop_btn)
        ll.addLayout(btn_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100); self._progress.setValue(0)
        self._progress.setFixedHeight(6)
        self._progress.setStyleSheet(
            f"QProgressBar{{background:{SURFACE_3};border:none;border-radius:3px;}}"
            f"QProgressBar::chunk{{background:{ACCENT};border-radius:3px;}}")
        self._progress.setTextVisible(False)
        ll.addWidget(self._progress)

        # Log
        log_gb, logl = self._card("Log de entrenamiento", "info-circle")
        logl.setContentsMargins(0,0,0,0)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(120)
        self._log.setStyleSheet(
            f"QTextEdit{{background:{SURFACE_2};border:1px solid {BORDER_SOFT};border-radius:3px;"
            f"color:{TEXT_DIM};font-family:'Consolas';font-size:10.5px;padding:5px;}}")
        logl.addWidget(self._log)
        ll.addWidget(log_gb, 1)

        splitter.addWidget(left)

        # Right: chart + metrics
        right = QWidget(); right.setStyleSheet(f"background:{SURFACE_2};")
        rl = QVBoxLayout(right); rl.setContentsMargins(12,12,12,12); rl.setSpacing(10)

        chart_gb, chartl = self._card("Curvas de entrenamiento", "graph-up")
        self._chart = _LossChart()
        chartl.addWidget(self._chart)

        # Legend
        leg = QHBoxLayout(); leg.setSpacing(14)
        for col, txt in [(ACCENT,"train loss"),(ACCENT_STRONG,"val loss"),(OK,"mIoU")]:
            dot = QLabel("━━")
            dot.setStyleSheet(f"color:{col};font-size:10.5px;")
            lbl = QLabel(txt); lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;")
            leg.addWidget(dot); leg.addWidget(lbl)
        leg.addStretch()
        chartl.addLayout(leg)
        rl.addWidget(chart_gb, 1)

        # Metrics
        metrics_gb, ml = self._card("Métricas actuales", "bullseye")
        ml_row = QHBoxLayout(); ml_row.setSpacing(16)
        self._m_epoch = self._metric_lbl("Epoch", "—")
        self._m_train = self._metric_lbl("Train Loss", "—")
        self._m_val   = self._metric_lbl("Val Loss", "—")
        self._m_miou  = self._metric_lbl("mIoU", "—")
        self._m_best  = self._metric_lbl("Mejor mIoU", "—")
        for w in [self._m_epoch, self._m_train, self._m_val, self._m_miou, self._m_best]:
            ml_row.addWidget(w, 1)
        ml.addLayout(ml_row)
        rl.addWidget(metrics_gb)

        splitter.addWidget(right)
        splitter.setSizes([320, 600])
        root.addWidget(splitter, 1)

    # ── Helpers UI ────────────────────────────────────────────────────────────

    def _card(self, title, icon_name=None):
        """Tarjeta con encabezado que envuelve — mismo patrón que ToolPanel/GeoPanel."""
        frame = QFrame(); frame.setObjectName("trainCard")
        frame.setStyleSheet(
            f"QFrame#trainCard{{background:{SURFACE};border:1px solid {BORDER};border-radius:4px;}}")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(11, 10, 11, 11); outer.setSpacing(7)
        hdr_row = QHBoxLayout(); hdr_row.setSpacing(6)
        if icon_name:
            ic = QLabel(); ic.setStyleSheet("background:transparent;border:none;")
            ic.setPixmap(qpixmap(icon_name, ACCENT_STRONG, 13))
            hdr_row.addWidget(ic)
        title_lbl = QLabel(title); title_lbl.setWordWrap(True)
        title_lbl.setStyleSheet(f"color:{ACCENT_STRONG};font-size:10.5px;font-weight:700;background:transparent;")
        hdr_row.addWidget(title_lbl, 1)
        outer.addLayout(hdr_row)
        content = QVBoxLayout(); content.setSpacing(6)
        outer.addLayout(content)
        return frame, content

    def _icon_btn(self, icon_name):
        b = QPushButton(); b.setIcon(qicon(icon_name, TEXT_DIM)); b.setFixedSize(26, 24)
        b.setStyleSheet(
            f"QPushButton{{background:{SURFACE_2};border:1px solid {BORDER};"
            f"border-radius:3px;}}"
            f"QPushButton:hover{{border-color:{ACCENT};background:{ACCENT_SOFT};}}")
        return b

    def _row(self, label, widget):
        hl = QHBoxLayout(); hl.setSpacing(6)
        lbl = QLabel(label); lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;min-width:74px;")
        lbl.setWordWrap(True)
        hl.addWidget(lbl); hl.addWidget(widget, 1)
        return hl

    def _spin(self, default, mn, mx, step=1):
        s = QSpinBox()
        s.setRange(mn, mx); s.setValue(default); s.setSingleStep(step)
        s.setStyleSheet(
            f"QSpinBox{{background:{SURFACE_2};border:1px solid {BORDER};"
            f"border-radius:3px;color:{TEXT_DIM};padding:3px 5px;font-size:10.5px;}}"
            f"QSpinBox:hover{{border-color:{ACCENT};}}")
        return s

    def _dspin(self, default, mn, mx):
        s = QDoubleSpinBox()
        s.setRange(mn, mx); s.setValue(default); s.setDecimals(5); s.setSingleStep(0.001)
        s.setStyleSheet(
            f"QDoubleSpinBox{{background:{SURFACE_2};border:1px solid {BORDER};"
            f"border-radius:3px;color:{TEXT_DIM};padding:3px 5px;font-size:10.5px;}}"
            f"QDoubleSpinBox:hover{{border-color:{ACCENT};}}")
        return s

    def _make_arch_combo(self):
        self._arch_combo = QComboBox()
        for a in ["RandLA-Net", "PointNet++", "KPConv"]:
            self._arch_combo.addItem(a)
        self._arch_combo.setStyleSheet(
            f"QComboBox{{background:{SURFACE_2};border:1px solid {BORDER};border-radius:3px;"
            f"color:{TEXT_DIM};padding:3px 6px;font-size:10.5px;}}"
            f"QComboBox:hover{{border-color:{ACCENT};}}"
            f"QComboBox QAbstractItemView{{background:{SURFACE};color:{TEXT_DIM};"
            f"selection-background-color:{ACCENT_SOFT};selection-color:{ACCENT_STRONG};}}")
        return self._arch_combo

    def _metric_lbl(self, title, value):
        w = QWidget(); w.setStyleSheet("background:transparent;")
        l = QVBoxLayout(w); l.setContentsMargins(0,0,0,0); l.setSpacing(2)
        t = QLabel(title); t.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;")
        t.setAlignment(Qt.AlignCenter); t.setWordWrap(True)
        v = QLabel(value); v.setStyleSheet(f"color:{ACCENT_STRONG};font-size:14px;font-weight:700;")
        v.setAlignment(Qt.AlignCenter)
        v.setObjectName("val")
        l.addWidget(t); l.addWidget(v)
        return w

    def _set_metric(self, widget, value: str):
        widget.findChild(QLabel, "val").setText(value)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _browse_resume_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar checkpoint para reanudar", "",
            "Checkpoints PyTorch (*.pth *.pt);;Todos (*)")
        if path:
            self._resume_path = path
            self._resume_lbl.setText(f"…/{Path(path).parent.name}/{Path(path).name}")
            self._resume_lbl.setToolTip(path)
        # (sin "else": si el usuario cancela el diálogo, se conserva la
        # selección previa — no se limpia _resume_path por accidente)

    def _browse_data(self):
        d = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta del dataset")
        if d:
            p = Path(d)
            # Si el usuario seleccionó data/, subir un nivel para encontrar custom_dataset.py
            # Estructura esperada: export/custom_dataset.py + export/data/train|val|test
            if p.name == "data" and (p.parent / "custom_dataset.py").exists():
                # auto-use: data_dir = data/, python file in parent
                self._data_dir = str(p)
            elif (p / "data").is_dir():
                # seleccionó la carpeta raíz del export → usar data/ dentro
                self._data_dir = str(p / "data")
                d = str(p / "data")
            else:
                self._data_dir = str(p)
            short = Path(self._data_dir).name
            self._data_dir_lbl.setText(f"…/{Path(self._data_dir).parent.name}/{short}")
            self._data_dir_lbl.setToolTip(self._data_dir)
            # Auto-set out dir next to data/
            self._out_dir = str(Path(self._data_dir).parent / "checkpoints")
            self._out_dir_lbl.setText(f"…/checkpoints")

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de salida")
        if d:
            self._out_dir = d
            self._out_dir_lbl.setText(f"…/{Path(d).name}")

    def _on_start(self):
        data_dir = getattr(self, '_data_dir', None)
        if not data_dir:
            self._log.append("[ERROR] Selecciona la carpeta del dataset primero.")
            return
        out_dir = getattr(self, '_out_dir', str(Path(data_dir) / "checkpoints"))
        os.makedirs(out_dir, exist_ok=True)

        cfg = {
            "arch":        self._arch_combo.currentText(),
            "data_dir":    data_dir,
            "out_dir":     out_dir,
            "epochs":      self._epochs_spin.value(),
            "batch":       self._batch_spin.value(),
            "lr":          self._lr_spin.value(),
            "pts":         self._pts_spin.value(),
            "num_classes": getattr(self, '_num_classes', 13),
            "class_names": getattr(self, '_class_names', []),
            "resume_from": getattr(self, '_resume_path', None),
        }

        self._log.clear()
        self._chart.reset()
        self._best_miou = 0.0
        self._current_epoch = 0
        self._start_time = time.time()
        self._timer.start(1000)

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress.setValue(0)

        self._worker = TrainingWorker(cfg)
        self._worker.log_line.connect(self._on_log)
        self._worker.epoch_done.connect(self._on_epoch)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker: self._worker.stop()
        self._stop_btn.setEnabled(False)
        self._log.append("[Entrenamiento] Deteniendo…")

    def _on_log(self, line: str):
        self._log.append(line)
        self._log.moveCursor(self._log.textCursor().End)

    def _on_epoch(self, tl: float, vl: float, miou: float):
        self._chart.add_epoch(tl, vl, miou)
        self._current_epoch += 1
        self._set_metric(self._m_epoch, str(self._current_epoch))
        self._set_metric(self._m_train, f"{tl:.4f}")
        self._set_metric(self._m_val,   f"{vl:.4f}")
        self._set_metric(self._m_miou,  f"{miou:.4f}")
        if miou > getattr(self, '_best_miou', 0):
            self._best_miou = miou
            self._set_metric(self._m_best, f"{miou:.4f}")

    def _on_progress(self, ep: int, total: int):
        self._progress.setValue(int(100 * ep / max(total, 1)))

    def _on_finished(self, path: str):
        self._timer.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setValue(100)
        self._log.append(f"\n✓ Entrenamiento completado. Modelo guardado en:\n{path}")
        self.training_finished.emit(path)

    def _on_error(self, msg: str):
        self._timer.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._log.append(f"\n[ERROR]\n{msg}")

    def _update_elapsed(self):
        elapsed = int(time.time() - self._start_time)
        h, r = divmod(elapsed, 3600)
        m, s = divmod(r, 60)
        self._elapsed_lbl.setText(f"{h:02d}:{m:02d}:{s:02d}")

    # ── API pública ────────────────────────────────────────────────────────────

    def set_project_context(self, project):
        """Recibe el proyecto para pre-configurar clases y dataset."""
        if project is None: return
        self._num_classes = len(project.schema) + 1  # +1 for unlabeled
        self._class_names = [s.name for s in project.schema]
        # Auto-detect dataset dir next to source_file
        if project.source_file:
            candidate = Path(project.source_file).parent / "randlanet" / "data"
            if candidate.exists():
                self._data_dir = str(candidate)
                self._data_dir_lbl.setText(f"…/randlanet/data")
                self._data_dir_lbl.setToolTip(str(candidate))
                self._out_dir = str(candidate.parent / "checkpoints")
                self._out_dir_lbl.setText("…/randlanet/checkpoints")
