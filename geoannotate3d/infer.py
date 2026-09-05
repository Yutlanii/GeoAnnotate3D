"""
infer.py — Inferencia de nube de puntos completa con RandLA-Net

Uso:
    python infer.py --model checkpoints/best_model.pth \
                    --input  mi_nube.las \
                    --output mi_nube_clasificada.las \
                    [--pts 8192] [--overlap 0.5] [--batch 4] [--device cuda]

Proceso:
    1. Carga la nube completa (LAS/LAZ/PLY/NPY)
    2. Divide en parches de N puntos con overlap (como YOLO hace con imágenes grandes)
    3. Corre el modelo en cada parche
    4. Combina predicciones por votación de confianza (suaviza bordes entre parches)
    5. Guarda el resultado como .las con campo 'classification' para GeoAnnotate3D
"""
from __future__ import annotations
import argparse
import sys
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

# ── Argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Inferencia de nube de puntos con modelo RandLA-Net entrenado")
    p.add_argument("--model",  required=True,
                   help="Ruta al checkpoint .pth (ej: checkpoints/best_model.pth)")
    p.add_argument("--input",  required=True,
                   help="Nube de puntos de entrada (.las .laz .ply .npy .txt)")
    p.add_argument("--output", required=False, default=None,
                   help="Archivo de salida (.las). Por defecto: input_classified.las")
    p.add_argument("--pts",    type=int,   default=8192,
                   help="Puntos por parche (default 8192, igual que en entrenamiento)")
    p.add_argument("--overlap",type=float, default=0.5,
                   help="Fracción de solapamiento entre parches 0-0.9 (default 0.5)")
    p.add_argument("--batch",  type=int,   default=4,
                   help="Parches por batch en GPU (default 4, bajar si VRAM insuficiente)")
    p.add_argument("--device", default="auto",
                   choices=["auto","cuda","cpu"],
                   help="Dispositivo de inferencia (default auto)")
    p.add_argument("--min_pts",type=int, default=100,
                   help="Puntos mínimos en un parche para procesarlo (default 100)")
    return p.parse_args()


# ── Carga de nube ─────────────────────────────────────────────────────────────

def load_cloud(path: str) -> dict:
    """
    Carga una nube de puntos y devuelve dict con xyz, intensity, rgb (si existe).
    Soporta: .las .laz .ply .npy .txt .csv .xyz .pts .asc
    """
    p = Path(path)
    ext = p.suffix.lower()
    print(f"[Carga] {path} ({p.stat().st_size / 1e6:.1f} MB)")

    if ext in (".las", ".laz"):
        try:
            import laspy
        except ImportError:
            sys.exit("laspy no instalado: pip install laspy[lazrs]")
        las = laspy.read(path)
        xyz = np.column_stack([
            np.array(las.x, dtype=np.float64),
            np.array(las.y, dtype=np.float64),
            np.array(las.z, dtype=np.float64),
        ])
        try:
            intensity = np.array(las.intensity, dtype=np.float32) / 65535.0
        except Exception:
            intensity = np.zeros(len(xyz), dtype=np.float32)
        try:
            rgb = np.column_stack([
                np.array(las.red,   dtype=np.float32) / 65535.0,
                np.array(las.green, dtype=np.float32) / 65535.0,
                np.array(las.blue,  dtype=np.float32) / 65535.0,
            ])
        except Exception:
            rgb = None
        # Try to get existing classification
        try:
            existing_cls = np.array(las.classification, dtype=np.uint8)
        except Exception:
            existing_cls = np.zeros(len(xyz), dtype=np.uint8)
        return {"xyz": xyz, "intensity": intensity, "rgb": rgb,
                "existing_cls": existing_cls, "las": las}

    elif ext == ".ply":
        try:
            from plyfile import PlyData
            ply = PlyData.read(path)
            v = ply["vertex"]
            xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
            intensity = (np.array(v["intensity"], dtype=np.float32) / 65535.0
                        if "intensity" in v.data.dtype.names else
                        np.zeros(len(xyz), dtype=np.float32))
        except ImportError:
            sys.exit("plyfile no instalado: pip install plyfile")
        return {"xyz": xyz, "intensity": intensity, "rgb": None,
                "existing_cls": np.zeros(len(xyz), dtype=np.uint8)}

    elif ext == ".npy":
        data = np.load(path)
        if data.ndim == 2 and data.shape[1] >= 3:
            xyz = data[:, :3].astype(np.float64)
            intensity = data[:, 3].astype(np.float32) if data.shape[1] > 3 else \
                        np.zeros(len(xyz), dtype=np.float32)
        else:
            sys.exit(f"Formato NPY no reconocido: shape {data.shape}")
        return {"xyz": xyz, "intensity": intensity, "rgb": None,
                "existing_cls": np.zeros(len(xyz), dtype=np.uint8)}

    elif ext in (".txt", ".csv", ".xyz", ".pts", ".asc"):
        try:
            data = np.loadtxt(path, comments=["//", "#"])
        except Exception:
            data = np.genfromtxt(path, invalid_raise=False)
        xyz = data[:, :3].astype(np.float64)
        intensity = data[:, 3].astype(np.float32) if data.shape[1] > 3 else \
                    np.zeros(len(xyz), dtype=np.float32)
        return {"xyz": xyz, "intensity": intensity, "rgb": None,
                "existing_cls": np.zeros(len(xyz), dtype=np.uint8)}

    else:
        sys.exit(f"Formato no soportado: {ext}")


# ── Carga de modelo ───────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str):
    """
    Carga el checkpoint y reconstruye el modelo RandLA-Net con las clases del checkpoint.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = ckpt.get("num_classes", 13)
    class_names = ckpt.get("class_names", [f"clase_{i}" for i in range(num_classes)])
    arch        = ckpt.get("arch", "RandLA-Net")
    ep          = ckpt.get("epoch", "?")
    miou        = ckpt.get("miou", 0.0)

    print(f"[Modelo] {arch} | epoch {ep} | best mIoU {miou:.4f}")
    print(f"[Modelo] {num_classes} clases: {class_names}")

    # ── Reconstruir la arquitectura según el checkpoint ──────────────────────
    arch_name = ckpt.get("arch", "RandLA-Net")
    print(f"[Modelo] Arquitectura detectada: {arch_name}")

    # ── Import builders from training_panel if available ─────────────────────
    import sys, os
    _tp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
    if _tp_dir not in sys.path:
        sys.path.insert(0, _tp_dir)

    # ── Inline RandLA-Net ─────────────────────────────────────────────────────
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

    # ── PointNet++ inline ────────────────────────────────────────────────────
    def sq_dist(src, dst):
        return (src.unsqueeze(2) - dst.unsqueeze(1)).pow(2).sum(-1)

    def farthest_point_sample(xyz, n_pts):
        B, N, _ = xyz.shape; device = xyz.device
        sel = torch.zeros(B, n_pts, dtype=torch.long, device=device)
        dist= torch.full((B, N), float('inf'), device=device)
        cur = torch.zeros(B, dtype=torch.long, device=device)
        for i in range(n_pts):
            sel[:,i] = cur
            dp = (xyz - xyz[torch.arange(B),cur,:].unsqueeze(1)).pow(2).sum(-1)
            dist = torch.min(dist, dp); cur = dist.argmax(1)
        return sel

    def index_points(pts, idx):
        B = pts.shape[0]
        if idx.dim() == 2:
            return pts.gather(1, idx.unsqueeze(-1).expand(-1,-1,pts.shape[-1]))
        M,K = idx.shape[1],idx.shape[2]
        return pts.gather(1,idx.reshape(B,-1,1).expand(-1,-1,pts.shape[-1])).reshape(B,M,K,-1)

    def ball_query(xyz, new_xyz, radius, k):
        """Vectorized — sin bucles Python."""
        dist   = sq_dist(new_xyz, xyz)
        masked = dist.clone(); masked[dist > radius*radius] = 1e10
        topk   = masked.topk(k, dim=-1, largest=False)
        idx    = topk.indices
        first  = idx[:,:,0:1].expand_as(idx)
        return torch.where(topk.values < 1e9, idx, first)

    class SAModule(nn.Module):
        def __init__(self,n_pts,radius,k,in_ch,mlp_chs):
            super().__init__(); self.n_pts=n_pts; self.radius=radius; self.k=k
            layers=[]; prev=in_ch+3
            for out in mlp_chs:
                layers+=[nn.Conv2d(prev,out,1,bias=False),nn.BatchNorm2d(out),nn.ReLU(True)]; prev=out
            self.mlp=nn.Sequential(*layers)
        def forward(self,xyz,feat):
            B,N,_=xyz.shape; n=min(self.n_pts,N)
            fps_idx=farthest_point_sample(xyz,n); new_xyz=index_points(xyz,fps_idx)
            bq_idx=ball_query(xyz,new_xyz,self.radius,self.k)
            grouped=index_points(xyz,bq_idx)-new_xyz.unsqueeze(2)
            if feat is not None: grouped=torch.cat([grouped,index_points(feat,bq_idx)],dim=-1)
            x=grouped.permute(0,3,1,2); x=self.mlp(x).max(-1)[0]
            return new_xyz,x.permute(0,2,1)

    class FPModule(nn.Module):
        def __init__(self,in_ch,mlp_chs):
            super().__init__(); layers=[]; prev=in_ch
            for out in mlp_chs:
                layers+=[nn.Conv1d(prev,out,1,bias=False),nn.BatchNorm1d(out),nn.ReLU(True)]; prev=out
            self.mlp=nn.Sequential(*layers)
        def forward(self,xyz1,xyz2,feat1,feat2):
            B,N,_=xyz1.shape; M=xyz2.shape[1]
            if M==1: interp=feat2.expand(B,N,-1)
            else:
                dist=sq_dist(xyz1,xyz2)+1e-10; knn3=dist.topk(min(3,M),dim=-1,largest=False)
                w=1.0/knn3.values; w=w/w.sum(-1,keepdim=True)
                interp=(index_points(feat2,knn3.indices)*w.unsqueeze(-1)).sum(2)
            if feat1 is not None: interp=torch.cat([feat1,interp],dim=-1)
            return self.mlp(interp.permute(0,2,1))

    class PointNetPP(nn.Module):
        def __init__(self,nc):
            super().__init__()
            self.sa1=SAModule(512,0.2,32,6,[32,32,64]); self.sa2=SAModule(128,0.4,64,64,[64,64,128])
            self.sa3=SAModule(64,0.8,64,128,[128,128,256]); self.sa4=SAModule(32,1.6,64,256,[256,256,512])
            self.fp4=FPModule(512+256,[256,256]); self.fp3=FPModule(256+128,[256,128])
            self.fp2=FPModule(128+64,[128,128]); self.fp1=FPModule(128,[128,128,128])
            self.clf=nn.Sequential(nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),
                                   nn.ReLU(True),nn.Dropout(0.5),nn.Conv1d(128,nc,1))
        def forward(self,x):
            xyz=x[:,:3].permute(0,2,1); feat=x[:,3:].permute(0,2,1)
            xyz1,f1=self.sa1(xyz,feat); xyz2,f2=self.sa2(xyz1,f1)
            xyz3,f3=self.sa3(xyz2,f2); xyz4,f4=self.sa4(xyz3,f3)
            f3u=self.fp4(xyz3,xyz4,f3,f4); f2u=self.fp3(xyz2,xyz3,f2,f3u.permute(0,2,1))
            f1u=self.fp2(xyz1,xyz2,f1,f2u.permute(0,2,1)); f0u=self.fp1(xyz,xyz1,None,f1u.permute(0,2,1))
            return self.clf(f0u)

    # ── KPConv inline ─────────────────────────────────────────────────────────
    def knn_idx_kp(xyz, k):
        B,_,N=xyz.shape; k=min(k,N-1); xy=xyz.permute(0,2,1)
        dist=(xy.unsqueeze(2)-xy.unsqueeze(1)).pow(2).sum(-1)
        return dist.topk(k+1,dim=-1,largest=False).indices[:,:,1:]

    class KPConvLayer(nn.Module):
        def __init__(self,in_ch,out_ch,k=16,sigma=0.1):
            super().__init__(); self.k=k; self.sigma=sigma
            self.kernel=nn.Parameter(torch.randn(k,in_ch,out_ch)*0.01)
            self.bn=nn.BatchNorm1d(out_ch); self.act=nn.LeakyReLU(0.2)
        def forward(self,xyz,feat):
            B,C,N=feat.shape; k=min(self.k,N-1)
            idx=knn_idx_kp(xyz,k); xyz_t=xyz.permute(0,2,1)
            pts_k=xyz_t.gather(1,idx.reshape(B,-1,1).expand(-1,-1,3)).reshape(B,N,k,3)
            rel=pts_k-xyz_t.unsqueeze(2); w=torch.exp(-rel.pow(2).sum(-1)/(2*self.sigma**2))
            feat_t=feat.permute(0,2,1)
            neigh=feat_t.gather(1,idx.reshape(B,-1,1).expand(-1,-1,C)).reshape(B,N,k,C)
            k_sel=torch.arange(k,device=xyz.device).unsqueeze(0).unsqueeze(0).expand(B,N,k)
            kern=self.kernel[k_sel]
            out=(neigh.unsqueeze(-1)*kern).sum(-2)
            out=(out*w.unsqueeze(-1)).sum(2)
            return self.act(self.bn(out.permute(0,2,1)))

    class KPConvNet(nn.Module):
        def __init__(self,nc):
            super().__init__()
            self.enc0=KPConvLayer(9,32,k=16); self.enc1=KPConvLayer(32,64,k=16)
            self.enc2=KPConvLayer(64,128,k=16); self.enc3=KPConvLayer(128,256,k=8)
            self.dec3=nn.Sequential(nn.Conv1d(256+128,128,1,bias=False),nn.BatchNorm1d(128),nn.LeakyReLU(0.2))
            self.dec2=nn.Sequential(nn.Conv1d(128+64,64,1,bias=False),nn.BatchNorm1d(64),nn.LeakyReLU(0.2))
            self.dec1=nn.Sequential(nn.Conv1d(64+32,32,1,bias=False),nn.BatchNorm1d(32),nn.LeakyReLU(0.2))
            self.clf=nn.Sequential(nn.Conv1d(32,64,1,bias=False),nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2),nn.Dropout(0.5),nn.Conv1d(64,nc,1))
        def forward(self,x):
            xyz=x[:,:3]; e0=self.enc0(xyz,x); e1=self.enc1(xyz,e0)
            e2=self.enc2(xyz,e1); e3=self.enc3(xyz,e2)
            d3=self.dec3(torch.cat([e3,e2],1)); d2=self.dec2(torch.cat([d3,e1],1))
            d1=self.dec1(torch.cat([d2,e0],1)); return self.clf(d1)

    # ── Selección de modelo ───────────────────────────────────────────────────
    if arch_name == "PointNet++":
        model = PointNetPP(num_classes)
    elif arch_name == "KPConv":
        model = KPConvNet(num_classes)
    else:
        model = RandLA(num_classes)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    print(f"[Modelo] {arch_name} cargado en {device}")
    return model, num_classes, class_names


# ── Construcción de features ──────────────────────────────────────────────────

def build_features(xyz: np.ndarray, intensity: np.ndarray,
                   rgb: Optional[np.ndarray], cx: float, cy: float, cz: float) -> np.ndarray:
    """
    Construye el vector de 9 features por punto igual que el exporter:
    [x_local, y_local, z_local, intensity, r, g, b, agl_aprox, dist_2d]
    """
    xyz_local = (xyz - np.array([cx, cy, cz])).astype(np.float32)

    # Normalizar intensidad 0-1
    intens = np.clip(intensity.astype(np.float32), 0.0, 1.0)

    # RGB (o ceros si no existe)
    if rgb is not None:
        r = np.clip(rgb[:, 0].astype(np.float32), 0.0, 1.0)
        g = np.clip(rgb[:, 1].astype(np.float32), 0.0, 1.0)
        b = np.clip(rgb[:, 2].astype(np.float32), 0.0, 1.0)
    else:
        r = g = b = np.zeros(len(xyz), dtype=np.float32)

    # AGL aproximado (Z relativo al mínimo del parche)
    agl_approx = (xyz[:, 2] - xyz[:, 2].min()).astype(np.float32)

    # Distancia 2D al centro del parche
    dist_2d = np.sqrt(xyz_local[:, 0]**2 + xyz_local[:, 1]**2).astype(np.float32)

    return np.column_stack([
        xyz_local, intens, r, g, b, agl_approx, dist_2d
    ])  # (N, 9)


# ── Inferencia por parches ────────────────────────────────────────────────────

def infer_cloud(model, cloud: dict, num_classes: int,
                pts: int, overlap: float, batch_size: int,
                device: str) -> np.ndarray:
    """
    Inferencia sobre la nube completa usando parches solapados.

    Estrategia (similar a YOLO con imágenes grandes):
    - Dividir la nube en bloques espaciales de pts puntos
    - Cada punto puede caer en múltiples parches (overlap)
    - Predicción final: promedio de logits sobre todos los parches
    """
    import torch

    xyz       = cloud["xyz"].astype(np.float32)
    intensity = cloud["intensity"]
    rgb       = cloud.get("rgb")
    N         = len(xyz)

    print(f"[Inferencia] {N:,} puntos · parches de {pts} pts · overlap {overlap:.0%}")

    # Acumuladores: sum de logits y count de votos por punto
    logit_sum = np.zeros((N, num_classes), dtype=np.float32)
    vote_cnt  = np.zeros(N, dtype=np.int32)

    # ── Construir parches ─────────────────────────────────────────────────────
    # Usar KD-tree para parches espaciales centrados en puntos del grid
    step_pts = max(1, int(pts * (1.0 - overlap)))

    # Grid de centros basado en coordenadas XY
    x_min, x_max = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    y_min, y_max = float(xyz[:, 1].min()), float(xyz[:, 1].max())

    # Tamaño de celda en metros para obtener ~pts puntos por celda
    area     = max((x_max - x_min) * (y_max - y_min), 1.0)
    density  = N / area                                        # pts/m²
    cell_sz  = max(5.0, np.sqrt(pts / max(density, 0.001)))   # metros
    step_sz  = cell_sz * (1.0 - overlap)

    print(f"[Inferencia] Densidad {density:.1f} pts/m² · celda {cell_sz:.1f}m · "
          f"paso {step_sz:.1f}m")

    # Crear centros del grid
    xs = np.arange(x_min + cell_sz/2, x_max, step_sz)
    ys = np.arange(y_min + cell_sz/2, y_max, step_sz)
    if len(xs) == 0: xs = np.array([(x_min + x_max) / 2])
    if len(ys) == 0: ys = np.array([(y_min + y_max) / 2])

    centers = [(x, y) for x in xs for y in ys]
    print(f"[Inferencia] {len(centers)} parches a procesar")

    # Construir KD-tree para búsqueda eficiente
    try:
        from scipy.spatial import KDTree
        kdtree = KDTree(xyz[:, :2])
        use_kdtree = True
    except ImportError:
        use_kdtree = False
        print("[INFO] scipy no disponible — usando búsqueda directa (más lenta)")

    batch_feats  = []
    batch_idxs   = []
    patches_done = 0
    t_start      = time.time()

    def run_batch():
        if not batch_feats: return
        with torch.no_grad():
            X = torch.from_numpy(np.stack(batch_feats)).to(device)  # (B, N, 9)
            X = X.permute(0, 2, 1)                                   # (B, 9, N)
            logits = model(X).permute(0, 2, 1).cpu().numpy()         # (B, N, C)
        for i, patch_idx in enumerate(batch_idxs):
            logit_sum[patch_idx] += logits[i]
            vote_cnt[patch_idx]  += 1

    for ci, (cx, cy) in enumerate(centers):
        # Encontrar puntos en el radio de la celda
        if use_kdtree:
            candidate_ids = kdtree.query_ball_point([cx, cy], r=cell_sz)
        else:
            dx = xyz[:, 0] - cx; dy = xyz[:, 1] - cy
            candidate_ids = np.where(dx*dx + dy*dy <= cell_sz*cell_sz)[0].tolist()

        if len(candidate_ids) < 10:
            continue

        candidate_ids = np.array(candidate_ids, dtype=np.int64)

        # Submuestrear o repetir para llegar exactamente a pts puntos
        if len(candidate_ids) >= pts:
            # Tomar los pts más cercanos al centro
            dist_sq = ((xyz[candidate_ids, 0] - cx)**2 +
                       (xyz[candidate_ids, 1] - cy)**2)
            order = np.argsort(dist_sq)[:pts]
            patch_idx = candidate_ids[order]
        else:
            # Repetir puntos aleatorios para completar
            extra = pts - len(candidate_ids)
            repeats = np.random.choice(candidate_ids, extra, replace=True)
            patch_idx = np.concatenate([candidate_ids, repeats])

        # Construir features
        patch_xyz   = xyz[patch_idx]
        patch_intens = intensity[patch_idx]
        patch_rgb   = rgb[patch_idx] if rgb is not None else None
        cz          = float(patch_xyz[:, 2].mean())
        feats       = build_features(patch_xyz, patch_intens, patch_rgb, cx, cy, cz)

        batch_feats.append(feats.astype(np.float32))
        batch_idxs.append(patch_idx)

        if len(batch_feats) >= batch_size:
            run_batch()
            batch_feats.clear()
            batch_idxs.clear()

        patches_done += 1
        if patches_done % 50 == 0 or ci == len(centers) - 1:
            elapsed = time.time() - t_start
            pct = (ci + 1) / len(centers) * 100
            eta = elapsed / max(ci + 1, 1) * (len(centers) - ci - 1)
            covered = int((vote_cnt > 0).sum())
            print(f"  [{pct:5.1f}%] parches {patches_done}/{len(centers)} | "
                  f"pts cubiertos {covered:,}/{N:,} | "
                  f"ETA {eta:.0f}s")

    # Último batch
    run_batch()

    # ── Puntos sin predicción → buscar vecino más cercano ─────────────────────
    uncovered = np.where(vote_cnt == 0)[0]
    if len(uncovered) > 0:
        print(f"[Inferencia] {len(uncovered):,} puntos sin cobertura → "
              f"propagando desde vecinos")
        covered_mask = vote_cnt > 0
        covered_ids  = np.where(covered_mask)[0]
        if len(covered_ids) > 0:
            if use_kdtree:
                tree = KDTree(xyz[covered_ids, :2])
                _, nn_idx = tree.query(xyz[uncovered, :2], k=1)
                logit_sum[uncovered] = logit_sum[covered_ids[nn_idx]]
                vote_cnt[uncovered]  = 1
            else:
                # Propagación simple: clase mayoritaria de los cubiertos
                maj_class = np.argmax(logit_sum[covered_ids].sum(0))
                logit_sum[uncovered, maj_class] = 1.0
                vote_cnt[uncovered] = 1

    # ── Predicción final: argmax de logits promediados ────────────────────────
    avg_logits  = logit_sum / np.maximum(vote_cnt[:, None], 1)
    predictions = avg_logits.argmax(axis=1).astype(np.uint8)

    covered_pct = float((vote_cnt > 0).mean()) * 100
    print(f"[Inferencia] Cobertura: {covered_pct:.1f}%")

    return predictions


# ── Guardar resultado ─────────────────────────────────────────────────────────

def save_result(input_cloud: dict, predictions: np.ndarray,
                class_names: list, output_path: str):
    """
    Guarda la nube clasificada como .las con:
    - Campo 'classification' con los índices de clase predichos
    - Colores RGB por clase para visualización directa en GeoAnnotate3D / CloudCompare
    """
    try:
        import laspy
    except ImportError:
        sys.exit("laspy no instalado: pip install laspy[lazrs]")

    # ── Paleta de colores (misma que GeoAnnotate3D default schema) ────────────
    # Las clases deben coincidir con las entrenadas
    PALETTE = [
        (100, 100, 100),  # 0 - sin clasificar (gris)
        (139,  90,  43),  # 1 - suelo (marrón)
        ( 34, 139,  34),  # 2 - vegetación baja (verde)
        (  0, 100,   0),  # 3 - árbol (verde oscuro)
        ( 70,  70, 180),  # 4 - edificio (azul grisáceo)
        (204, 102,   0),  # 5 - vehículo (naranja)
        (180, 180, 180),  # 6 - techo (gris claro)
        (255, 200,   0),  # 7 - cable/tendido (amarillo)
        (100,  50, 200),  # 8 - poste (violeta)
        (  0, 200, 200),  # 9 - agua (cian)
        (255, 128, 128),  # 10 - fachada (salmón)
        ( 50, 200, 100),  # 11 - hierba (verde lima)
        (200, 100, 200),  # 12 - acera (malva)
        (128, 128,   0),  # 13 - carretera (oliva)
    ]

    # Extender paleta si hay más clases
    while len(PALETTE) <= predictions.max():
        import random; random.seed(len(PALETTE))
        PALETTE.append((random.randint(50,200), random.randint(50,200), random.randint(50,200)))

    # ── Crear LAS ─────────────────────────────────────────────────────────────
    original_las = input_cloud.get("las")
    xyz          = input_cloud["xyz"]

    if original_las is not None:
        # Copiar header del LAS original
        hdr = laspy.LasHeader(
            point_format=original_las.point_format.id,
            version=original_las.header.version)
        hdr.scales  = original_las.header.scales
        hdr.offsets = original_las.header.offsets
        out_las = laspy.LasData(header=hdr)

        # Copiar campos originales
        for dim in original_las.point_format.dimension_names:
            try:
                setattr(out_las, dim, getattr(original_las, dim))
            except Exception:
                pass
    else:
        # Crear nuevo LAS desde cero
        hdr = laspy.LasHeader(point_format=2, version="1.4")
        hdr.scales  = np.array([0.001, 0.001, 0.001])
        hdr.offsets = xyz.mean(axis=0)
        out_las = laspy.LasData(header=hdr)
        out_las.x = xyz[:, 0]
        out_las.y = xyz[:, 1]
        out_las.z = xyz[:, 2]

    # Asignar clasificación
    out_las.classification = predictions.astype(np.uint8)

    # Asignar colores por clase para visualización directa
    has_rgb = hasattr(out_las, 'red') or 2 in original_las.point_format.id \
              if original_las else False
    try:
        reds   = np.array([PALETTE[min(c, len(PALETTE)-1)][0] * 256 for c in predictions], dtype=np.uint16)
        greens = np.array([PALETTE[min(c, len(PALETTE)-1)][1] * 256 for c in predictions], dtype=np.uint16)
        blues  = np.array([PALETTE[min(c, len(PALETTE)-1)][2] * 256 for c in predictions], dtype=np.uint16)
        out_las.red   = reds
        out_las.green = greens
        out_las.blue  = blues
    except Exception as e:
        print(f"[Aviso] No se pudieron escribir colores RGB: {e}")

    out_las.write(output_path)

    # ── Estadísticas ──────────────────────────────────────────────────────────
    print(f"\n[Resultado] Guardado en: {output_path}")
    print(f"{'Clase':<30} {'Pts':>10} {'%':>7}")
    print("─" * 50)
    N = len(predictions)
    for c in range(predictions.max() + 1):
        n = int((predictions == c).sum())
        if n == 0: continue
        name = class_names[c] if c < len(class_names) else f"clase_{c}"
        label = "sin clasificar" if c == 0 else name
        print(f"  {label:<28} {n:>10,} {100*n/N:>6.1f}%")
    print("─" * 50)
    print(f"  {'TOTAL':<28} {N:>10,}  100.0%\n")

    # ── Exportar también un .txt de clases para referencia ────────────────────
    txt_path = Path(output_path).with_suffix(".classes.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# Clases predichas por GeoAnnotate3D\n")
        f.write(f"# Modelo: {output_path}\n\n")
        f.write(f"{'ID':<5} {'Nombre':<30} {'Puntos':>12} {'%':>8}\n")
        f.write("-" * 58 + "\n")
        for c in range(predictions.max() + 1):
            n = int((predictions == c).sum())
            name = class_names[c] if c < len(class_names) else f"clase_{c}"
            if c == 0: name = "sin clasificar"
            r,g,b = PALETTE[min(c, len(PALETTE)-1)]
            f.write(f"{c:<5} {name:<30} {n:>12,} {100*n/N:>7.1f}%  RGB({r},{g},{b})\n")
    print(f"[Resultado] Tabla de clases: {txt_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Dispositivo
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            sys.exit("PyTorch no instalado: pip install torch")
    else:
        device = args.device
    print(f"[Config] Dispositivo: {device}")

    # Output path
    if args.output is None:
        stem = Path(args.input).stem
        args.output = str(Path(args.input).parent / f"{stem}_classified.las")

    t0 = time.time()

    # 1. Cargar nube
    cloud = load_cloud(args.input)
    print(f"[Carga] {len(cloud['xyz']):,} puntos cargados en {time.time()-t0:.1f}s")

    # 2. Cargar modelo
    import torch
    model, num_classes, class_names = load_model(args.model, device)

    # 3. Inferencia por parches
    t1 = time.time()
    predictions = infer_cloud(
        model, cloud,
        num_classes=num_classes,
        pts=args.pts,
        overlap=args.overlap,
        batch_size=args.batch,
        device=device)
    print(f"[Inferencia] Completada en {time.time()-t1:.1f}s")

    # 4. Guardar
    save_result(cloud, predictions, class_names, args.output)
    print(f"[Total] {time.time()-t0:.1f}s")
    print(f"\n Para visualizar en GeoAnnotate3D: Archivo → Abrir → {args.output}")


if __name__ == "__main__":
    main()
