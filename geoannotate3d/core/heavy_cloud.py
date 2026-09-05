"""
core/heavy_cloud.py — Formato GA3D-Bin v2.0 (estable)

Formato simple: header 512 bytes + xyz float32 plano + attrs.
Sin compresion, sin cuantizacion. Lectura directa con mmap.
Soporta LAS, LAZ y E57.
"""
from __future__ import annotations
import hashlib, os, struct
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

HEAVY_THRESHOLD = 200_000_000
HDR_SIZE  = 512
HDR_MAGIC = b"GA3D"
HDR_VER   = 2
HDR_FMT   = "<4sBQBBB3d3f3f256s"
HDR_PACK  = struct.calcsize(HDR_FMT)

# ─── rutas ────────────────────────────────────────────────────────────────────

def ga3d_bin_path(source_path: str) -> Path:
    p = Path(source_path)
    return p.parent / (p.stem + ".ga3d_bin")

def is_converted(source_path: str) -> bool:
    p = ga3d_bin_path(source_path)
    return p.exists() and p.stat().st_size > HDR_SIZE

# ─── header ───────────────────────────────────────────────────────────────────

def _write_hdr(f, n, has_int, has_rgb, has_cls, offset, bmin, bmax, crs):
    crs_b = crs.encode("utf-8")[:255].ljust(256, b"\x00")
    packed = struct.pack(HDR_FMT, HDR_MAGIC, HDR_VER, int(n),
        int(has_int), int(has_rgb), int(has_cls),
        float(offset[0]), float(offset[1]), float(offset[2]),
        float(bmin[0]), float(bmin[1]), float(bmin[2]),
        float(bmax[0]), float(bmax[1]), float(bmax[2]), crs_b)
    f.seek(0)
    f.write(packed + b"\x00" * (HDR_SIZE - len(packed)))

def _read_hdr(path: Path) -> dict:
    raw = open(path, "rb").read(HDR_SIZE)
    magic, ver, n, hi, hr, hc, ox, oy, oz, x0, y0, z0, x1, y1, z1, crs_b = \
        struct.unpack_from(HDR_FMT, raw)
    if magic != HDR_MAGIC:
        raise ValueError(
            f"Archivo no reconocido como GA3D-Bin. "
            "Si es de una version anterior, elimínalo y reconvierte.")
    n = int(n)
    if not (0 < n < 10_000_000_000):
        raise ValueError(
            "GA3D-Bin incompatible con esta version.\n"
            "Elimina el archivo .ga3d_bin y vuelve a abrir el .las para reconvertir.")
    return {"n": n, "has_int": bool(hi), "has_rgb": bool(hr), "has_cls": bool(hc),
            "offset": np.array([ox, oy, oz], np.float64),
            "bmin":   np.array([x0, y0, z0], np.float32),
            "bmax":   np.array([x1, y1, z1], np.float32),
            "crs": crs_b.rstrip(b"\x00").decode("utf-8", "ignore")}

# ─── cargador ─────────────────────────────────────────────────────────────────

def load_ga3d_bin(path: str):
    """Carga .ga3d_bin como PointCloud con mmap. Identico a cargar un LAS."""
    from core.pointcloud import PointCloud
    p = Path(path)
    # Validar antes de cualquier memmap
    file_size = p.stat().st_size
    if file_size <= HDR_SIZE:
        raise ValueError("GA3D-Bin vacio o corrupto.")
    hdr = _read_hdr(p)
    n   = hdr["n"]
    # Verificar que el archivo tiene el tamano esperado
    min_size = HDR_SIZE + n * 12
    if file_size < min_size - 1024:
        raise ValueError(
            f"GA3D-Bin incompleto ({file_size/1e9:.2f}GB, necesita ≥{min_size/1e9:.2f}GB).\n"
            "Elimina el archivo y reconvierte.")
    pc = PointCloud()
    pc.fmt = "GA3D-Bin"; pc.offset = hdr["offset"]; pc.crs = hdr["crs"]
    pc.is_mmap = True; pc._cache_path = p
    off = HDR_SIZE
    pc.xyz = np.memmap(str(p), dtype=np.float32, mode="r", offset=off, shape=(n, 3)); off += n * 12
    if hdr["has_int"]:
        pc.intensity = np.memmap(str(p), dtype=np.float32, mode="r", offset=off, shape=(n,)); off += n * 4
    if hdr["has_rgb"]:
        pc.rgb = np.memmap(str(p), dtype=np.uint8, mode="r", offset=off, shape=(n, 3)); off += n * 3
    if hdr["has_cls"]:
        pc.classification = np.memmap(str(p), dtype=np.uint8, mode="r", offset=off, shape=(n,))
    pc.finalize()
    return pc

# ─── conversor ────────────────────────────────────────────────────────────────

class HeavyCloudConverter(QThread):
    """
    Convierte LAS/LAZ/E57 a .ga3d_bin v2.
    Lee en chunks de 2M pts — RAM usada: ~100MB por chunk.
    """
    progress = pyqtSignal(int, str)
    done     = pyqtSignal(str)
    error    = pyqtSignal(str)
    CHUNK = 2_000_000

    def __init__(self, source_path: str, parent=None):
        super().__init__(parent)
        self.source_path = source_path

    def run(self):
        try:
            self.done.emit(str(self._convert()))
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _convert(self) -> Path:
        ext = Path(self.source_path).suffix.lower()
        if ext in (".las", ".laz"):
            return self._convert_las()
        elif ext == ".e57":
            return self._convert_e57()
        else:
            raise ValueError(f"Formato no soportado para conversion: {ext}")

    # ── LAS / LAZ ─────────────────────────────────────────────────────────────

    def _convert_las(self) -> Path:
        import laspy
        src   = self.source_path
        out   = ga3d_bin_path(src)
        chunk = self.CHUNK

        self.progress.emit(1, "Leyendo cabecera LAS…")
        with laspy.open(src) as rdr:
            h = rdr.header
            n = h.point_count
            crs = ""
            try: crs = str(getattr(h.parse_crs(), "name", ""))
            except: pass
            try:
                mn = np.array([h.mins[0], h.mins[1], h.mins[2]], np.float64)
                mx = np.array([h.maxs[0], h.maxs[1], h.maxs[2]], np.float64)
                if np.any(mn >= mx): raise ValueError
                offset = (mn + mx) * 0.5
            except:
                offset = self._scan_offset_las(laspy, src, n)

        self.progress.emit(3, f"{n/1e6:.1f}M pts")

        # Detectar atributos
        has_int = has_rgb = has_cls = False
        with laspy.open(src) as rdr:
            for ck in rdr.chunk_iterator(100):
                try: _ = ck.intensity;      has_int = True
                except: pass
                try: _ = ck.red;            has_rgb = True
                except: pass
                try: _ = ck.classification; has_cls = True
                except: pass
                break

        # Pre-alocar con mmap (no RAM para attrs)
        total = (HDR_SIZE + n*12
                 + (n*4 if has_int else 0)
                 + (n*3 if has_rgb else 0)
                 + (n   if has_cls else 0))
        self.progress.emit(5, "Creando archivo…")
        with open(out, "wb") as f:
            _write_hdr(f, n, has_int, has_rgb, has_cls,
                       offset, np.zeros(3), np.zeros(3), crs)
            f.seek(total - 1); f.write(b"\x00")

        # Mmaps de escritura
        xyz_mm = np.memmap(str(out), dtype=np.float32, mode="r+",
                           offset=HDR_SIZE, shape=(n, 3))
        oa = HDR_SIZE + n * 12
        int_mm = np.memmap(str(out), dtype=np.float32, mode="r+",
                           offset=oa, shape=(n,)) if has_int else None
        oa += n * 4 if has_int else 0
        rgb_mm = np.memmap(str(out), dtype=np.uint8, mode="r+",
                           offset=oa, shape=(n, 3)) if has_rgb else None
        oa += n * 3 if has_rgb else 0
        cls_mm = np.memmap(str(out), dtype=np.uint8, mode="r+",
                           offset=oa, shape=(n,)) if has_cls else None

        # Leer LAS en chunks y escribir directamente en mmaps
        pos = 0
        xmn=ymn=zmn=1e18; xmx=ymx=zmx=-1e18; rgb16 = None

        self.progress.emit(8, "Convirtiendo…")
        with laspy.open(src) as rdr:
            for ck in rdr.chunk_iterator(chunk):
                end = pos + len(ck.x)
                x = np.asarray(ck.x, np.float64) - offset[0]
                y = np.asarray(ck.y, np.float64) - offset[1]
                z = np.asarray(ck.z, np.float64) - offset[2]
                xmn=min(xmn,float(x.min())); xmx=max(xmx,float(x.max()))
                ymn=min(ymn,float(y.min())); ymx=max(ymx,float(y.max()))
                zmn=min(zmn,float(z.min())); zmx=max(zmx,float(z.max()))
                xyz_mm[pos:end, 0] = x.astype(np.float32)
                xyz_mm[pos:end, 1] = y.astype(np.float32)
                xyz_mm[pos:end, 2] = z.astype(np.float32)
                if has_int and int_mm is not None:
                    try: int_mm[pos:end] = np.asarray(ck.intensity, np.float32)
                    except: pass
                if has_rgb and rgb_mm is not None:
                    try:
                        r=np.asarray(ck.red,np.float32); g=np.asarray(ck.green,np.float32)
                        b=np.asarray(ck.blue,np.float32)
                        # Detectar 16-bit en CADA chunk (no solo el primero)
                        if max(float(r.max()),float(g.max()),float(b.max())) > 255:
                            rgb16 = True
                        if rgb16 is None: rgb16 = False
                        sc = 255.0 / (65535.0 if rgb16 else 255.0)
                        rgb_mm[pos:end,0] = np.clip(r*sc,0,255).astype(np.uint8)
                        rgb_mm[pos:end,1] = np.clip(g*sc,0,255).astype(np.uint8)
                        rgb_mm[pos:end,2] = np.clip(b*sc,0,255).astype(np.uint8)
                    except Exception as _rgb_e:
                        # Si falla la escritura RGB, deshabilitar para no tener datos corruptos
                        print(f"[heavy_cloud] RGB write error en chunk pos={pos}: {_rgb_e}")
                        has_rgb = False; rgb_mm = None
                if has_cls and cls_mm is not None:
                    try: cls_mm[pos:end] = np.asarray(ck.classification, np.uint8)
                    except: pass
                pos = end
                if pos % (chunk * 5) < chunk:
                    self.progress.emit(8 + int(85 * pos / max(n,1)),
                        f"Convirtiendo {pos/1e6:.0f}M/{n/1e6:.0f}M pts…")

        # Normalizar intensidad
        if has_int and int_mm is not None:
            mx = float(int_mm.max())
            if mx > 0: int_mm /= mx
            int_mm.flush()
        if rgb_mm is not None: rgb_mm.flush()
        if cls_mm is not None: cls_mm.flush()
        xyz_mm.flush(); del xyz_mm, int_mm, rgb_mm, cls_mm

        # Reescribir header con bounds reales
        bmin = np.array([xmn,ymn,zmn], np.float32)
        bmax = np.array([xmx,ymx,zmx], np.float32)
        # has_rgb puede haberse desactivado durante la conversión si hubo errores
        with open(out, "r+b") as f:
            _write_hdr(f, n, has_int, has_rgb, has_cls, offset, bmin, bmax, crs)
        rgb_flag = "SÍ" if has_rgb else "NO (falló durante conversión)"
        print(f"[heavy_cloud] RGB en GA3D-Bin: {rgb_flag}")

        sz = out.stat().st_size
        self.progress.emit(100, f"Listo — {n/1e6:.1f}M pts · {sz/1e9:.1f}GB")
        return out

    # ── E57 ───────────────────────────────────────────────────────────────────

    def _convert_e57(self) -> Path:
        """
        Convierte E57 a GA3D-Bin escribiendo scan por scan directamente al mmap.
        Sin concatenar todo en RAM — funciona con nubes de 2600M+ puntos.
        RAM usada: solo un scan a la vez (pye57 no tiene chunk iterator).
        """
        try:
            import pye57
        except ImportError:
            raise ImportError("Para E57 instala: pip install pye57")

        src = self.source_path
        out = ga3d_bin_path(src)
        self.progress.emit(1, "Leyendo E57…")
        e57 = pye57.E57(src)
        n_scans = e57.scan_count
        n = sum(int(e57.get_header(i).point_count) for i in range(n_scans))
        crs = ""
        self.progress.emit(3, f"{n/1e6:.1f}M pts · {n_scans} scan(s)")

        # ── Pasada 1: calcular offset (bounds globales) con muestra pequeña ──
        self.progress.emit(5, "Calculando bounds…")
        xmn=ymn=zmn= 1e18; xmx=ymx=zmx=-1e18
        has_rgb = False
        for i in range(n_scans):
            # colors=True e intensity=True necesarios explícitamente en pye57
            try:
                data = e57.read_scan(i, ignore_missing_fields=True,
                                     colors=True, intensity=True)
            except Exception:
                data = e57.read_scan(i, ignore_missing_fields=True)
            x = data.get("cartesianX", np.zeros(1))
            y = data.get("cartesianY", np.zeros(1))
            z = data.get("cartesianZ", np.zeros(1))
            step = max(1, len(x) // 50_000)
            xmn=min(xmn,float(x[::step].min())); xmx=max(xmx,float(x[::step].max()))
            ymn=min(ymn,float(y[::step].min())); ymx=max(ymx,float(y[::step].max()))
            zmn=min(zmn,float(z[::step].min())); zmx=max(zmx,float(z[::step].max()))
            # Detectar RGB: pye57 usa 'colorRed' o a veces 'r'/'R'
            for _rkey in ("colorRed", "r", "R", "red"):
                if _rkey in data:
                    has_rgb = True
                    break
            del data, x, y, z
            self.progress.emit(5 + int(10 * i / max(n_scans,1)),
                               f"Bounds scan {i+1}/{n_scans}…")

        offset = np.array([(xmn+xmx)*0.5, (ymn+ymx)*0.5, (zmn+zmx)*0.5], np.float64)

        # ── Pre-alocar archivo con mmap ────────────────────────────────────
        total = HDR_SIZE + n*12 + (n*3 if has_rgb else 0)
        self.progress.emit(16, "Creando archivo binario…")
        with open(out, "wb") as f:
            _write_hdr(f, n, False, has_rgb, False,
                       offset, np.zeros(3,np.float32), np.zeros(3,np.float32), crs)
            f.seek(total - 1); f.write(b"\x00")

        xyz_mm = np.memmap(str(out), dtype=np.float32, mode="r+",
                           offset=HDR_SIZE, shape=(n, 3))
        rgb_mm = (np.memmap(str(out), dtype=np.uint8, mode="r+",
                            offset=HDR_SIZE + n*12, shape=(n, 3))
                  if has_rgb else None)

        # ── Pasada 2: leer cada scan y escribir al mmap ────────────────────
        pos = 0
        rx_mn=ry_mn=rz_mn= 1e18; rx_mx=ry_mx=rz_mx=-1e18
        rgb16 = None

        for i in range(n_scans):
            pct = 17 + int(78 * i / max(n_scans, 1))
            self.progress.emit(pct, f"Convirtiendo scan {i+1}/{n_scans}…")
            # colors=True necesario en pye57 para incluir datos de color
            try:
                data = e57.read_scan(i, ignore_missing_fields=True,
                                     colors=True, intensity=True)
            except Exception:
                data = e57.read_scan(i, ignore_missing_fields=True)
            x = np.asarray(data.get("cartesianX", np.zeros(1)), np.float64) - offset[0]
            y = np.asarray(data.get("cartesianY", np.zeros(1)), np.float64) - offset[1]
            z = np.asarray(data.get("cartesianZ", np.zeros(1)), np.float64) - offset[2]
            cnt = len(x); end = pos + cnt

            xyz_mm[pos:end, 0] = x.astype(np.float32)
            xyz_mm[pos:end, 1] = y.astype(np.float32)
            xyz_mm[pos:end, 2] = z.astype(np.float32)

            rx_mn=min(rx_mn,float(x.min())); rx_mx=max(rx_mx,float(x.max()))
            ry_mn=min(ry_mn,float(y.min())); ry_mx=max(ry_mx,float(y.max()))
            rz_mn=min(rz_mn,float(z.min())); rz_mx=max(rz_mx,float(z.max()))
            del x, y, z

            if has_rgb and rgb_mm is not None:
                # Soportar múltiples nombres de campo según versión de pye57
                rk = next((k for k in ("colorRed","r","R","red") if k in data), None)
                gk = next((k for k in ("colorGreen","g","G","green") if k in data), None)
                bk = next((k for k in ("colorBlue","b","B","blue") if k in data), None)
                if rk and gk and bk:
                    r = np.asarray(data[rk], np.float32)
                    g = np.asarray(data[gk], np.float32)
                    b = np.asarray(data[bk], np.float32)
                    # Detectar 16-bit en cada chunk
                    if max(float(r.max()),float(g.max()),float(b.max())) > 255:
                        rgb16 = True
                    if rgb16 is None: rgb16 = False
                    sc = 255.0 / (65535.0 if rgb16 else 255.0)
                    rgb_mm[pos:end, 0] = np.clip(r*sc, 0, 255).astype(np.uint8)
                    rgb_mm[pos:end, 1] = np.clip(g*sc, 0, 255).astype(np.uint8)
                    rgb_mm[pos:end, 2] = np.clip(b*sc, 0, 255).astype(np.uint8)
                    del r, g, b
                else:
                    print(f"[E57] scan {i}: sin datos RGB (keys: {list(data.keys())[:8]})")
            del data
            pos = end

        xyz_mm.flush(); del xyz_mm
        if rgb_mm is not None: rgb_mm.flush(); del rgb_mm

        # Reescribir header con bounds reales
        bmin = np.array([rx_mn, ry_mn, rz_mn], np.float32)
        bmax = np.array([rx_mx, ry_mx, rz_mx], np.float32)
        with open(out, "r+b") as f:
            _write_hdr(f, n, False, has_rgb, False, offset, bmin, bmax, crs)

        sz = out.stat().st_size
        self.progress.emit(100,
            f"E57 listo — {n/1e6:.1f}M pts · {sz/1e9:.1f}GB")
        return out

    def _scan_offset_las(self, laspy, src, n):
        xs=[]; ys=[]; zs=[]; step=max(1, n//500_000)
        with laspy.open(src) as rdr:
            for ck in rdr.chunk_iterator(2_000_000):
                xs.append(np.asarray(ck.x, np.float64)[::step])
                ys.append(np.asarray(ck.y, np.float64)[::step])
                zs.append(np.asarray(ck.z, np.float64)[::step])
                if sum(len(a) for a in xs) > 500_000: break
        x=np.concatenate(xs); y=np.concatenate(ys); z=np.concatenate(zs)
        return np.array([(x.min()+x.max())*0.5,
                         (y.min()+y.max())*0.5,
                         (z.min()+z.max())*0.5], np.float64)
