"""
core/pointcloud.py — PointCloud + CloudPipeline v2.0 (Out-of-Core)

ESTRATEGIA POTREE/LP360:
  Para nubes > RAM_THRESHOLD (8GB estimado), usamos:
  1. MMAP CACHE: xyz se escribe a disco como .ga3d_cache binary, luego se
     accede via np.memmap. El OS maneja el paginado — solo las páginas
     accedidas viven en RAM.
  2. LOD CACHE: los índices LOD se persisten en el cache. Segunda apertura
     = instantánea (no rebuild).
  3. STREAMING: solo se lee lo que los índices LOD señalan.

Para nubes normales (<500M pts), carga en RAM como antes.
"""
from __future__ import annotations
import io, time, os, struct
from pathlib import Path
from typing import Optional, Dict
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# Threshold en puntos para activar out-of-core (>500M o si es mayor que RAM disponible)
OUT_OF_CORE_THRESHOLD = 500_000_000

def _available_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 8.0  # default fallback

class PointCloud:
    def __init__(self):
        self.xyz: Optional[np.ndarray]            = None  # may be np.memmap
        self.n_points  = 0
        self.intensity: Optional[np.ndarray]      = None  # may be np.memmap
        self.rgb: Optional[np.ndarray]            = None  # may be np.memmap
        self.classification: Optional[np.ndarray] = None  # may be np.memmap
        self.return_num: Optional[np.ndarray]     = None
        # Confianza del modelo por punto (0-1), solo presente tras correr
        # inferencia con return_confidence=True (ver infer.py::infer_cloud).
        # No viene de ningún archivo — es un resultado derivado, como las
        # predicciones mismas.
        self.confidence: Optional[np.ndarray]     = None
        self.offset = np.zeros(3, np.float64)
        self.crs    = ""
        self.bounds: Optional[np.ndarray] = None
        self.center: Optional[np.ndarray] = None
        self.scale   = 1.0
        self.filename = ""; self.fmt = ""; self.load_time = 0.0
        self.octree   = None
        self.is_mmap  = False
        self.is_heavy = False
        self._cache_path: Optional[Path] = None

    def finalize(self):
        if self.xyz is None or len(self.xyz) == 0: return
        self.n_points = len(self.xyz)
        # Diagnóstico RGB en consola
        if self.rgb is not None:
            print(f"[PointCloud] RGB: SÍ ({self.n_points/1e6:.1f}M pts con color)")
        else:
            print(f"[PointCloud] RGB: NO (la nube no tiene datos de color)")
        # Use sample for bounds on huge mmaps (faster than full min/max)
        if len(self.xyz) > 5_000_000:
            step = max(1, len(self.xyz) // 1_000_000)
            sample = self.xyz[::step]
            self.bounds = np.array([sample.min(0), sample.max(0)], np.float32)
        else:
            self.bounds = np.array([self.xyz.min(0), self.xyz.max(0)], np.float32)
        self.center = self.bounds.mean(0)
        self.scale  = float((self.bounds[1]-self.bounds[0]).max()) or 1.0

    def get_attrs(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
        """Return attributes for given indices. mmap arrays auto-page."""
        d = {}
        if self.intensity      is not None: d["intensity"]      = self.intensity[idx]
        if self.rgb            is not None: d["rgb"]            = self.rgb[idx]
        if self.classification is not None: d["classification"] = self.classification[idx]
        if self.return_num     is not None: d["return_num"]     = self.return_num[idx]
        if self.confidence     is not None: d["confidence"]     = self.confidence[idx]
        return d

    @property
    def has_rgb(self): return self.rgb is not None
    @property
    def has_intensity(self): return self.intensity is not None
    @property
    def memory_mb(self):
        total = 0
        for a in [self.xyz, self.intensity, self.rgb, self.classification, self.return_num]:
            if a is not None and not isinstance(a, np.memmap):
                total += a.nbytes
        return total / 1e6

    def close(self):
        for attr in ['xyz', 'intensity', 'rgb', 'classification']:
            a = getattr(self, attr, None)
            if isinstance(a, np.memmap):
                try: del a
                except: pass


# ── RAM management ────────────────────────────────────────────────────────────

def _available_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 8.0

def _total_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return 16.0

def _should_mmap(n_pts: int, bytes_per_pt: int = 12) -> bool:
    """Return True if this array should be memory-mapped instead of loaded in RAM."""
    needed_gb = n_pts * bytes_per_pt / 1e9
    available = _available_ram_gb()
    # mmap if array would use >30% of available RAM
    return needed_gb > available * 0.30

def _attr_mmap_path(source_path: str, n: int, attr: str) -> Path:
    import hashlib
    key = f"{source_path}_{n}_{attr}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return _cache_dir() / f"{attr}_{h}_{n}.bin"

def _make_attr_mmap(source_path: str, n: int, attr: str, dtype: np.dtype,
                     shape: tuple) -> Optional[np.ndarray]:
    """Create or load a memory-mapped attribute array."""
    p = _attr_mmap_path(source_path, n, attr)
    if p.exists() and p.stat().st_size == np.prod(shape) * np.dtype(dtype).itemsize:
        return np.memmap(str(p), dtype=dtype, mode='r', shape=shape)
    try:
        return np.memmap(str(p), dtype=dtype, mode='w+', shape=shape)
    except Exception:
        return None


# ── Cache paths ───────────────────────────────────────────────────────────────

def _cache_dir() -> Path:
    """Returns system cache directory for mmap files."""
    d = Path(os.environ.get("GEOANNOTATE_CACHE", Path.home() / ".cache" / "geoannotate3d"))
    d.mkdir(parents=True, exist_ok=True)
    return d

def _xyz_cache_path(source_path: str, n: int, offset: np.ndarray) -> Path:
    """Unique cache file name based on source file + size + offset."""
    import hashlib
    key = f"{source_path}_{n}_{offset[0]:.3f}_{offset[1]:.3f}_{offset[2]:.3f}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return _cache_dir() / f"xyz_{h}_{n}.f32"


# ── CloudPipeline ─────────────────────────────────────────────────────────────

class CloudPipeline(QThread):
    progress     = pyqtSignal(int, str)
    cloud_ready  = pyqtSignal(object)
    octree_ready = pyqtSignal(object, object)
    heavy_needed = pyqtSignal(object)
    error        = pyqtSignal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent); self.path = path

    def run(self):
        try:
            t0 = time.perf_counter()
            self.progress.emit(2, "Leyendo archivo…")
            pc = self._load()
            pc.filename  = Path(self.path).name
            pc.load_time = time.perf_counter() - t0
            pc.finalize()

            # Heavy cloud: detener antes del octree build
            if getattr(pc, 'is_heavy', False):
                self.progress.emit(100, "Listo ✓")
                self.heavy_needed.emit(pc)   # main_window gestiona carga y display
                return

            from core.octree import Octree
            from render.colors import set_z_range_cache
            octree = Octree()
            self._l1_emitted = False

            def cb(pct, msg):
                self.progress.emit(40 + int(pct*0.55), msg)
                if pct >= 30 and not self._l1_emitted:
                    pc.octree = octree
                    self.cloud_ready.emit(pc)
                    self._l1_emitted = True

            self.progress.emit(40, "Construyendo LOD (C)…")
            octree.build(pc.xyz, cb)
            pc.octree = octree
            if not self._l1_emitted:
                self.cloud_ready.emit(pc)

            try:
                z = pc.xyz[:,2]
                samp = z[::max(1, len(z)//200_000)]
                set_z_range_cache(id(pc.xyz),
                                   float(np.percentile(samp,1)),
                                   float(np.percentile(samp,99)))
            except Exception: pass

            self.progress.emit(100, "Listo ✓")
            self.octree_ready.emit(octree, pc)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")

    def _load(self) -> PointCloud:
        ext = Path(self.path).suffix.lower()
        loaders = {
            ".las": self._las, ".laz": self._las,
            ".ply": self._ply, ".pcd": self._pcd,
            ".xyz": self._xyz_txt, ".txt": self._xyz_txt,
            ".csv": self._xyz_txt, ".asc": self._xyz_txt,
            ".pts": self._xyz_txt, ".npy": self._npy,
            ".e57": self._e57,
            ".ga3d_bin": self._ga3d_bin,
            ".ga3d_bin": self._ga3d_bin,
        }
        fn = loaders.get(ext)
        if fn is None: raise ValueError(f"Formato no soportado: {ext!r}")
        return fn()

    # ------------------------------------------------------------------
    # LAS/LAZ — streaming loader + mmap cache
    # ------------------------------------------------------------------

    def _read_laz_file(self, laspy, path):
        """
        Intenta abrir un .laz probando, en orden, varias combinaciones de
        backend de descompresión (lazrs/laszip) y `read_evlrs` (True/False).

        Antes: si el primer intento (`laspy.read(path)` con backend por
        defecto) fallaba con un mensaje que NO contenía "backend" ni "laz"
        —como "read length must be non-negative or -1", que en realidad
        indica un EVLR con longitud corrupta al FINAL del archivo, nada
        que ver con qué backend de compresión está instalado— el código
        re-lanzaba inmediatamente sin probar nada más, y el mensaje final
        seguía diciendo "pip install laspy[lazrs]" aunque instalar lazrs
        no soluciona un EVLR corrupto. Ahora se prueban TODAS las
        combinaciones razonables antes de rendirse, y el mensaje final
        distingue el caso de EVLR/encabezado corrupto del de backend
        faltante.

        Bug adicional encontrado y corregido: el código original buscaba
        `laspy.LazBackend.lazrs` / `.laszip` (minúsculas) — los miembros
        reales del enum son `Lazrs`/`Laszip`/`LazrsParallel` (con
        mayúscula inicial). `hasattr(..., "lazrs")` era siempre False, así
        que la ruta de "probar un backend explícito" nunca se ejecutaba
        realmente, ni en este código ni en el original.
        """
        attempts = []   # (kwargs, descripción) — para logging del último error
        if hasattr(laspy, "LazBackend"):
            backends = [getattr(laspy.LazBackend, n) for n in
                        ("LazrsParallel", "Lazrs", "Laszip")
                        if hasattr(laspy.LazBackend, n)]
        else:
            backends = [None]
        for backend in (backends or [None]):
            for read_evlrs in (True, False):
                kwargs = {}
                if backend is not None: kwargs["laz_backend"] = backend
                kwargs["read_evlrs"] = read_evlrs
                attempts.append(kwargs)

        last_err = None
        for kwargs in attempts:
            try:
                with laspy.open(path, **kwargs) as reader:
                    return reader.read()
            except Exception as e:
                last_err = e
                continue

        msg = str(last_err).lower() if last_err else ""
        if any(k in msg for k in ("evlr", "length", "header")):
            raise RuntimeError(
                "El encabezado del archivo .laz parece estar dañado o "
                "usar una variante no estándar (EVLR con longitud "
                "inválida).\n\nPrueba a re-exportarlo con CloudCompare, "
                "PDAL o lastools (lasinfo/las2las) y vuelve a cargarlo.\n\n"
                f"Detalle técnico: {last_err}")
        raise RuntimeError(
            f"No se pudo abrir el .laz. Prueba: pip install \"laspy[lazrs]\"\n{last_err}")

    def _las(self) -> PointCloud:
        try: import laspy
        except ImportError: raise ImportError("pip install laspy[lazrs]")
        self.progress.emit(5, "Leyendo LAS/LAZ…")
        ext = Path(self.path).suffix.lower()
        pc = PointCloud()

        # Open to get header info first — con fallback a read_evlrs=False:
        # algunos .laz/.las traen un EVLR con longitud corrupta al final
        # del archivo que hace fallar incluso la simple lectura del
        # encabezado ("read length must be non-negative or -1"); saltarse
        # los EVLRs suele bastar para leer xyz/intensity/rgb igual.
        header_opened = False
        # Recordamos con qué valor de read_evlrs se pudo abrir el archivo
        # para reutilizarlo en las siguientes llamadas a laspy.open() de
        # esta misma carga (cálculo de offset, streaming de puntos,
        # atributos) — si el encabezado solo abre con read_evlrs=False,
        # las llamadas posteriores con el valor por defecto (True)
        # fallarían igual al llegar a leer los EVLRs corruptos.
        self._laz_read_evlrs = True
        for read_evlrs in (True, False):
            try:
                with laspy.open(self.path, read_evlrs=read_evlrs) as reader:
                    n = reader.header.point_count
                    try: pc.fmt = f"LAS {reader.header.version}"
                    except: pc.fmt = "LAS"
                header_opened = True
                self._laz_read_evlrs = read_evlrs
                break
            except Exception:
                continue
        if not header_opened:
            # Fallback: probar todas las combinaciones backend/read_evlrs
            # (incluye el caso .las con EVLR corrupto, no solo .laz)
            las = self._read_laz_file(laspy, self.path)
            return self._las_from_object(las, pc)

        self.progress.emit(8, f"{n:,} puntos…")

        # Detectar heavy antes de cualquier early return
        from core.heavy_cloud import HEAVY_THRESHOLD as _HT
        if n > _HT:
            pc.is_heavy   = True
            pc._heavy_src = self.path

        # Compute offset from header bounds (no reading of coords)
        try:
            hdr = None
            with laspy.open(self.path, read_evlrs=self._laz_read_evlrs) as reader:
                hdr = reader.header
                try:
                    mn = np.array([hdr.mins[0], hdr.mins[1], hdr.mins[2]], np.float64)
                    mx = np.array([hdr.maxs[0], hdr.maxs[1], hdr.maxs[2]], np.float64)
                    center = (mn + mx) * 0.5
                except Exception:
                    center = np.zeros(3, np.float64)
            pc.offset = center
        except Exception:
            center = np.zeros(3, np.float64); pc.offset = center

        # Check if we can use mmap cache
        cache_path = _xyz_cache_path(self.path, n, center)
        if cache_path.exists():
            self.progress.emit(15, f"Cache mmap encontrado ({n/1e6:.0f}M pts)…")
            pc.xyz = np.memmap(str(cache_path), dtype=np.float32,
                               mode='r', shape=(n, 3))
            pc.is_mmap = True
            pc._cache_path = cache_path
            self.progress.emit(30, "Cargando atributos…")
            self._load_las_attrs_only(laspy, pc, n)
            return pc

        # Determine if we need mmap (huge cloud / low RAM)
        xyz_bytes = n * 12  # float32 * 3
        avail_ram = _available_ram_gb() * 1e9
        need_mmap = (n > OUT_OF_CORE_THRESHOLD) or (xyz_bytes > avail_ram * 0.4)

        if need_mmap:
            self.progress.emit(10, f"Nube grande ({n/1e6:.0f}M pts) — modo out-of-core…")
            return self._las_streaming_mmap(laspy, pc, n, center, cache_path)
        else:
            return self._las_streaming_ram(laspy, pc, n, center)

    def _las_streaming_ram(self, laspy, pc, n, center):
        """Streaming into RAM. Chunk reading keeps peak memory at 1.2x."""
        pc.xyz = np.empty((n, 3), np.float32)
        self.progress.emit(12, f"Streaming {n/1e6:.0f}M pts…")
        self._las_fill_chunks(laspy, pc, n, center)
        return pc

    def _las_streaming_mmap(self, laspy, pc, n, center, cache_path: Path):
        """
        Writes xyz to disk cache as float32, then opens as mmap.
        Peak RAM during write: one chunk (2M pts = 24MB).
        Final RAM: only OS-paged pages (typically <500MB for any cloud size).
        """
        self.progress.emit(10, f"Creando cache mmap ({n*12/1e9:.1f}GB)…")

        # Create memmap for writing
        xyz_mmap = np.memmap(str(cache_path), dtype=np.float32,
                             mode='w+', shape=(n, 3))
        pc.xyz = xyz_mmap
        pc.is_mmap = True
        pc._cache_path = cache_path

        self._las_fill_chunks(laspy, pc, n, center, xyz_target=xyz_mmap)

        # Flush and reopen read-only (allows OS to optimize paging)
        xyz_mmap.flush()
        del xyz_mmap
        pc.xyz = np.memmap(str(cache_path), dtype=np.float32,
                           mode='r', shape=(n, 3))
        return pc

    def _las_fill_chunks(self, laspy, pc, n, center, xyz_target=None):
        """Fill xyz (and attrs) from LAS via chunk_iterator."""
        chunk_size = 2_000_000
        has_int = has_rgb = has_cls = has_ret = False
        int_buf = rgb_buf = cls_buf = ret_buf = None
        rgb_16bit = False

        target = xyz_target if xyz_target is not None else pc.xyz

        # Determine if attributes should be mmap'd
        use_attr_mmap = _should_mmap(n, bytes_per_pt=8)  # 8 bytes avg for attrs
        source_path = self.path

        pos = 0
        try:
            with laspy.open(self.path, read_evlrs=getattr(self, '_laz_read_evlrs', True)) as reader:
                try:
                    crs = reader.header.parse_crs()
                    if crs: pc.crs = str(getattr(crs,"name",crs))[:200]
                except Exception: pass

                for chunk in reader.chunk_iterator(chunk_size):
                    cn = len(chunk.x)
                    end = pos + cn

                    # xyz
                    x = np.asarray(chunk.x, np.float64) - center[0]
                    y = np.asarray(chunk.y, np.float64) - center[1]
                    z = np.asarray(chunk.z, np.float64) - center[2]
                    target[pos:end, 0] = x.astype(np.float32)
                    target[pos:end, 1] = y.astype(np.float32)
                    target[pos:end, 2] = z.astype(np.float32)
                    del x, y, z

                    # Intensity
                    try:
                        iv = np.asarray(chunk.intensity, np.float32)
                        if int_buf is None:
                            if use_attr_mmap:
                                int_buf = _make_attr_mmap(source_path, n, 'intensity', np.float32, (n,))
                                if int_buf is None: int_buf = np.empty(n, np.float32)
                            else:
                                int_buf = np.empty(n, np.float32)
                            has_int = True
                        int_buf[pos:end] = iv; del iv
                    except Exception: pass

                    # Classification
                    try:
                        cv = np.asarray(chunk.classification, np.uint8)
                        if cls_buf is None:
                            if use_attr_mmap:
                                cls_buf = _make_attr_mmap(source_path, n, 'classification', np.uint8, (n,))
                                if cls_buf is None: cls_buf = np.empty(n, np.uint8)
                            else:
                                cls_buf = np.empty(n, np.uint8)
                            has_cls = True
                        cls_buf[pos:end] = cv; del cv
                    except Exception: pass

                    # RGB
                    try:
                        r = np.asarray(chunk.red, np.float32)
                        g = np.asarray(chunk.green, np.float32)
                        b = np.asarray(chunk.blue, np.float32)
                        if rgb_buf is None:
                            if use_attr_mmap:
                                rgb_buf = _make_attr_mmap(source_path, n, 'rgb', np.uint8, (n, 3))
                                if rgb_buf is None: rgb_buf = np.empty((n, 3), np.uint8)
                            else:
                                rgb_buf = np.empty((n, 3), np.uint8)
                            has_rgb = True
                        # Detectar 16-bit en cada chunk (no solo el primero)
                        chunk_max_rgb = max(float(r.max()), float(g.max()), float(b.max()))
                        if chunk_max_rgb > 255: rgb_16bit = True
                        scale = 255.0/65535.0 if rgb_16bit else 1.0
                        rgb_buf[pos:end, 0] = np.clip(r*scale, 0, 255).astype(np.uint8)
                        rgb_buf[pos:end, 1] = np.clip(g*scale, 0, 255).astype(np.uint8)
                        rgb_buf[pos:end, 2] = np.clip(b*scale, 0, 255).astype(np.uint8)
                        del r, g, b
                    except Exception: pass

                    # Return number
                    try:
                        rv = np.asarray(chunk.return_number, np.uint8)
                        if ret_buf is None:
                            ret_buf = np.empty(n, np.uint8); has_ret = True
                        ret_buf[pos:end] = rv; del rv
                    except Exception: pass

                    pos = end
                    if pos % (chunk_size * 5) < chunk_size:
                        pct = 12 + int(18 * pos / max(n, 1))
                        self.progress.emit(pct, f"{pos/1e6:.0f}M / {n/1e6:.0f}M…")

        except AttributeError:
            las = self._read_laz_file(laspy, self.path) if self.path.endswith('.laz') else laspy.read(self.path)
            self._las_from_object_fill(las, target, center, n)
            has_int = has_rgb = has_cls = has_ret = False

        # Normalize and flush mmaps
        if has_int and int_buf is not None:
            mx = float(int_buf.max())
            if mx > 0: int_buf[:] = (int_buf / mx).astype(np.float32)
            if isinstance(int_buf, np.memmap): int_buf.flush()
            pc.intensity = int_buf
        if has_cls and cls_buf is not None:
            if isinstance(cls_buf, np.memmap): cls_buf.flush()
            pc.classification = cls_buf
        if has_rgb and rgb_buf is not None:
            if isinstance(rgb_buf, np.memmap): rgb_buf.flush()
            pc.rgb = rgb_buf
        if has_ret: pc.return_num = ret_buf

    def _load_las_attrs_only(self, laspy, pc, n):
        """When xyz comes from cache, still load attrs via streaming."""
        chunk_size = 2_000_000
        has_int = has_rgb = has_cls = has_ret = False
        int_buf = rgb_buf = cls_buf = ret_buf = None
        rgb_16bit = False
        pos = 0
        try:
            with laspy.open(self.path, read_evlrs=getattr(self, '_laz_read_evlrs', True)) as reader:
                try:
                    crs = reader.header.parse_crs()
                    if crs: pc.crs = str(getattr(crs,"name",crs))[:200]
                except Exception: pass
                for chunk in reader.chunk_iterator(chunk_size):
                    cn = len(chunk.x); end = pos + cn
                    try:
                        iv = np.asarray(chunk.intensity, np.float32)
                        if int_buf is None: int_buf = np.empty(n, np.float32); has_int=True
                        int_buf[pos:end] = iv
                    except: pass
                    try:
                        cv = np.asarray(chunk.classification, np.uint8)
                        if cls_buf is None: cls_buf = np.empty(n, np.uint8); has_cls=True
                        cls_buf[pos:end] = cv
                    except: pass
                    try:
                        r=np.asarray(chunk.red,np.float32); g=np.asarray(chunk.green,np.float32); b=np.asarray(chunk.blue,np.float32)
                        if rgb_buf is None: rgb_buf=np.empty((n,3),np.uint8); has_rgb=True
                        chunk_max=max(float(r.max()),float(g.max()),float(b.max()))
                        if chunk_max>255: rgb_16bit=True
                        s=255./65535. if rgb_16bit else 1.
                        rgb_buf[pos:end,0]=np.clip(r*s,0,255).astype(np.uint8)
                        rgb_buf[pos:end,1]=np.clip(g*s,0,255).astype(np.uint8)
                        rgb_buf[pos:end,2]=np.clip(b*s,0,255).astype(np.uint8)
                    except: pass
                    pos=end
        except AttributeError:
            pass
        if has_int and int_buf is not None:
            mx=int_buf.max()
            if mx>0: int_buf/=mx
            pc.intensity=int_buf
        if has_cls: pc.classification=cls_buf
        if has_rgb: pc.rgb=rgb_buf

    def _las_from_object(self, las, pc):
        """Fallback: load from already-read LAS object."""
        try: pc.fmt = f"LAS {las.header.version}"
        except: pc.fmt = "LAS"
        n = las.header.point_count
        self.progress.emit(14, f"{n:,} puntos…")
        x=np.asarray(las.x,np.float64); y=np.asarray(las.y,np.float64); z=np.asarray(las.z,np.float64)
        cx,cy,cz=x.mean(),y.mean(),z.mean(); pc.offset=np.array([cx,cy,cz],np.float64)
        x-=cx; y-=cy; z-=cz
        pc.xyz=np.ascontiguousarray(np.column_stack([x,y,z]).astype(np.float32))
        del x,y,z
        try:
            crs=las.header.parse_crs()
            if crs: pc.crs=str(getattr(crs,"name",crs))[:200]
        except: pass
        self.progress.emit(30,"Atributos…")
        try:
            i=np.asarray(las.intensity,np.float32); mx=i.max()
            if mx>0: pc.intensity=i/mx
        except: pass
        try: pc.classification=np.asarray(las.classification,np.uint8)
        except: pass
        try:
            r=np.asarray(las.red,np.float32); g=np.asarray(las.green,np.float32); b=np.asarray(las.blue,np.float32)
            mx=max(r.max(),g.max(),b.max())
            if mx>0:
                s=255./(65535. if mx>255 else 255.)
                pc.rgb=np.column_stack([r*s,g*s,b*s]).astype(np.uint8)
        except: pass
        try: pc.return_num=np.asarray(las.return_number,np.uint8)
        except: pass
        return pc

    def _las_from_object_fill(self, las, target, center, n):
        """Fill mmap/array from LAS object without chunk_iterator."""
        x=np.asarray(las.x,np.float64)-center[0]; y=np.asarray(las.y,np.float64)-center[1]; z=np.asarray(las.z,np.float64)-center[2]
        target[:,0]=x.astype(np.float32); target[:,1]=y.astype(np.float32); target[:,2]=z.astype(np.float32)

    # ── E57 ──────────────────────────────────────────────────────────────────

    def _e57(self) -> PointCloud:
        """
        Load E57 format. Supports:
        - Cartesian (cartesianX/Y/Z) — most common
        - Spherical (sphericalRange/Azimuth/Elevation) — some scanners
        - Multi-scan files (concatenates all scans)
        """
        try:
            import pye57
        except ImportError:
            raise ImportError("E57 requiere pye57: pip install pye57")

        self.progress.emit(5, "Leyendo E57…")
        pc = PointCloud(); pc.fmt = "E57"

        try:
            e57 = pye57.E57(self.path)
        except Exception as e:
            raise RuntimeError(f"No se pudo abrir E57: {e}")

        n_scans = e57.scan_count
        if n_scans == 0:
            raise ValueError("E57: El archivo no contiene escaneos")

        # Detectar heavy para E57
        try:
            _n_e57 = sum(int(e57.get_header(i).point_count) for i in range(n_scans))
            from core.heavy_cloud import HEAVY_THRESHOLD as _HT
            if _n_e57 > _HT:
                pc.is_heavy = True; pc._heavy_src = self.path
        except Exception: pass

        self.progress.emit(8, f"E57: {n_scans} escaneo(s)…")

        all_xyz = []
        all_int = []
        all_rgb = []
        has_int = has_rgb = False

        for scan_idx in range(n_scans):
            pct = 8 + int(22 * scan_idx / max(n_scans, 1))
            self.progress.emit(pct, f"Escaneo {scan_idx+1}/{n_scans}…")
            try:
                data = e57.read_scan(scan_idx, intensity=True, colors=True,
                                     row_column=False, ignore_missing_fields=True)
            except Exception as e:
                print(f"[E57] scan {scan_idx} error: {e}")
                continue

            # ── Extract XYZ (Cartesian or Spherical) ──────────────────
            xyz = None

            # Try Cartesian first (most common)
            if 'cartesianX' in data:
                try:
                    x = np.asarray(data['cartesianX'], np.float64)
                    y = np.asarray(data['cartesianY'], np.float64)
                    z = np.asarray(data['cartesianZ'], np.float64)
                    xyz = np.column_stack([x, y, z])
                    del x, y, z
                except Exception: pass

            # Fallback: Spherical coordinates
            if xyz is None and 'sphericalRange' in data:
                try:
                    r   = np.asarray(data['sphericalRange'],     np.float64)
                    az  = np.asarray(data['sphericalAzimuth'],   np.float64)
                    el  = np.asarray(data['sphericalElevation'], np.float64)
                    x = r * np.cos(el) * np.cos(az)
                    y = r * np.cos(el) * np.sin(az)
                    z = r * np.sin(el)
                    xyz = np.column_stack([x, y, z])
                    del r, az, el, x, y, z
                except Exception: pass

            if xyz is None or len(xyz) == 0:
                print(f"[E57] scan {scan_idx}: no XYZ data found, skipping")
                continue

            # ── Filter invalid points ──────────────────────────────────
            valid = None
            if 'cartesianInvalidState' in data:
                valid_arr = np.asarray(data['cartesianInvalidState'])
                valid = valid_arr == 0
            elif 'returnInvalidState' in data:
                valid_arr = np.asarray(data['returnInvalidState'])
                valid = valid_arr == 0

            # Also filter NaN/Inf
            finite_mask = np.isfinite(xyz).all(axis=1)
            valid = finite_mask if valid is None else (valid & finite_mask)

            if not valid.all():
                xyz = xyz[valid]
            if len(xyz) == 0:
                continue

            # ── Optional attributes ────────────────────────────────────
            if 'intensity' in data:
                try:
                    iv = np.asarray(data['intensity'], np.float32)
                    if valid is not None and len(iv) > len(xyz):
                        iv = iv[valid]
                    all_int.append(iv); has_int = True
                except Exception: pass

            for r_key, g_key, b_key in [('colorRed','colorGreen','colorBlue'),
                                          ('red','green','blue')]:
                if all(k in data for k in (r_key, g_key, b_key)):
                    try:
                        r = np.asarray(data[r_key])
                        g = np.asarray(data[g_key])
                        b = np.asarray(data[b_key])
                        if valid is not None and len(r) > len(xyz):
                            r, g, b = r[valid], g[valid], b[valid]
                        # Normalize to 0-255
                        mx = max(float(r.max()), float(g.max()), float(b.max()))
                        if mx > 0:
                            scale = 255.0 / (65535.0 if mx > 255 else mx)
                            rgb_arr = np.column_stack([
                                np.clip(r * scale, 0, 255).astype(np.uint8),
                                np.clip(g * scale, 0, 255).astype(np.uint8),
                                np.clip(b * scale, 0, 255).astype(np.uint8)])
                            all_rgb.append(rgb_arr); has_rgb = True
                        break
                    except Exception: pass

            all_xyz.append(xyz.astype(np.float32))  # store float32 to save RAM
            del xyz

        if not all_xyz:
            raise ValueError(
                "E57: No se encontraron puntos válidos.\n"
                "Verifica que el archivo tenga datos cartesianos (cartesianX/Y/Z).")

        self.progress.emit(30, f"Concatenando {len(all_xyz)} escaneo(s)…")

        # Concatenate all scans
        xyz_all = np.vstack(all_xyz); del all_xyz
        n = len(xyz_all)

        # Compute offset (from mean or bounds)
        center = xyz_all.mean(0).astype(np.float64)
        pc.offset = center
        center_f32 = center.astype(np.float32)
        xyz_all -= center_f32  # in-place, float32
        self.progress.emit(35, f"{n:,} puntos cargados…")

        # Use mmap if cloud is too large
        avail_ram = _available_ram_gb() * 1e9
        if n > OUT_OF_CORE_THRESHOLD or n * 12 > avail_ram * 0.4:
            cache_path = _xyz_cache_path(self.path, n, center)
            mmap = np.memmap(str(cache_path), dtype=np.float32, mode='w+', shape=(n, 3))
            mmap[:] = xyz_all
            mmap.flush(); del mmap; del xyz_all
            pc.xyz = np.memmap(str(cache_path), dtype=np.float32, mode='r', shape=(n, 3))
            pc.is_mmap = True; pc._cache_path = cache_path
        else:
            pc.xyz = np.ascontiguousarray(xyz_all); del xyz_all

        if has_int and all_int:
            iv = np.concatenate(all_int).astype(np.float32)
            mx = iv.max()
            if mx > 0: iv /= mx
            pc.intensity = iv

        if has_rgb and all_rgb:
            pc.rgb = np.vstack(all_rgb).astype(np.uint8)

        self.progress.emit(40, f"E57 listo: {n:,} puntos ✓")
        return pc

    # ── PLY ──────────────────────────────────────────────────────────────────

    def _ply(self) -> PointCloud:
        self.progress.emit(5, "Abriendo PLY…")
        pc = PointCloud(); pc.fmt = "PLY"
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(self.path)
            pts = np.asarray(pcd.points, np.float64); c = pts.mean(0)
            pc.offset=c; pc.xyz=np.ascontiguousarray((pts-c).astype(np.float32))
            if pcd.has_colors(): pc.rgb=(np.asarray(pcd.colors)*255).astype(np.uint8)
            return pc
        except ImportError: pass
        with open(self.path,"rb") as f:
            props=[]; n=0; is_le=is_be=False
            while True:
                line=f.readline().decode("ascii",errors="ignore").strip()
                if line.startswith("format binary_little_endian"): is_le=True
                elif line.startswith("format binary_big_endian"): is_be=True
                elif line.startswith("element vertex"): n=int(line.split()[-1])
                elif line.startswith("property"): p=line.split(); props.append((p[1],p[2]))
                elif line=="end_header": break
            pn=[p[1] for p in props]
            dm={"float":"f4","double":"f8","uchar":"u1","uint8":"u1","int":"i4","short":"i2"}
            end=">\" if is_be else \"<"
            if is_le or is_be:
                dt=np.dtype([(nm,end+dm.get(t,"f4")) for t,nm in props])
                data=np.frombuffer(f.read(n*dt.itemsize),dtype=dt)
            else:
                arr=np.loadtxt(io.BytesIO(f.read()))
                arr=arr.reshape(-1,len(props)) if arr.ndim==1 else arr
                c=arr[:,:3].mean(0); pc.offset=c
                pc.xyz=np.ascontiguousarray((arr[:,:3]-c).astype(np.float32)); return pc
        xi,yi,zi=pn.index("x"),pn.index("y"),pn.index("z")
        x=data[pn[xi]].astype(np.float64); y=data[pn[yi]].astype(np.float64)
        z=data[pn[zi]].astype(np.float64); c=np.array([x.mean(),y.mean(),z.mean()])
        pc.offset=c
        pc.xyz=np.ascontiguousarray(np.column_stack([(x-c[0]),(y-c[1]),(z-c[2])]).astype(np.float32))
        for rn in ["red","r"]:
            if rn in pn:
                gn,bn=("green","blue") if rn=="red" else ("g","b")
                if gn in pn and bn in pn:
                    pc.rgb=np.column_stack([data[rn],data[gn],data[bn]]).astype(np.uint8); break
        return pc

    # ── PCD ──────────────────────────────────────────────────────────────────

    def _pcd(self) -> PointCloud:
        self.progress.emit(5,"Abriendo PCD…"); pc=PointCloud(); pc.fmt="PCD"
        with open(self.path,"rb") as f:
            fields=[]; sizes=[]; types=[]; w=h=0; dtp="ascii"
            while True:
                line=f.readline().decode("ascii",errors="ignore").strip()
                if line.startswith("FIELDS"): fields=line.split()[1:]
                elif line.startswith("SIZE"): sizes=[int(x) for x in line.split()[1:]]
                elif line.startswith("TYPE"): types=line.split()[1:]
                elif line.startswith("WIDTH"): w=int(line.split()[1])
                elif line.startswith("HEIGHT"): h=int(line.split()[1])
                elif line.startswith("DATA"): dtp=line.split()[1]; break
            n=w*h
            tm={"F":{4:np.float32,8:np.float64},"U":{1:np.uint8,2:np.uint16,4:np.uint32},"I":{1:np.int8,2:np.int16,4:np.int32}}
            if dtp=="binary":
                dt=np.dtype([(fields[i],tm.get(types[i],{}).get(sizes[i],np.float32)) for i in range(len(fields))])
                data=np.frombuffer(f.read(n*dt.itemsize),dtype=dt)
                x=data["x"].astype(np.float64); y=data["y"].astype(np.float64); z=data["z"].astype(np.float64)
            else:
                arr=np.loadtxt(io.BytesIO(f.read())); x,y,z=arr[:,0].astype(np.float64),arr[:,1].astype(np.float64),arr[:,2].astype(np.float64)
        c=np.array([x.mean(),y.mean(),z.mean()]); pc.offset=c
        pc.xyz=np.ascontiguousarray(np.column_stack([(x-c[0]),(y-c[1]),(z-c[2])]).astype(np.float32)); return pc

    # ── XYZ/TXT ──────────────────────────────────────────────────────────────

    def _xyz_txt(self) -> PointCloud:
        self.progress.emit(5,"Detectando formato…"); pc=PointCloud(); pc.fmt="XYZ"
        with open(self.path,"r",errors="ignore") as f: head=[f.readline() for _ in range(10)]
        skip=0
        for line in head:
            s=line.strip()
            if not s: skip+=1; continue
            try: float(s.split()[0]); break
            except: skip+=1
        sample=head[skip] if skip<len(head) else head[-1]
        sep = ("," if "," in sample else "\t" if "\t" in sample else ";" if ";" in sample else None)
        self.progress.emit(15,"Cargando…")
        try: data=np.genfromtxt(self.path,delimiter=sep,skip_header=skip,dtype=np.float32,invalid_raise=False,filling_values=np.nan)
        except: data=np.loadtxt(self.path,delimiter=sep,skiprows=skip,dtype=np.float32)
        if data.ndim==1: data=data.reshape(1,-1)
        data=data[~np.isnan(data[:,:3]).any(1)]
        pts=data[:,:3].astype(np.float64); c=pts.mean(0); pc.offset=c
        pc.xyz=np.ascontiguousarray((pts-c).astype(np.float32))
        if data.shape[1]>=4:
            iv=data[:,3].astype(np.float32); mx=float(np.nanmax(iv))
            if not np.isnan(mx) and mx>0: pc.intensity=iv/mx
        if data.shape[1]>=6:
            cs=3 if data.shape[1]==6 else 4
            rgb=data[:,cs:cs+3].astype(np.float32); mx=float(np.nanmax(rgb))
            if mx>1: pc.rgb=np.clip(rgb,0,255).astype(np.uint8)
            elif mx>0: pc.rgb=(rgb*255).astype(np.uint8)
        return pc

    # ── NPY ──────────────────────────────────────────────────────────────────

    def _npy(self) -> PointCloud:
        self.progress.emit(20,"NPY…"); arr=np.load(self.path)
        if arr.ndim!=2 or arr.shape[1]<3: raise ValueError("NPY debe ser (N,3+)")
        pc=PointCloud(); pc.fmt="NPY"
        pts=arr[:,:3].astype(np.float64); c=pts.mean(0); pc.offset=c
        pc.xyz=np.ascontiguousarray((pts-c).astype(np.float32))
        if arr.shape[1]>=4:
            iv=arr[:,3].astype(np.float32); mx=iv.max()
            if mx>0: pc.intensity=iv/mx
        return pc

    def _ga3d_bin(self) -> "PointCloud":
        """Carga un .ga3d_bin directamente usando el loader heavy_cloud."""
        from core.heavy_cloud import load_ga3d_bin
        return load_ga3d_bin(self.path)
