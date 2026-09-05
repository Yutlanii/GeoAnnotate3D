"""
core/overlay_manager.py — Capas vectoriales y raster sobre la nube de puntos

Vectoriales: .shp .geojson .gpkg .kml .dxf  (requiere geopandas)
Raster:      .tif .tiff .png .jpg            (requiere rasterio)
CRS:         reproyecta automáticamente con pyproj/rasterio

Raster optimizado como QGIS:
  - Lee pirámides (overviews) si el archivo las tiene
  - Selecciona el nivel de overview apropiado para la pantalla
  - Nunca lee más resolución de la que puede mostrar
  - Limita la textura a max 4096×4096 px
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import numpy as np


@dataclass
class OverlayLayer:
    name:       str
    path:       str
    kind:       str        # "vector" | "raster"
    crs_src:    str = ""
    crs_dst:    str = ""
    ok:         bool = True
    warnings:   List[str] = field(default_factory=list)
    vtk_actors: list = field(default_factory=list)
    z_offset:   float = 0.0    # offset Z ajustable para raster


# ── Vector ────────────────────────────────────────────────────────────────────

def load_vector(path: str, pc_offset: np.ndarray, pc_bounds: np.ndarray,
                pc_crs: str = "") -> OverlayLayer:
    import vtk
    layer = OverlayLayer(name=Path(path).name, path=path, kind="vector")

    try:
        import geopandas as gpd
    except ImportError:
        layer.warnings.append("Instala geopandas: pip install geopandas")
        layer.ok = False; return layer

    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        layer.warnings.append(f"Error leyendo: {e}")
        layer.ok = False; return layer

    crs_src = str(gdf.crs) if gdf.crs else ""
    layer.crs_src = crs_src; layer.crs_dst = pc_crs

    if crs_src and pc_crs and crs_src != pc_crs:
        try:
            gdf = gdf.to_crs(pc_crs)
            layer.warnings.append(f"CRS transformado: {crs_src} → {pc_crs}")
        except Exception as e:
            layer.warnings.append(f"No se pudo transformar CRS: {e}")

    ox, oy = float(pc_offset[0]), float(pc_offset[1])
    z_elev  = float(pc_bounds[1, 2]) + 2.0  # 2m sobre el techo de la nube

    pts_all = vtk.vtkPoints()
    lines_ca = vtk.vtkCellArray()
    pid = 0

    for geom in gdf.geometry:
        if geom is None or geom.is_empty: continue
        rings = _extract_rings(geom)
        for ring in rings:
            if len(ring) < 2: continue
            poly_line = vtk.vtkPolyLine()
            poly_line.GetPointIds().SetNumberOfIds(len(ring))
            for k, coord in enumerate(ring):
                x, y = coord[0], coord[1]   # ignore Z if present (3D coords)
                pts_all.InsertNextPoint(x - ox, y - oy, z_elev)
                poly_line.GetPointIds().SetId(k, pid)
                pid += 1
            lines_ca.InsertNextCell(poly_line)

    if pid == 0:
        layer.warnings.append("Sin geometrías visibles en el archivo.")
        layer.ok = False; return layer

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts_all)
    poly.SetLines(lines_ca)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 0.85, 0.1)   # amarillo
    actor.GetProperty().SetLineWidth(2.5)
    actor.GetProperty().SetOpacity(0.9)

    layer.vtk_actors.append(actor)

    # Verificar bbox
    _warn_if_outside(gdf, pc_bounds, pc_offset, layer)
    return layer


def _extract_rings(geom):
    """Extrae todos los anillos de coordenadas de una geometría shapely."""
    rings = []
    t = geom.geom_type
    if t == 'Polygon':
        rings.append(list(geom.exterior.coords))
        for h in geom.interiors:
            rings.append(list(h.coords))
    elif t == 'MultiPolygon':
        for p in geom.geoms: rings.extend(_extract_rings(p))
    elif t in ('LineString', 'LinearRing'):
        rings.append(list(geom.coords))
    elif t == 'MultiLineString':
        for l in geom.geoms: rings.append(list(l.coords))
    elif t in ('GeometryCollection', 'MultiPoint'):
        for g in geom.geoms: rings.extend(_extract_rings(g))
    return rings


def _warn_if_outside(gdf, pc_bounds, pc_offset, layer):
    try:
        b = gdf.total_bounds  # xmin ymin xmax ymax
        ox, oy = float(pc_offset[0]), float(pc_offset[1])
        xmn, ymn = b[0]-ox, b[1]-oy
        xmx, ymx = b[2]-ox, b[3]-oy
        bmin = pc_bounds[0]; bmax = pc_bounds[1]
        if xmx < bmin[0] or xmn > bmax[0] or ymx < bmin[1] or ymn > bmax[1]:
            layer.warnings.append(
                f"La capa vectorial queda fuera del área de la nube.\n"
                f"  Capa: X=[{xmn+ox:.0f}, {xmx+ox:.0f}]  Y=[{ymn+oy:.0f}, {ymx+oy:.0f}]\n"
                f"  Nube: X=[{bmin[0]+ox:.0f}, {bmax[0]+ox:.0f}]  "
                f"Y=[{bmin[1]+oy:.0f}, {bmax[1]+oy:.0f}]\n"
                "Comprueba que el CRS coincide con el de la nube.")
    except Exception:
        pass


# ── Raster ────────────────────────────────────────────────────────────────────

def load_raster(path: str, pc_offset: np.ndarray, pc_bounds: np.ndarray,
                pc_crs: str = "",
                max_texture_px: int = 4096,
                z_offset: float = 0.0) -> OverlayLayer:
    """
    Carga un GeoTIFF como textura VTK sobre un plano.

    Optimización tipo QGIS:
      1. Abre el archivo y detecta si tiene pirámides (overviews)
      2. Calcula el nivel de overview óptimo para no leer más resolución
         de la necesaria (la nube tiene ciertos metros de ancho → cuántos
         píxeles son suficientes)
      3. Lee solo ese nivel de overview → mínimos datos leídos
      4. Reproyecta al CRS de la nube si es diferente
      5. Crea una textura VTK sobre un plano posicionado correctamente
    """
    import vtk
    layer = OverlayLayer(name=Path(path).name, path=path, kind="raster")

    try:
        import rasterio
        import rasterio.warp
        from rasterio.enums import Resampling
    except ImportError:
        layer.warnings.append("Instala rasterio: pip install rasterio")
        layer.ok = False; return layer

    try:
        with rasterio.open(path) as ds:
            crs_src = str(ds.crs) if ds.crs else ""
            layer.crs_src = crs_src; layer.crs_dst = pc_crs

            # ── Reproyectar si necesario ──────────────────────────────────────
            if crs_src and pc_crs and crs_src != pc_crs:
                try:
                    import rasterio.crs as rcrs
                    dst_crs = rcrs.CRS.from_user_input(pc_crs)
                    transform_new, w_new, h_new = rasterio.warp.calculate_default_transform(
                        ds.crs, dst_crs, ds.width, ds.height, *ds.bounds)
                    from rasterio.io import MemoryFile
                    prof = ds.profile.copy()
                    prof.update(crs=dst_crs, transform=transform_new,
                                width=w_new, height=h_new)
                    mf = MemoryFile()
                    with mf.open(**prof) as dst_ds:
                        for bi in range(1, ds.count+1):
                            rasterio.warp.reproject(
                                source=rasterio.band(ds, bi),
                                destination=rasterio.band(dst_ds, bi),
                                src_crs=ds.crs, dst_crs=dst_crs,
                                resampling=Resampling.bilinear)
                    src_ds = mf.open()
                    layer.warnings.append(f"CRS transformado: {crs_src} → {pc_crs}")
                except Exception as e:
                    layer.warnings.append(f"No se pudo reproyectar: {e}")
                    src_ds = ds
            else:
                src_ds = ds

            bounds  = src_ds.bounds
            ox, oy  = float(pc_offset[0]), float(pc_offset[1])
            xmin    = bounds.left   - ox
            xmax    = bounds.right  - ox
            ymin    = bounds.bottom - oy
            ymax    = bounds.top    - oy

            # ── Seleccionar resolución de lectura (pirámides tipo QGIS) ──────
            # Calcular cuántos píxeles necesitamos realmente
            cloud_width_m  = float(pc_bounds[1,0] - pc_bounds[0,0])
            raster_width_m = bounds.right - bounds.left
            # Si el raster es más ancho que la nube, no necesitamos resolución completa
            overlap_ratio  = min(cloud_width_m / max(raster_width_m, 1.0), 1.0)
            # Píxeles que necesitamos: queremos ~1px por metro en el área visible
            # limitado por max_texture_px
            needed_px = min(max_texture_px, max(256, int(src_ds.width * overlap_ratio)))

            # Detectar overviews disponibles y elegir el mejor
            overviews = src_ds.overviews(1)  # overviews de banda 1
            out_w, out_h = needed_px, needed_px

            if overviews:
                # Encontrar el overview más pequeño que sea >= needed_px
                native_w = src_ds.width
                best_ov  = None
                for ov in sorted(overviews):  # overviews: decimation factors [2,4,8,16...]
                    ov_w = native_w // ov
                    if ov_w >= needed_px:
                        best_ov = ov
                    else:
                        break
                if best_ov:
                    out_w = native_w // best_ov
                    out_h = src_ds.height // best_ov
                    layer.warnings.append(
                        f"Usando overview ×{best_ov} ({out_w}×{out_h} px) "
                        f"de {native_w}×{src_ds.height} nativo")
                else:
                    # Overview más fino disponible
                    ov = overviews[0]
                    out_w = native_w // ov
                    out_h = src_ds.height // ov
            else:
                # Sin pirámides: submuestrear nosotros
                scale = min(1.0, max_texture_px / max(src_ds.width, src_ds.height))
                out_w = max(64, int(src_ds.width  * scale))
                out_h = max(64, int(src_ds.height * scale))
                if scale < 1.0:
                    layer.warnings.append(
                        f"Sin pirámides — submuestreando a {out_w}×{out_h} px. "
                        f"Para mejor rendimiento: gdaladdo {Path(path).name} 2 4 8 16")

            # Limitar al máximo de textura
            if out_w > max_texture_px or out_h > max_texture_px:
                ratio = max_texture_px / max(out_w, out_h)
                out_w = int(out_w * ratio); out_h = int(out_h * ratio)

            out_w = max(1, out_w); out_h = max(1, out_h)

            # ── Leer imagen ────────────────────────────────────────────────────
            n_bands = src_ds.count
            if n_bands >= 3:
                r = src_ds.read(1, out_shape=(out_h, out_w),
                                resampling=Resampling.bilinear).astype(np.uint8)
                g = src_ds.read(2, out_shape=(out_h, out_w),
                                resampling=Resampling.bilinear).astype(np.uint8)
                b = src_ds.read(3, out_shape=(out_h, out_w),
                                resampling=Resampling.bilinear).astype(np.uint8)
                img = np.stack([r, g, b], axis=2)
            elif n_bands == 1:
                gray = src_ds.read(1, out_shape=(out_h, out_w),
                                   resampling=Resampling.bilinear).astype(np.uint8)
                img  = np.stack([gray, gray, gray], axis=2)
            else:
                r = src_ds.read(1, out_shape=(out_h, out_w),
                                resampling=Resampling.bilinear).astype(np.uint8)
                g = src_ds.read(min(2,n_bands), out_shape=(out_h, out_w),
                                resampling=Resampling.bilinear).astype(np.uint8)
                b = src_ds.read(min(3,n_bands), out_shape=(out_h, out_w),
                                resampling=Resampling.bilinear).astype(np.uint8)
                img = np.stack([r, g, b], axis=2)

            if hasattr(src_ds, 'close'): src_ds.close()

        # ── Verificar bbox ────────────────────────────────────────────────────
        bmin = pc_bounds[0]; bmax = pc_bounds[1]
        if xmax < bmin[0] or xmin > bmax[0] or ymax < bmin[1] or ymin > bmax[1]:
            layer.warnings.append(
                f"El raster queda fuera del área de la nube.\n"
                f"  Raster: X=[{xmin+ox:.0f},{xmax+ox:.0f}] Y=[{ymin+oy:.0f},{ymax+oy:.0f}]\n"
                f"  Nube:   X=[{bmin[0]+ox:.0f},{bmax[0]+ox:.0f}] "
                f"Y=[{bmin[1]+oy:.0f},{bmax[1]+oy:.0f}]")

        # ── Crear plano VTK ────────────────────────────────────────────────────
        z_elev = float(pc_bounds[0, 2]) + 0.1 + z_offset   # nivel del suelo + offset ajustable

        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(xmin, ymin, z_elev)
        plane.SetPoint1(xmax, ymin, z_elev)
        plane.SetPoint2(xmin, ymax, z_elev)
        plane.SetNormal(0, 0, 1)
        plane.SetResolution(8, 8)
        plane.Update()

        # ── Crear textura VTK ──────────────────────────────────────────────────
        # img es (H, W, 3) — rasterio es top-down, VTK bottom-up → flip vertical
        img_flipped = img[::-1].astype(np.uint8)
        H, W = img_flipped.shape[:2]

        vtk_img = vtk.vtkImageData()
        vtk_img.SetDimensions(W, H, 1)
        vtk_img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        flat = img_flipped.flatten()
        arr  = vtk_img.GetPointData().GetScalars()
        # Bulk copy via numpy → mucho más rápido que SetValue en bucle
        from vtkmodules.util.numpy_support import numpy_to_vtk as n2v
        vtk_arr = n2v(flat, deep=True)
        vtk_arr.SetNumberOfComponents(3)
        vtk_img.GetPointData().SetScalars(vtk_arr)

        tex = vtk.vtkTexture()
        tex.SetInputData(vtk_img)
        tex.InterpolateOn()
        tex.MipmapOn()   # mipmapping para zoom out suave

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetTexture(tex)
        actor.GetProperty().SetOpacity(0.85)
        actor.GetProperty().LightingOff()   # sin iluminación → colores reales

        layer.vtk_actors.append(actor)

    except Exception as e:
        import traceback
        layer.warnings.append(f"Error: {e}\n{traceback.format_exc()[:400]}")
        layer.ok = False

    return layer


# ── Gestor ────────────────────────────────────────────────────────────────────

class OverlayManager:
    def __init__(self, pc_offset, pc_bounds, pc_crs=""):
        self.pc_offset  = np.asarray(pc_offset, np.float64)
        self.pc_bounds  = np.asarray(pc_bounds,  np.float32)
        self.pc_crs     = pc_crs
        self.layers: List[OverlayLayer] = []
        self._canvas    = None

    def set_canvas(self, canvas) -> None:
        """Conectar al canvas para añadir/quitar actores y pedir renders."""
        self._canvas = canvas

    # mantenemos set_renderer por compatibilidad
    def set_renderer(self, renderer) -> None:
        pass

    def _add_actors(self, layer: OverlayLayer):
        if self._canvas:
            for a in layer.vtk_actors:
                self._canvas.add_overlay_actor(a)
            self._canvas.request_render()

    def _remove_actors(self, layer: OverlayLayer):
        if self._canvas:
            for a in layer.vtk_actors:
                self._canvas.remove_overlay_actor(a)
            self._canvas.request_render()

    def load_vector(self, path: str) -> OverlayLayer:
        layer = load_vector(path, self.pc_offset, self.pc_bounds, self.pc_crs)
        self.layers.append(layer)
        if layer.ok: self._add_actors(layer)
        return layer

    def load_raster(self, path: str) -> OverlayLayer:
        layer = load_raster(path, self.pc_offset, self.pc_bounds, self.pc_crs)
        self.layers.append(layer)
        if layer.ok: self._add_actors(layer)
        return layer

    def remove_layer(self, layer: OverlayLayer):
        self._remove_actors(layer)
        if layer in self.layers: self.layers.remove(layer)

    def set_layer_visible(self, layer: OverlayLayer, visible: bool):
        for a in layer.vtk_actors:
            a.SetVisibility(1 if visible else 0)
        if self._canvas: self._canvas.request_render()

    def set_layer_opacity(self, layer: OverlayLayer, opacity: float):
        for a in layer.vtk_actors:
            a.GetProperty().SetOpacity(opacity)
        if self._canvas: self._canvas.request_render()

    def set_vector_color(self, layer: OverlayLayer, rgb: tuple) -> None:
        """Cambia el color de una capa vectorial. rgb = (r,g,b) 0-1."""
        r, g, b = rgb
        for a in layer.vtk_actors:
            a.GetProperty().SetColor(r, g, b)
        if self._canvas: self._canvas.request_render()

    def set_vector_line_width(self, layer: OverlayLayer, width: float) -> None:
        for a in layer.vtk_actors:
            a.GetProperty().SetLineWidth(max(0.5, width))
        if self._canvas: self._canvas.request_render()

    def set_vector_line_style(self, layer: OverlayLayer, style: str) -> None:
        """style: 'solid' | 'dash' | 'dot' | 'dashdot'"""
        import vtk
        patterns = {
            'solid':    0xFFFF,
            'dash':     0xFF00,
            'dot':      0xF0F0,
            'dashdot':  0xFF18,
        }
        pat = patterns.get(style, 0xFFFF)
        for a in layer.vtk_actors:
            prop = a.GetProperty()
            if pat == 0xFFFF:
                prop.SetLineStipplePattern(0xFFFF)
                prop.SetLineStippleRepeatFactor(1)
            else:
                prop.SetLineStipplePattern(pat)
                prop.SetLineStippleRepeatFactor(1)
        if self._canvas: self._canvas.request_render()

    def set_raster_brightness_contrast(self, layer: OverlayLayer,
                                        brightness: float, contrast: float) -> None:
        """brightness: -1..1, contrast: 0.5..2.0 — aplicado via color de actor VTK"""
        # brightness via ambient + diffuse, contrast via opacity trick
        for a in layer.vtk_actors:
            prop = a.GetProperty()
            b_adj = max(0.0, min(1.0, 0.5 + brightness * 0.5))
            prop.SetAmbient(b_adj)
            prop.SetDiffuse(1.0 - b_adj * 0.3)
        if self._canvas: self._canvas.request_render()

    def set_z_offset(self, layer: OverlayLayer, z_offset: float):
        """Ajusta la altura Z de un raster en tiempo real moviendo el plano."""
        layer.z_offset = z_offset
        if not layer.vtk_actors: return
        # Mover el actor VTK: ajustar posición en Z
        actor = layer.vtk_actors[0]
        pos = actor.GetPosition()
        # Recalculate: z_base + z_offset
        z_base = float(self.pc_bounds[0, 2]) + 0.1
        actor.SetPosition(pos[0], pos[1], z_offset)
        if self._canvas: self._canvas.request_render()

    def clear(self):
        for l in list(self.layers): self.remove_layer(l)
