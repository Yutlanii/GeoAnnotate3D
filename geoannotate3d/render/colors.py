"""
render/colors.py — Motor de color v2.0

Cambios vs v1.0:
  - compute_colors_u8(): retorna uint8 DIRECTAMENTE — elimina cadena circular
    f32 LUT → f32 → *255 → u8 → /255 → f32. Ahorra ~950ms para 15M pts.
  - build_annotation_lut_u8(): LUT (256,4) uint8 pre-cacheada
  - _enhance_colors_u8(): aplica gamma/sat/bright y retorna u8 directo
  - _apply_lut_u8(): LUT lookup directo en uint8
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np

_LUT_CACHE_F32: Dict[str, np.ndarray] = {}
_LUT_CACHE_U8:  Dict[str, np.ndarray] = {}
_CLASS_LUT_CACHE: Optional[np.ndarray] = None
_CLASS_LUT_U8_CACHE: Optional[np.ndarray] = None
_Z_RANGE_CACHE: Dict[int, Tuple[float, float]] = {}

_VIS_SATURATION = 1.45
_VIS_BRIGHTNESS = 1.20
_VIS_GAMMA      = 0.75

LAS_CLASS_COLORS: Dict[int, Tuple[float,float,float]] = {
    0:(0.50,0.50,0.50), 1:(0.75,0.75,0.75), 2:(0.60,0.40,0.15),
    3:(0.30,0.85,0.30), 4:(0.10,0.70,0.10), 5:(0.00,0.50,0.00),
    6:(1.00,0.30,0.30), 7:(1.00,0.00,0.00), 8:(0.90,0.90,0.00),
    9:(0.20,0.45,1.00),10:(0.65,0.65,1.00),11:(0.95,0.65,0.20),
   12:(0.75,0.40,0.95),13:(0.40,0.75,0.95),14:(0.20,0.65,0.85),
   15:(1.00,0.85,0.20),17:(0.95,0.72,0.20),18:(0.85,0.42,0.85),
}

# ── C fast module ────────────────────────────────────────────────────────────
try:
    from core._fast import (apply_lut_annotation as _c_apply_lut,
                             compute_z_colors_fast as _c_z_colors,
                             enhance_colors_fast as _c_enhance,
                             float_to_u8_fast as _c_f2u8,
                             _USE_C)
    _FAST = _USE_C
except ImportError:
    _FAST = False


def set_z_range_cache(xyz_id, z_p1, z_p99):
    _Z_RANGE_CACHE.clear()
    _Z_RANGE_CACHE[xyz_id] = (z_p1, z_p99)


def get_z_range(xyz):
    key = id(xyz)
    if key in _Z_RANGE_CACHE:
        return _Z_RANGE_CACHE[key]
    z = xyz[:,2]; n = len(z)
    sample = z[::max(1, n//200_000)] if n > 500_000 else z
    p1, p99 = np.percentile(sample, [1, 99])
    return float(p1), float(p99)


# ── LUT caches ───────────────────────────────────────────────────────────────

def _get_lut_f32(name):
    if name not in _LUT_CACHE_F32:
        lut = None
        try:
            import matplotlib.cm as cm
            lut = cm.get_cmap(name)(np.linspace(0,1,256)).astype(np.float32)
        except Exception: pass
        if lut is None:
            g = np.linspace(0,1,256,dtype=np.float32)
            lut = np.column_stack([g,g,g,np.ones(256,np.float32)])
        _LUT_CACHE_F32[name] = lut
    return _LUT_CACHE_F32[name]


def _get_lut_u8(name):
    """LUT (256,4) uint8 — cacheada. Evita f32 intermediario."""
    if name not in _LUT_CACHE_U8:
        f32 = _get_lut_f32(name)
        _LUT_CACHE_U8[name] = np.clip(f32 * 255, 0, 255).astype(np.uint8)
    return _LUT_CACHE_U8[name]

# Aliases for backward compat
_get_lut = _get_lut_f32


def _get_classification_lut():
    global _CLASS_LUT_CACHE
    if _CLASS_LUT_CACHE is not None: return _CLASS_LUT_CACHE
    import colorsys
    lut = np.full((256,3), 0.5, np.float32)
    for cid in range(256):
        lut[cid] = (LAS_CLASS_COLORS[cid] if cid in LAS_CLASS_COLORS
                    else colorsys.hsv_to_rgb((cid*137.508%360)/360, .90, .98))
    _CLASS_LUT_CACHE = lut
    return lut


def _get_classification_lut_u8():
    global _CLASS_LUT_U8_CACHE
    if _CLASS_LUT_U8_CACHE is not None: return _CLASS_LUT_U8_CACHE
    f32 = _get_classification_lut()
    _CLASS_LUT_U8_CACHE = np.clip(f32 * 255, 0, 255).astype(np.uint8)
    return _CLASS_LUT_U8_CACHE


# ── Enhancement ──────────────────────────────────────────────────────────────

def _enhance_colors(rgba, saturation=1.0, brightness=1.0, gamma=1.0):
    """Post-proceso de color en float32."""
    out = np.ascontiguousarray(rgba, dtype=np.float32)
    if _FAST:
        _c_enhance(out, gamma, saturation, brightness)
        return out
    rgb = np.clip(out[:,:3].copy(), 0, 1)
    if gamma != 1.0:
        rgb = np.power(rgb + 1e-7, 1.0/gamma)
    if saturation != 1.0:
        lum = (rgb[:,0]*0.299 + rgb[:,1]*0.587 + rgb[:,2]*0.114)[:,None]
        rgb = np.clip(lum + saturation*(rgb-lum), 0, 1)
    if brightness != 1.0:
        rgb = np.clip(rgb*brightness, 0, 1)
    out[:,:3] = rgb
    return np.ascontiguousarray(out)


def _enhance_to_u8(rgba_f32, saturation=1.0, brightness=1.0, gamma=1.0):
    """Enhance + convert to uint8 en un solo paso."""
    enhanced = _enhance_colors(rgba_f32, saturation, brightness, gamma)
    return (enhanced * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# RUTA DIRECTA UINT8 — la cadena principal de rendimiento
# ══════════════════════════════════════════════════════════════════════════════

# Debe coincidir con DELETED_LABEL en annotation/label_store.py — no se
# importa de ahí para no crear una dependencia circular render↔annotation;
# es un valor centinela estable (255, nunca una clase real), no una opción.
DELETED_LABEL = 255


def compute_colors_u8(xyz, attrs, mode, cmap="viridis",
                      z_range=None, annotation_labels=None,
                      annotation_lut_u8=None):
    """
    Retorna (N,4) uint8 DIRECTAMENTE — sin intermediario float32.

    Para modo Anotación (80% del uso):
      Antes: lut_f32[labels] → f32 → *255 → u8 → /255 → f32  (1362ms/15M)
      Ahora: lut_u8[labels] → u8                               (275ms/15M)
    """
    out = _compute_colors_u8_impl(xyz, attrs, mode, cmap, z_range,
                                  annotation_labels, annotation_lut_u8)
    # Puntos eliminados (ver DELETED_LABEL): invisibles en CUALQUIER modo de
    # color, no solo en "Anotación" — por eso este chequeo vive aquí, fuera
    # de las ramas por modo, y no dentro de la rama "Anotación" nada más.
    # alpha=0 evita tocar la geometría de vtkPoints (nada de reindexar/
    # reconstruir el mapper), coherente con que xyz puede ser un mmap.
    if annotation_labels is not None:
        deleted = annotation_labels == DELETED_LABEL
        if deleted.any():
            out[deleted, 3] = 0
    return out


def _compute_colors_u8_impl(xyz, attrs, mode, cmap="viridis",
                            z_range=None, annotation_labels=None,
                            annotation_lut_u8=None):
    n = len(xyz)
    if n == 0:
        return np.zeros((0,4), np.uint8)

    # ── Anotación ─────────────────────────────────────────────────────────────
    if mode == "Anotación":
        if annotation_labels is not None and annotation_lut_u8 is not None:
            return annotation_lut_u8[annotation_labels]  # DIRECTO: 275ms
        out = np.empty((n,4), np.uint8)
        out[:,:3] = 216; out[:,3] = 255  # gris opaco — ver build_annotation_lut_u8
        return out

    # ── RGB real ──────────────────────────────────────────────────────────────
    if mode == "RGB" and "rgb" in attrs:
        rgb = attrs["rgb"]  # ya uint8 (N,3)
        try:
            from core._fast import FC
            if FC.available:
                return FC._fc.rgb_to_rgba_u8(np.ascontiguousarray(rgb, np.uint8))
        except Exception: pass
        out = np.empty((n,4), np.uint8)
        out[:,:3] = rgb; out[:,3] = 255
        return out

    # ── Clasificación LAS ─────────────────────────────────────────────────────
    if mode == "Clasificación" and "classification" in attrs:
        lut = _get_classification_lut_u8()
        try:
            from core._fast import FC
            if FC.available:
                cls_u8 = np.ascontiguousarray(attrs["classification"], np.uint8)
                lut_full = np.empty((256, 4), np.uint8)
                lut_full[:, :3] = lut; lut_full[:, 3] = 255
                return FC.lut_lookup_u8(cls_u8, lut_full)
        except Exception: pass
        cls = attrs["classification"].astype(np.int32)
        out = np.empty((n,4), np.uint8)
        out[:,:3] = lut[np.clip(cls,0,255)]; out[:,3] = 255
        return out

    # ── Intensidad ────────────────────────────────────────────────────────────
    if mode in ("Intensidad","Intensidad Color") and "intensity" in attrs:
        nm  = "grays" if mode=="Intensidad" else "plasma"
        lut = _get_lut_u8(nm)
        try:
            from core._fast import FC
            if FC.available:
                iv = np.ascontiguousarray(attrs["intensity"], np.float32)
                return FC._fc.intensity_colors_u8(iv, lut)
        except Exception: pass
        idx = np.clip((attrs["intensity"] * 255).astype(np.int32), 0, 255)
        return lut[idx]

    # ── Confianza (post-inferencia) ────────────────────────────────────────────
    # "Future update" añadido en esta ronda: infer_cloud(..., return_confidence=
    # True) ya calculaba esto internamente (softmax de los logits promediados)
    # y lo descartaba, quedándose solo con el argmax — con el mismo cálculo ya
    # hecho, exponerlo como modo de color es casi gratis, y ayuda a saber dónde
    # revisar primero después de inferir (rojo = insegura, verde = segura).
    if mode == "Confianza" and "confidence" in attrs:
        lut = _get_lut_u8("RdYlGn")
        idx = np.clip((attrs["confidence"] * 255).astype(np.int32), 0, 255)
        return lut[idx]

    # ── Retorno ───────────────────────────────────────────────────────────────
    if mode == "Retorno" and "return_num" in attrs:
        v = attrs["return_num"].astype(np.float32)
        mx = v.max()
        if mx > 0: v /= mx
        lut = _get_lut_u8("tab10")
        idx = np.clip((v * 255).astype(np.int32), 0, 255)
        return lut[idx]

    # ── Densidad ──────────────────────────────────────────────────────────────
    if mode == "Densidad":
        xy = xyz[:,:2]; bb = xy.min(0)
        ext = np.where((xy.max(0)-bb)<1e-6, 1., xy.max(0)-bb)
        ci = np.clip(((xy-bb)/(ext/80)).astype(np.int32),0,79)
        keys = ci[:,0]*80+ci[:,1]
        _,inv,cnt = np.unique(keys,return_inverse=True,return_counts=True)
        d = np.log1p(cnt[inv].astype(np.float32))
        lut = _get_lut_u8("hot")
        idx = np.clip((d/(d.max()+1e-10) * 255).astype(np.int32), 0, 255)
        return lut[idx]

    # ── Color único ───────────────────────────────────────────────────────────
    if mode == "Color único":
        out = np.empty((n,4), np.uint8)
        out[:,0] = 89; out[:,1] = 204; out[:,2] = 255; out[:,3] = 255
        return out

    # ── Elevación (default) ───────────────────────────────────────────────────
    z = xyz[:,2]
    p1, p99 = z_range if z_range is not None else get_z_range(xyz)
    rng = max(p99-p1, 1e-6)
    vals = np.clip((z-p1)/rng, 0, 1)
    lut = _get_lut_u8(cmap)
    idx = np.clip((vals * 255).astype(np.int32), 0, 255)
    return lut[idx]


# ══════════════════════════════════════════════════════════════════════════════
# RUTA LEGACY FLOAT32 — para backward compat (refresh_colors, etc.)
# ══════════════════════════════════════════════════════════════════════════════

def compute_colors(xyz, attrs, mode, cmap="viridis",
                   z_range=None, annotation_labels=None,
                   annotation_lut=None):
    """Retorna (N,4) float32 — ruta legacy para refresh_colors."""
    n = len(xyz)
    if n == 0:
        return np.zeros((0,4), np.float32)

    if mode == "Anotación":
        if annotation_labels is not None and annotation_lut is not None:
            return _colors_annotation(annotation_labels, annotation_lut)
        out = np.empty((n,4), np.float32)
        out[:,:3] = 0.85; out[:,3] = 1.0
        return out

    if mode == "RGB" and "rgb" in attrs:
        rgb  = attrs["rgb"].astype(np.float32) / 255.
        rgba = np.empty((n,4), np.float32)
        rgba[:,:3] = rgb; rgba[:,3] = 1.0
        return _enhance_colors(rgba, 1.1, 1.15, 0.90)

    if mode == "Clasificación" and "classification" in attrs:
        cls = attrs["classification"].astype(np.int32)
        lut = _get_classification_lut()
        out = np.empty((n,4), np.float32)
        out[:,:3] = lut[np.clip(cls,0,255)]; out[:,3] = 1.0
        return _enhance_colors(out, 1.2, _VIS_BRIGHTNESS, _VIS_GAMMA)

    if mode in ("Intensidad","Intensidad Color") and "intensity" in attrs:
        nm  = "grays" if mode=="Intensidad" else "plasma"
        lut = _get_lut_f32(nm)
        idx = np.clip((attrs["intensity"]*255).astype(np.int32), 0, 255)
        raw = lut[idx]
        return _enhance_colors(raw, 1.0 if nm=="grays" else _VIS_SATURATION,
                               _VIS_BRIGHTNESS, _VIS_GAMMA)

    if mode == "Retorno" and "return_num" in attrs:
        v = attrs["return_num"].astype(np.float32)
        mx = v.max()
        if mx > 0: v /= mx
        lut = _get_lut_f32("tab10")
        idx = np.clip((v*255).astype(np.int32), 0, 255)
        return _enhance_colors(lut[idx], _VIS_SATURATION, _VIS_BRIGHTNESS, _VIS_GAMMA)

    if mode == "Densidad":
        xy = xyz[:,:2]; bb = xy.min(0)
        ext = np.where((xy.max(0)-bb)<1e-6, 1., xy.max(0)-bb)
        ci = np.clip(((xy-bb)/(ext/80)).astype(np.int32),0,79)
        keys = ci[:,0]*80+ci[:,1]
        _,inv,cnt = np.unique(keys,return_inverse=True,return_counts=True)
        d = np.log1p(cnt[inv].astype(np.float32))
        lut = _get_lut_f32("hot")
        idx = np.clip((d/(d.max()+1e-10)*255).astype(np.int32), 0, 255)
        return _enhance_colors(lut[idx], _VIS_SATURATION, _VIS_BRIGHTNESS, _VIS_GAMMA)

    if mode == "Color único":
        return np.tile([.35,.80,1.,1.],(n,1)).astype(np.float32)

    z = xyz[:,2]
    p1, p99 = z_range if z_range is not None else get_z_range(xyz)
    if _FAST:
        lut = _get_lut_f32(cmap)
        rgba = _c_z_colors(z.astype(np.float32), float(p1), float(p99), lut)
        return _enhance_colors(rgba, _VIS_SATURATION, _VIS_BRIGHTNESS, _VIS_GAMMA)
    rng  = max(p99-p1, 1e-6)
    vals = np.clip((z-p1)/rng, 0, 1).astype(np.float32)
    lut = _get_lut_f32(cmap)
    idx = np.clip((vals*255).astype(np.int32), 0, 255)
    return _enhance_colors(lut[idx], _VIS_SATURATION, _VIS_BRIGHTNESS, _VIS_GAMMA)


def _colors_annotation(labels, lut):
    if _FAST:
        rgba = _c_apply_lut(labels.astype(np.uint8), lut)
    else:
        clipped = np.clip(labels.astype(np.int32), 0, 255)
        rgba = lut[clipped].astype(np.float32)
    # Antes forzaba alpha=0.45 (translúcido) para sin-etiquetar aquí de
    # nuevo, deshaciendo el alpha=1.0 ya correcto que pone la LUT (ver
    # build_annotation_lut) — mismo bug de fondo que causaba el freeze de
    # rendering en modo Anotación (VTK renderiza TODO el actor translúcido
    # en cuanto CUALQUIER punto tiene alpha<1.0, y sin-etiquetar suele ser
    # la mayoría de la nube). Gris opaco, sin tocar alpha.
    mask0 = labels == 0
    if mask0.any():
        rgba[mask0, 0] = 0.85; rgba[mask0, 1] = 0.85
        rgba[mask0, 2] = 0.85; rgba[mask0, 3] = 1.0
    return rgba


def _apply_lut(vals, name):
    lut = _get_lut_f32(name)
    idx = np.clip((vals*255).astype(np.int32), 0, 255)
    return np.ascontiguousarray(lut[idx])


def build_annotation_lut(schema_list):
    """LUT (256,4) float32 desde el schema — para ruta legacy.

    Puntos sin etiquetar (clase 0): antes alpha=0.45 (semi-transparente)
    para verse "apagados" frente a las clases ya etiquetadas. Un punto
    con alpha<1.0 obliga a VTK a renderizar TODO el actor con blending
    translúcido en vez de opaco — mucho más lento por frame en nubes
    densas — y como los puntos sin etiquetar son casi siempre la
    mayoría de la nube mientras se está anotando, esto hacía lento
    justamente el modo "Anotación" frente a RGB (siempre alpha=1.0,
    siempre opaco). Ahora se ven "apagados" con un gris opaco
    (alpha=1.0) en vez de con transparencia — mismo efecto visual de
    "no etiquetado todavía", sin pagar el costo de renderizado
    translúcido. Los puntos ELIMINADOS (DELETED_LABEL) siguen usando
    alpha=0 — son la minoría real de casos, no la mayoría constante.
    """
    lut = np.full((256,4), [0.85,0.85,0.85,1.0], dtype=np.float32)
    for sc in schema_list:
        idx = int(sc.id)
        if 0 <= idx < DELETED_LABEL:   # 255 nunca es una clase real, ver DELETED_LABEL
            h = sc.color.lstrip("#")
            r,g,b = int(h[0:2],16)/255., int(h[2:4],16)/255., int(h[4:6],16)/255.
            lut[idx] = [r,g,b,1.0]
    lut[0] = [0.85,0.85,0.85,1.0]
    # Defensa en profundidad: compute_colors_u8() ya enmascara alpha=0 para
    # puntos eliminados en CUALQUIER modo, esto solo cubre por si algo usa
    # la LUT directamente sin pasar por ese wrapper.
    lut[DELETED_LABEL] = [0.0, 0.0, 0.0, 0.0]
    return lut


def build_annotation_lut_u8(schema_list):
    """LUT (256,4) uint8 — ruta directa para render pipeline.

    Ver docstring de `build_annotation_lut` — mismo cambio: sin
    etiquetar ahora es gris OPACO (alpha=255), no semi-transparente.
    """
    lut = np.full((256,4), [216,216,216,255], dtype=np.uint8)
    for sc in schema_list:
        idx = int(sc.id)
        if 0 <= idx < DELETED_LABEL:   # 255 nunca es una clase real, ver DELETED_LABEL
            h = sc.color.lstrip("#")
            r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            lut[idx] = [r,g,b,255]
    lut[0] = [216,216,216,255]
    lut[DELETED_LABEL] = [0, 0, 0, 0]
    return lut
