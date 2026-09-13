"""
annotation/plane_fit.py — Ajuste de plano por RANSAC (detección de
primitivas planas: techos, paredes, suelos locales, etc.)

Mismo principio que CloudCompare "RANSAC Shape Detection" (modo plano):
sortear tríos de puntos al azar, contar cuántos puntos del vecindario
caen cerca del plano que definen (inliers), quedarse con el mejor trío
tras N intentos, y refinar el plano final con mínimos cuadrados (SVD)
sobre sus inliers. Mucho más rápido que ajustar normales por punto y
sirve para pre-clasificar superficies planas grandes de un solo clic en
vez de pintarlas a mano punto por punto.

PlaneFitTool: Ctrl+clic con radio configurable — encierra un vecindario
esférico (mismo índice espacial rápido que SphereSelectTool/BrushTool),
ajusta el plano dominante ahí dentro, y etiqueta SOLO los puntos que
resultan inliers del plano (los que no son planos, dentro del mismo
radio, quedan sin tocar) — a diferencia de SphereSelectTool, que
etiqueta la esfera completa sin distinguir geometría.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from annotation.tools import BaseTool


def ransac_plane(xyz: np.ndarray, distance_threshold: float = 0.05,
                 n_iterations: int = 200, rng=None
                 ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Ajusta el plano dominante de xyz vía RANSAC.

    Retorna (inlier_mask, normal, d) donde el plano es normal·p + d = 0,
    o None si xyz tiene menos de 3 puntos o no converge ningún trío
    válido (puntos degenerados/colineales en todos los intentos).
    """
    n = len(xyz)
    if n < 3:
        return None
    xyz64 = np.ascontiguousarray(xyz, dtype=np.float64)
    rng = rng if rng is not None else np.random.default_rng()

    best_count = -1
    best_inliers = None

    # Sortear todos los tríos de una vez (vectorizado) — más rápido que
    # un np.random.choice por iteración dentro del bucle Python.
    triplets = rng.integers(0, n, size=(n_iterations, 3))

    for i in range(n_iterations):
        i0, i1, i2 = triplets[i]
        if i0 == i1 or i1 == i2 or i0 == i2:
            continue
        p0, p1, p2 = xyz64[i0], xyz64[i1], xyz64[i2]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue   # puntos colineales — trío degenerado
        normal = normal / norm
        d = -np.dot(normal, p0)
        dist = np.abs(xyz64 @ normal + d)
        count = int((dist <= distance_threshold).sum())
        if count > best_count:
            best_count = count
            best_inliers = dist <= distance_threshold

    if best_inliers is None or best_count < 3:
        return None

    # Refinar el plano final con mínimos cuadrados (SVD) sobre los inliers
    # del mejor trío — más preciso que quedarse con el plano de 3 puntos.
    pts = xyz64[best_inliers]
    centroid = pts.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
        normal = vh[-1]
    except np.linalg.LinAlgError:
        # Fallback: quedarse con el plano del mejor trío sin refinar
        return best_inliers, None, None
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return best_inliers, None, None
    normal = normal / norm
    d = -float(np.dot(normal, centroid))
    dist = np.abs(xyz64 @ normal + d)
    inliers = dist <= distance_threshold
    return inliers, normal.astype(np.float32), d


class PlaneFitTool(BaseTool):
    """
    Ctrl+clic: ajusta el plano dominante dentro de un radio esférico y
    etiqueta SOLO los puntos que resultan planos (inliers) con la clase
    activa — los puntos no-planos del mismo radio quedan sin tocar.
    Útil para techos, paredes, taludes, suelo local, etc.
    """
    name    = "Ajustar plano"
    key     = "P"
    icon    = "plane"
    tooltip = "Ctrl+clic → ajusta el plano dominante en el radio y etiqueta solo los puntos planos"

    def __init__(self):
        super().__init__()
        self.radius_m:            float = 5.0
        self.distance_threshold_m: float = 0.08
        self.n_iterations:        int   = 200
        self.last_normal: Optional[np.ndarray] = None

    def activate(self, canvas) -> None:
        super().activate(canvas)
        # Mensaje inicial en consola — mientras no haya un lugar más
        # visible en la UI para instrucciones por herramienta, esto al
        # menos deja claro el flujo la primera vez que se activa.
        print(f"[PlaneFit] Ctrl+clic para ajustar el plano dominante dentro de "
              f"un radio de {self.radius_m:.1f}m (círculo naranja = vista previa "
              f"del radio). Solo se etiquetan los puntos que resultan PLANOS "
              f"dentro de ese radio — el resto queda intacto.")

    def deactivate(self) -> None:
        self._clear_plane_preview()
        super().deactivate()

    def on_mouse_press(self, event):
        if event.button != 1: return
        p = self._world_pos(event.pos)
        if p is None: return
        self._fit_and_apply(p)

    def _fit_and_apply(self, center: np.ndarray) -> None:
        c = self.canvas
        if c is None or c.pc is None: return

        idx = self._octree_sphere_query(center, self.radius_m)
        if idx is None or len(idx) < 3:
            idx = self._sphere_in_visible(center, self.radius_m)
        if idx is None or len(idx) < 3:
            print(f"[PlaneFit] Muy pocos puntos dentro del radio "
                  f"({self.radius_m:.1f}m) para ajustar un plano — "
                  f"sube el radio (panel derecho) o clica en una zona más densa.")
            return

        xyz = c.pc.xyz[idx].astype(np.float32)
        result = ransac_plane(xyz, distance_threshold=self.distance_threshold_m,
                              n_iterations=self.n_iterations)
        if result is None:
            print(f"[PlaneFit] No se encontró un plano dominante con el umbral "
                  f"actual ({self.distance_threshold_m:.2f}m) — prueba a subirlo "
                  f"si la superficie es rugosa, o el radio si el clic cayó en "
                  f"una zona sin geometría plana.")
            return
        inliers, normal, d = result
        self.last_normal = normal
        n_inliers = int(inliers.sum())
        pct = 100.0 * n_inliers / len(idx)
        if normal is not None:
            print(f"[PlaneFit] Plano ajustado: {n_inliers}/{len(idx)} pts "
                  f"({pct:.0f}%) — normal=({normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f}) "
                  f"— pintados con la clase activa, el resto del radio queda intacto.")
        else:
            print(f"[PlaneFit] Plano ajustado: {n_inliers}/{len(idx)} pts ({pct:.0f}%)")
        global_inliers = np.asarray(idx, dtype=np.int64)[inliers]

        # Feedback visual: dibujar el contorno del plano detectado (un
        # cuadrado orientado según la normal, del tamaño real de los
        # inliers) — para que se vea EXACTAMENTE qué se consideró "plano",
        # no solo confiar en el mensaje de consola.
        if normal is not None and len(global_inliers) >= 3:
            self._draw_plane_preview(xyz[inliers], normal)

        if len(global_inliers) > 0:
            self._apply_selection(global_inliers.astype(np.int32))

    def _draw_plane_preview(self, inlier_xyz: np.ndarray, normal: np.ndarray) -> None:
        c = self.canvas
        if c is None: return
        try:
            centroid = inlier_xyz.mean(axis=0)
            # Dos vectores ortogonales dentro del plano (cualquier base
            # perpendicular a la normal sirve para dibujar un cuadrado).
            arbitrary = np.array([1.0, 0.0, 0.0], np.float32)
            if abs(float(np.dot(arbitrary, normal))) > 0.9:
                arbitrary = np.array([0.0, 1.0, 0.0], np.float32)
            u = np.cross(normal, arbitrary); u /= (np.linalg.norm(u) + 1e-9)
            v = np.cross(normal, u); v /= (np.linalg.norm(v) + 1e-9)
            # Tamaño del cuadrado: la extensión real de los inliers proyectada
            # sobre u/v, no un tamaño fijo arbitrario.
            rel = inlier_xyz - centroid
            eu = float(np.abs(rel @ u).max()) if len(rel) else 1.0
            ev = float(np.abs(rel @ v).max()) if len(rel) else 1.0
            eu = max(eu, 0.3); ev = max(ev, 0.3)
            corners = np.array([
                centroid + eu*u + ev*v, centroid - eu*u + ev*v,
                centroid - eu*u - ev*v, centroid + eu*u - ev*v,
                centroid + eu*u + ev*v,
            ], np.float32)
            c._mline.set_data(corners, color=(0.2, 0.85, 1.0, 0.9), width=2.5)
            c._mline.visible = True
            c.update()
        except Exception:
            pass

    def _clear_plane_preview(self) -> None:
        c = self.canvas
        if c is None: return
        try:
            c._mline.visible = False
            c.update()
        except Exception:
            pass
