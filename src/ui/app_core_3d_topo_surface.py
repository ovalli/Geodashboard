# ======================================================
# src/ui/app_core_3d_topo_surface.py  (COMPLET)
# Surface engine:
# - Delaunay XY (SciPy si dispo)
# - filtres edge / quality / dihedral / components
# - (prêt) masque excavation par polygones XY
# ======================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ======================================================
# Config & debug
# ======================================================
@dataclass(frozen=True)
class SurfaceConfig:
    # Edge continuity (anti "ponts" / parasites)
    max_edge_factor: float = 10.0
    max_edge_min: float = 10.0

    # Quality filter (anti slivers / degenerates)
    enable_quality_filter: bool = True
    min_quality_rel: float = 0.12
    min_quality_abs: float = 1e-10

    # Dihedral filter (anti triangles "sharp")
    enable_dihedral_filter: bool = True
    min_dihedral_deg: float = 45.0
    dihedral_drop_mode: str = "weaker"  # "weaker" (reco) ou "both"

    # Components filter (drop tiny floating islands)
    drop_tiny_components: bool = True
    min_tris_per_component: int = 5

    # Excavation mask (holes)
    enable_excavation_mask: bool = True
    excavation_mode: str = "centroid"  # "centroid" ou "any_vertex"


@dataclass
class SurfaceDebug:
    used_scipy: bool = False
    n0: int = 0
    n_edge: int = 0
    n_qual: int = 0
    n_dihedral: int = 0
    n_comp: int = 0
    n_excav: int = 0


# ======================================================
# Public API
# ======================================================
def build_surface(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    config: SurfaceConfig,
    excavations_xy: Optional[List[List[Tuple[float, float]]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SurfaceDebug]:
    """
    Retourne (i,j,k, debug) pour Mesh3d.
    Si SciPy indisponible ou triangulation impossible => i/j/k vides.
    """
    dbg = SurfaceDebug()

    tri_i, tri_j, tri_k, used_scipy = _triangles_from_xy(xs, ys)
    dbg.used_scipy = bool(used_scipy)
    dbg.n0 = int(tri_i.size)

    if not used_scipy or tri_i.size == 0:
        # Pas de triangulation SciPy => laisse fallback côté caller (alphahull)
        return tri_i, tri_j, tri_k, dbg

    # 1) continuité (edges)
    tri_i, tri_j, tri_k = _filter_triangles_continuous(
        xs, ys, zs, tri_i, tri_j, tri_k,
        max_edge_factor=float(config.max_edge_factor),
        max_edge_min=float(config.max_edge_min),
    )
    dbg.n_edge = int(tri_i.size)

    # 2) qualité (slivers / degenerates)
    if config.enable_quality_filter and tri_i.size > 0:
        tri_i, tri_j, tri_k = _filter_triangles_by_quality(
            xs, ys, zs, tri_i, tri_j, tri_k,
            min_quality_rel=float(config.min_quality_rel),
            min_quality_abs=float(config.min_quality_abs),
        )
    dbg.n_qual = int(tri_i.size)

    # 3) dièdre (sharp)
    if config.enable_dihedral_filter and float(config.min_dihedral_deg) > 0.0 and tri_i.size > 0:
        tri_i, tri_j, tri_k = _filter_triangles_by_dihedral(
            xs, ys, zs, tri_i, tri_j, tri_k,
            min_dihedral_deg=float(config.min_dihedral_deg),
            drop_mode=str(config.dihedral_drop_mode),
        )
    dbg.n_dihedral = int(tri_i.size)

    # 4) drop composants trop petits
    if config.drop_tiny_components and tri_i.size > 0 and int(config.min_tris_per_component) > 1:
        tri_i, tri_j, tri_k = _drop_tiny_triangle_components(
            tri_i, tri_j, tri_k,
            min_tris=int(config.min_tris_per_component),
        )
    dbg.n_comp = int(tri_i.size)

    # 5) excavation mask (optionnel)
    if (
        config.enable_excavation_mask
        and excavations_xy
        and tri_i.size > 0
    ):
        tri_i, tri_j, tri_k = _mask_triangles_by_excavations_xy(
            xs, ys, tri_i, tri_j, tri_k,
            excavations_xy=excavations_xy,
            mode=str(config.excavation_mode),
        )
    dbg.n_excav = int(tri_i.size)

    return tri_i, tri_j, tri_k, dbg


# ======================================================
# Triangulation XY
# ======================================================
def _triangles_from_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    if len(x) < 3:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), False
    try:
        from scipy.spatial import Delaunay  # type: ignore
    except Exception:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), False

    try:
        tri = Delaunay(np.column_stack((x, y)))
        simp = tri.simplices
        if simp is None or len(simp) == 0:
            return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), False
        return simp[:, 0], simp[:, 1], simp[:, 2], True
    except Exception:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), False


# ======================================================
# Helpers
# ======================================================
def _local_scale_3d(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> float:
    n = int(xs.size)
    if n < 6:
        dx = float(xs.max() - xs.min())
        dy = float(ys.max() - ys.min())
        dz = float(zs.max() - zs.min())
        return max(1.0, 0.10 * float(np.sqrt(dx * dx + dy * dy + dz * dz)))

    pts = np.column_stack([xs, ys, zs])
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=6)  # inclut self
        s = float(np.median(d[:, 5]))
        return max(1e-6, s)
    except Exception:
        m = min(n, 250)
        idx = np.linspace(0, n - 1, m).astype(int)
        sub = pts[idx]
        svals: list[float] = []
        for p in sub:
            dd = np.linalg.norm(pts - p, axis=1)
            dd.sort()
            if dd.size >= 6:
                svals.append(float(dd[5]))
        if not svals:
            return 1.0
        return max(1e-6, float(np.median(svals)))


def _filter_triangles_continuous(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    k: np.ndarray,
    *,
    max_edge_factor: float,
    max_edge_min: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if i.size == 0:
        return i, j, k

    scale = _local_scale_3d(xs, ys, zs)
    thr = max(float(max_edge_min), float(max_edge_factor) * float(scale))

    xi, yi, zi = xs[i], ys[i], zs[i]
    xj, yj, zj = xs[j], ys[j], zs[j]
    xk, yk, zk = xs[k], ys[k], zs[k]

    eij = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
    ejk = np.sqrt((xj - xk) ** 2 + (yj - yk) ** 2 + (zj - zk) ** 2)
    eki = np.sqrt((xk - xi) ** 2 + (yk - yi) ** 2 + (zk - zi) ** 2)

    keep = (eij <= thr) & (ejk <= thr) & (eki <= thr)
    return i[keep], j[keep], k[keep]


def _tri_normals(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
    p0 = np.column_stack([xs[i], ys[i], zs[i]])
    p1 = np.column_stack([xs[j], ys[j], zs[j]])
    p2 = np.column_stack([xs[k], ys[k], zs[k]])
    u = p1 - p0
    v = p2 - p0
    n = np.cross(u, v)
    norm = np.linalg.norm(n, axis=1)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return n / norm[:, None]


def _tri_quality(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
    p0 = np.column_stack([xs[i], ys[i], zs[i]])
    p1 = np.column_stack([xs[j], ys[j], zs[j]])
    p2 = np.column_stack([xs[k], ys[k], zs[k]])
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p0 - p2, axis=1)
    per = a + b + c
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    return area / np.maximum(per * per, 1e-12)


def _filter_triangles_by_quality(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    k: np.ndarray,
    *,
    min_quality_rel: float,
    min_quality_abs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if i.size == 0:
        return i, j, k

    Q = _tri_quality(xs, ys, zs, i, j, k)
    q_med = float(np.median(Q)) if Q.size else 0.0
    thr = max(float(min_quality_abs), float(min_quality_rel) * q_med)

    keep = Q >= thr
    return i[keep], j[keep], k[keep]


def _filter_triangles_by_dihedral(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    k: np.ndarray,
    *,
    min_dihedral_deg: float,
    drop_mode: str = "weaker",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if i.size == 0:
        return i, j, k

    N = _tri_normals(xs, ys, zs, i, j, k)
    Q = _tri_quality(xs, ys, zs, i, j, k)

    from collections import defaultdict

    edge2tris: dict[tuple[int, int], list[int]] = defaultdict(list)
    T = int(i.size)
    tris = np.column_stack([i, j, k]).astype(int)

    for t in range(T):
        a, b, c = int(tris[t, 0]), int(tris[t, 1]), int(tris[t, 2])
        e1 = (a, b) if a < b else (b, a)
        e2 = (b, c) if b < c else (c, b)
        e3 = (c, a) if c < a else (a, c)
        edge2tris[e1].append(t)
        edge2tris[e2].append(t)
        edge2tris[e3].append(t)

    th = float(min_dihedral_deg)
    th = max(0.0, min(180.0, th))

    drop = np.zeros(T, dtype=bool)

    for _, ts in edge2tris.items():
        if len(ts) != 2:
            continue
        t1, t2 = ts[0], ts[1]

        c = float(np.dot(N[t1], N[t2]))
        c = max(-1.0, min(1.0, c))
        ang = float(np.degrees(np.arccos(c)))
        dihedral = 180.0 - ang

        if dihedral < th:
            if str(drop_mode).lower() == "both":
                drop[t1] = True
                drop[t2] = True
            else:
                if Q[t1] <= Q[t2]:
                    drop[t1] = True
                else:
                    drop[t2] = True

    keep = ~drop
    return i[keep], j[keep], k[keep]


def _drop_tiny_triangle_components(i: np.ndarray, j: np.ndarray, k: np.ndarray, *, min_tris: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if i.size == 0:
        return i, j, k

    from collections import defaultdict, deque

    m = int(i.size)
    tris = np.column_stack([i, j, k]).astype(int)

    v2t = defaultdict(list)
    for t_idx in range(m):
        a, b, c = int(tris[t_idx, 0]), int(tris[t_idx, 1]), int(tris[t_idx, 2])
        v2t[a].append(t_idx)
        v2t[b].append(t_idx)
        v2t[c].append(t_idx)

    visited = np.zeros(m, dtype=bool)
    keep_tri = np.zeros(m, dtype=bool)

    for t0 in range(m):
        if visited[t0]:
            continue
        q = deque([t0])
        comp: list[int] = []
        visited[t0] = True
        while q:
            t = q.popleft()
            comp.append(t)
            a, b, c = int(tris[t, 0]), int(tris[t, 1]), int(tris[t, 2])
            for v in (a, b, c):
                for nb in v2t[v]:
                    if not visited[nb]:
                        visited[nb] = True
                        q.append(nb)

        if len(comp) >= int(min_tris):
            keep_tri[np.array(comp, dtype=int)] = True

    return i[keep_tri], j[keep_tri], k[keep_tri]


# ======================================================
# Excavation mask (holes) — prêt (non branché UI ici)
# ======================================================
def _point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    # Ray casting (robuste enough pour polygones simples)
    inside = False
    n = len(poly)
    if n < 3:
        return False
    x0, y0 = poly[-1]
    for (x1, y1) in poly:
        # segment (x0,y0)->(x1,y1)
        cond = ((y1 > y) != (y0 > y))
        if cond:
            xinters = (x0 - x1) * (y - y1) / ((y0 - y1) if (y0 - y1) != 0 else 1e-18) + x1
            if x < xinters:
                inside = not inside
        x0, y0 = x1, y1
    return inside


def _mask_triangles_by_excavations_xy(
    xs: np.ndarray,
    ys: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    k: np.ndarray,
    *,
    excavations_xy: List[List[Tuple[float, float]]],
    mode: str = "centroid",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if i.size == 0:
        return i, j, k

    mode = str(mode).lower().strip()
    mode = "centroid" if mode not in {"centroid", "any_vertex"} else mode

    xi, yi = xs[i], ys[i]
    xj, yj = xs[j], ys[j]
    xk, yk = xs[k], ys[k]

    if mode == "any_vertex":
        # drop si au moins un sommet est dans un polygone
        drop = np.zeros(i.size, dtype=bool)
        for poly in excavations_xy:
            if len(poly) < 3:
                continue
            # vectorisé "light": on boucle triangles, c'est ok à ces tailles
            for t in range(i.size):
                if drop[t]:
                    continue
                if (
                    _point_in_poly(float(xi[t]), float(yi[t]), poly)
                    or _point_in_poly(float(xj[t]), float(yj[t]), poly)
                    or _point_in_poly(float(xk[t]), float(yk[t]), poly)
                ):
                    drop[t] = True
        keep = ~drop
        return i[keep], j[keep], k[keep]

    # centroid
    xc = (xi + xj + xk) / 3.0
    yc = (yi + yj + yk) / 3.0
    drop = np.zeros(i.size, dtype=bool)
    for poly in excavations_xy:
        if len(poly) < 3:
            continue
        for t in range(i.size):
            if drop[t]:
                continue
            if _point_in_poly(float(xc[t]), float(yc[t]), poly):
                drop[t] = True
    keep = ~drop
    return i[keep], j[keep], k[keep]
