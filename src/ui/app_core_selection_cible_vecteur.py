from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Iterator, Dict

import math
import numpy as np


# ======================================================
# Data models
# ======================================================
@dataclass(frozen=True)
class PointXYZ:
    name: str
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class Vector2D:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def dx(self) -> float:
        return self.x1 - self.x0

    @property
    def dy(self) -> float:
        return self.y1 - self.y0

    @property
    def length(self) -> float:
        return float(math.hypot(self.dx, self.dy))


@dataclass(frozen=True)
class VectorResult:
    vec: Vector2D
    alpha_signed_deg: float
    base_point: PointXYZ
    centroid_xy: Tuple[float, float]
    pca_dir: Tuple[float, float]
    used_length: float


@dataclass(frozen=True)
class VectorState:
    deleted: bool = False
    locked: bool = False
    custom: bool = False
    x0: Optional[float] = None
    y0: Optional[float] = None
    x1: Optional[float] = None
    y1: Optional[float] = None

    @staticmethod
    def from_any(v: Any) -> "VectorState":
        if isinstance(v, VectorState):
            return v
        if not isinstance(v, dict):
            return VectorState()

        def b(k: str, default: bool = False) -> bool:
            try:
                return bool(v.get(k, default))
            except Exception:
                return default

        def f(k: str) -> Optional[float]:
            x = v.get(k, None)
            if x is None:
                return None
            xf = _to_float(x, default=float("nan"))
            return None if not np.isfinite(xf) else float(xf)

        return VectorState(
            deleted=b("deleted", False),
            locked=b("locked", False),
            custom=b("custom", False),
            x0=f("x0"),
            y0=f("y0"),
            x1=f("x1"),
            y1=f("y1"),
        )

    def has_coords(self) -> bool:
        return (
            self.x0 is not None and self.y0 is not None and self.x1 is not None and self.y1 is not None
            and np.isfinite(self.x0) and np.isfinite(self.y0) and np.isfinite(self.x1) and np.isfinite(self.y1)
        )


# ======================================================
# Numeric helpers
# ======================================================
def _signed_deg(d: float) -> float:
    x = float(d)
    while x <= -180.0:
        x += 360.0
    while x > 180.0:
        x -= 360.0
    return x


def alpha_from_vector(dx: float, dy: float) -> float:
    # alpha in degrees, signed (-180, +180]
    return _signed_deg(math.degrees(math.atan2(dy, dx)))


def _dir_from_alpha(alpha_deg: float) -> Tuple[float, float]:
    r = math.radians(float(alpha_deg))
    return (float(math.cos(r)), float(math.sin(r)))


def _is_finite(*vals: float) -> bool:
    return all(np.isfinite(v) for v in vals)


def _circular_mean_deg(a_deg: np.ndarray) -> float:
    # mean of angles in degrees, return (-180, 180]
    a = np.deg2rad(a_deg.astype(float))
    s = float(np.sin(a).mean())
    c = float(np.cos(a).mean())
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return 0.0
    return _signed_deg(math.degrees(math.atan2(s, c)))


def _circular_median_deg(angles_deg: Sequence[float]) -> float:
    # robust median by minimizing circular absolute deviation (bruteforce over samples)
    if not angles_deg:
        return 0.0
    a = np.array([float(x) for x in angles_deg if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return 0.0
    # candidates: input angles
    best = float(a[0])
    best_cost = float("inf")
    for cand in a:
        d = np.array([_signed_deg(x - cand) for x in a], dtype=float)
        cost = float(np.abs(d).sum())
        if cost < best_cost:
            best_cost = cost
            best = float(cand)
    return _signed_deg(best)


def _centroid_xy(points: Sequence[PointXYZ]) -> Tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    xs = np.array([p.x for p in points], dtype=float)
    ys = np.array([p.y for p in points], dtype=float)
    return (float(xs.mean()), float(ys.mean()))


def _choose_base_point(points: Sequence[PointXYZ]) -> PointXYZ:
    # base = lowest z (then stable by name)
    if not points:
        return PointXYZ(name="base", x=0.0, y=0.0, z=0.0)
    return min(points, key=lambda p: (p.z, p.name))


def _lowest_points(points: Sequence[PointXYZ], *, frac: float = 0.30, min_pts: int = 12) -> Sequence[PointXYZ]:
    if not points:
        return []
    pts = [p for p in points if np.isfinite(p.z)]
    if not pts:
        return list(points)
    pts_sorted = sorted(pts, key=lambda p: p.z)
    n = max(int(len(pts_sorted) * float(frac)), int(min_pts))
    n = min(n, len(pts_sorted))
    return pts_sorted[:n]


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _iter_points_xyz(points: Iterable[Any]) -> Iterator[PointXYZ]:
    for p in points:
        if isinstance(p, PointXYZ):
            yield p
            continue
        if isinstance(p, dict):
            name = str(p.get("name", ""))
            x = _to_float(p.get("x", float("nan")), default=float("nan"))
            y = _to_float(p.get("y", float("nan")), default=float("nan"))
            z = _to_float(p.get("z", 0.0), default=0.0)
            if _is_finite(x, y, z):
                yield PointXYZ(name=name, x=float(x), y=float(y), z=float(z))
            continue

        # tuple-like
        try:
            name = str(p[0])
            x = _to_float(p[1], default=float("nan"))
            y = _to_float(p[2], default=float("nan"))
            z = _to_float(p[3] if len(p) > 3 else 0.0, default=0.0)
            if _is_finite(x, y, z):
                yield PointXYZ(name=name, x=float(x), y=float(y), z=float(z))
        except Exception:
            continue


def _diagnose_cloud(pts: Sequence[PointXYZ]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(pts)}
    if not pts:
        return out
    xs = np.array([p.x for p in pts], dtype=float)
    ys = np.array([p.y for p in pts], dtype=float)
    zs = np.array([p.z for p in pts], dtype=float)
    out.update(
        x_min=float(xs.min()), x_max=float(xs.max()),
        y_min=float(ys.min()), y_max=float(ys.max()),
        z_min=float(zs.min()), z_max=float(zs.max()),
        x_rng=float(xs.max() - xs.min()),
        y_rng=float(ys.max() - ys.min()),
        z_rng=float(zs.max() - zs.min()),
    )
    return out


def _auto_z_scale(xy: np.ndarray, z: np.ndarray) -> float:
    # heuristic: scale z to have comparable influence to xy (avoid vertical dominance)
    if xy.size == 0 or z.size == 0:
        return 1.0
    xr = float(xy[:, 0].max() - xy[:, 0].min())
    yr = float(xy[:, 1].max() - xy[:, 1].min())
    zr = float(z.max() - z.min())
    denom = max(zr, 1e-9)
    num = max((xr + yr) * 0.5, 1e-9)
    return float(num / denom)


def _knn_indices_xy(xy: np.ndarray, k: int) -> list[np.ndarray]:
    n = int(xy.shape[0])
    k = int(k)
    if n <= 1 or k <= 0:
        return [np.array([i], dtype=int) for i in range(n)]
    k = min(k, n - 1)
    out: list[np.ndarray] = []
    for i in range(n):
        d = xy - xy[i]
        dd = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]
        idx = np.argsort(dd)
        nn = idx[1 : 1 + k]
        out.append(nn.astype(int))
    return out


def _fit_plane_normal_svd(P: np.ndarray) -> Optional[np.ndarray]:
    # P: (m,3)
    if P.ndim != 2 or P.shape[0] < 3 or P.shape[1] != 3:
        return None
    C = P.mean(axis=0, keepdims=True)
    Q = P - C
    try:
        _, _, vh = np.linalg.svd(Q, full_matrices=False)
        n = vh[-1, :]
        nn = float(np.linalg.norm(n))
        if nn < 1e-12:
            return None
        return n / nn
    except Exception:
        return None


def _tilt_deg_from_normal(n_unit: np.ndarray) -> float:
    # tilt relative to Z axis (0 = horizontal plane)
    nz = float(n_unit[2])
    nz = max(-1.0, min(1.0, nz))
    tilt = math.degrees(math.acos(abs(nz)))
    return float(tilt)


def _alpha_from_normal_xy(n_unit: np.ndarray, add_180: bool) -> Optional[float]:
    # projection of normal on XY => direction
    nx = float(n_unit[0])
    ny = float(n_unit[1])
    L = math.hypot(nx, ny)
    if L < 1e-12:
        return None
    alpha = alpha_from_vector(nx, ny)
    if add_180:
        alpha = _signed_deg(alpha + 180.0)
    return float(alpha)


def _circular_hist_mode_deg(angles_deg: np.ndarray, *, bin_deg: float = 5.0) -> float:
    a = np.array([_signed_deg(float(x)) for x in angles_deg if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return 0.0
    binw = float(bin_deg)
    nb = max(int(360.0 / binw), 1)
    # shift to [0,360)
    a0 = (a + 180.0) % 360.0
    hist, edges = np.histogram(a0, bins=nb, range=(0.0, 360.0))
    j = int(np.argmax(hist))
    center = float((edges[j] + edges[j + 1]) * 0.5)
    # back to (-180, 180]
    return _signed_deg(center - 180.0)


def _select_cluster_around_mode(angles: Sequence[float], mode_deg: float, half_width_deg: float) -> list[float]:
    out: list[float] = []
    for a in angles:
        d = _signed_deg(float(a) - float(mode_deg))
        if abs(d) <= float(half_width_deg):
            out.append(float(a))
    return out


def _fallback_alpha_from_xy_line(pts: Sequence[PointXYZ], *, add_180: bool) -> Optional[float]:
    if len(pts) < 2:
        return None
    xy = np.array([[p.x, p.y] for p in pts], dtype=float)
    xy = xy - xy.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(xy, full_matrices=False)
        d = vh[0, :]
        alpha = alpha_from_vector(float(d[0]), float(d[1]))
        if add_180:
            alpha = _signed_deg(alpha + 180.0)
        return float(alpha)
    except Exception:
        return None


def _compute_alpha_from_points_robust(
    pts_all: Sequence[PointXYZ],
    *,
    add_180: bool,
    k_nn: Optional[int] = None,
) -> Tuple[Optional[float], Tuple[float, float], Dict[str, Any]]:
    # returns alpha, (dx,dy) (pca), debug dict
    dbg: Dict[str, Any] = {}
    if len(pts_all) < 3:
        return None, (0.0, 0.0), {"reason": "not_enough_points"}

    xy = np.array([[p.x, p.y] for p in pts_all], dtype=float)
    z = np.array([p.z for p in pts_all], dtype=float)

    # try local plane normals and take robust consensus on alpha
    if k_nn is None:
        k_nn = max(6, min(18, int(len(pts_all) * 0.12)))

    knn = _knn_indices_xy(xy, int(k_nn))
    zscale = _auto_z_scale(xy, z)
    dbg["k_nn"] = int(k_nn)
    dbg["zscale"] = float(zscale)

    alphas: list[float] = []
    tilts: list[float] = []
    for i, nn in enumerate(knn):
        if nn.size < 2:
            continue
        idx = np.concatenate([np.array([i], dtype=int), nn.astype(int)])
        P = np.stack([xy[idx, 0], xy[idx, 1], z[idx] * zscale], axis=1)
        n = _fit_plane_normal_svd(P)
        if n is None:
            continue
        tilt = _tilt_deg_from_normal(n)
        a = _alpha_from_normal_xy(n, add_180=add_180)
        if a is None:
            continue
        alphas.append(float(a))
        tilts.append(float(tilt))

    dbg["n_local_normals"] = int(len(alphas))
    if len(alphas) < 6:
        # fallback to xy line pca
        a2 = _fallback_alpha_from_xy_line(pts_all, add_180=add_180)
        if a2 is None:
            return None, (0.0, 0.0), {**dbg, "reason": "fallback_xy_failed"}
        # PCA direction for debug
        try:
            xy0 = xy - xy.mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(xy0, full_matrices=False)
            d = vh[0, :]
            return float(a2), (float(d[0]), float(d[1])), {**dbg, "reason": "fallback_xy_pca"}
        except Exception:
            return float(a2), (1.0, 0.0), {**dbg, "reason": "fallback_xy_pca_exception"}

    a_arr = np.array(alphas, dtype=float)

    # robust consensus: mode + cluster then median
    mode = _circular_hist_mode_deg(a_arr, bin_deg=5.0)
    cluster = _select_cluster_around_mode(a_arr.tolist(), mode, half_width_deg=18.0)
    dbg["mode_deg"] = float(mode)
    dbg["cluster_n"] = int(len(cluster))

    if len(cluster) < 4:
        alpha = _circular_mean_deg(a_arr)
        dbg["agg"] = "circular_mean"
    else:
        alpha = _circular_median_deg(cluster)
        dbg["agg"] = "circular_median_cluster"

    # PCA dir (xy) for reference only
    try:
        xy0 = xy - xy.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(xy0, full_matrices=False)
        d = vh[0, :]
        dxy = (float(d[0]), float(d[1]))
    except Exception:
        dxy = (1.0, 0.0)

    return float(alpha), dxy, dbg


def compute_default_vector(
    points: Iterable[PointXYZ],
    *,
    length: float = 170.0,
    low_frac: float = 0.30,
    low_min_pts: int = 12,
    add_180: bool = True,
    k_nn: Optional[int] = None,
    **_: Any,
) -> Optional[VectorResult]:
    pts_all = [p for p in points if _is_finite(p.x, p.y, p.z)]
    if len(pts_all) < 3:
        return None

    pts_low = list(_lowest_points(pts_all, frac=low_frac, min_pts=low_min_pts))
    if len(pts_low) < 1:
        pts_low = pts_all

    base = _choose_base_point(pts_low)
    cx, cy = _centroid_xy(pts_all)

    alpha, dxy, _dbg = _compute_alpha_from_points_robust(
        pts_all,
        add_180=add_180,
        k_nn=k_nn,
    )

    if alpha is None:
        a2 = _fallback_alpha_from_xy_line(pts_all, add_180=add_180)
        if a2 is None:
            alpha, dxy = 0.0, (1.0, 0.0)
        else:
            alpha = float(a2)
            dxy = _dir_from_alpha(alpha)

    ux, uy = _dir_from_alpha(alpha)
    L = float(length)

    x0, y0 = float(base.x), float(base.y)
    x1, y1 = x0 + ux * L, y0 + uy * L

    return VectorResult(
        vec=Vector2D(x0=x0, y0=y0, x1=x1, y1=y1),
        alpha_signed_deg=float(alpha),
        base_point=base,
        centroid_xy=(float(cx), float(cy)),
        pca_dir=(float(dxy[0]), float(dxy[1])),
        used_length=L,
    )


def resolve_vector(
    points: Iterable[PointXYZ],
    state: Any = None,
    *,
    length: float = 170.0,
    low_frac: float = 0.30,
    low_min_pts: int = 12,
    add_180: bool = True,
    k_nn: Optional[int] = None,
) -> Optional[VectorResult]:
    st = VectorState.from_any(state)
    if st.deleted:
        return None

    pts = [p for p in points if _is_finite(p.x, p.y, p.z)]
    if len(pts) < 3:
        return None

    if st.custom and st.has_coords():
        x0, y0, x1, y1 = float(st.x0), float(st.y0), float(st.x1), float(st.y1)
        dx, dy = x1 - x0, y1 - y0
        L = float(math.hypot(dx, dy))
        if L < 1e-9:
            return compute_default_vector(
                pts,
                length=length,
                low_frac=low_frac,
                low_min_pts=low_min_pts,
                add_180=add_180,
                k_nn=k_nn,
            )

        base = min(pts, key=lambda p: p.z)
        cx, cy = _centroid_xy(pts)
        alpha = alpha_from_vector(dx, dy)

        return VectorResult(
            vec=Vector2D(x0=x0, y0=y0, x1=x1, y1=y1),
            alpha_signed_deg=float(alpha),
            base_point=base,
            centroid_xy=(cx, cy),
            pca_dir=(0.0, 0.0),
            used_length=L,
        )

    return compute_default_vector(
        pts,
        length=length,
        low_frac=low_frac,
        low_min_pts=low_min_pts,
        add_180=add_180,
        k_nn=k_nn,
    )


def compute_vector_backend(
    points: Iterable[Any],
    state: Any = None,
    *,
    length: float = 170.0,
    min_points: int = 3,
    **kwargs: Any,
) -> Optional[VectorResult]:
    pts = list(_iter_points_xyz(points))
    if len(pts) < int(min_points):
        return None
    return resolve_vector(pts, state=state, length=length, **kwargs)


def compute_alpha_backend(
    points: Iterable[Any],
    state: Any = None,
    *,
    length: float = 170.0,
    min_points: int = 3,
    **kwargs: Any,
) -> Optional[float]:
    vr = compute_vector_backend(points, state=state, length=length, min_points=min_points, **kwargs)
    if vr is None:
        return None
    return float(vr.alpha_signed_deg)


# ------------------------------------------------------
# Debug helper (n'affecte pas l'API existante)
# ------------------------------------------------------
def compute_alpha_backend_debug(
    points: Iterable[Any],
    state: Any = None,
    *,
    length: float = 170.0,
    min_points: int = 3,
    add_180: bool = True,
    k_nn: Optional[int] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    À appeler UNIQUEMENT pour diagnostiquer:
      alpha, debug_dict
    """
    pts = list(_iter_points_xyz(points))
    diag = _diagnose_cloud(pts)
    if len(pts) < int(min_points):
        diag["reason"] = "min_points_not_reached"
        return None, diag

    st = VectorState.from_any(state)
    diag["state_deleted"] = bool(st.deleted)
    diag["state_custom"] = bool(st.custom)
    diag["state_has_coords"] = bool(st.has_coords())
    if st.deleted:
        diag["reason"] = "deleted"
        return None, diag

    alpha, _dxy, dbg = _compute_alpha_from_points_robust(
        pts,
        add_180=add_180,
        k_nn=k_nn,
    )
    diag["algo"] = dbg
    if alpha is None:
        diag["fallback_xy_pca"] = True
        alpha2 = _fallback_alpha_from_xy_line(pts, add_180=add_180)
        return alpha2, diag
    diag["fallback_xy_pca"] = False
    return alpha, diag


def debug_points_payload(points: Iterable[Any], title: str = "DEBUG POINTS") -> dict:
    pts = list(_iter_points_xyz(points))
    out = {"title": title, "n": len(pts)}

    if not pts:
        out["reason"] = "EMPTY after parsing"
        return out

    xs = np.array([p.x for p in pts], dtype=float)
    ys = np.array([p.y for p in pts], dtype=float)
    zs = np.array([p.z for p in pts], dtype=float)

    out["x_min"] = float(xs.min())
    out["x_max"] = float(xs.max())
    out["y_min"] = float(ys.min())
    out["y_max"] = float(ys.max())
    out["z_min"] = float(zs.min())
    out["z_max"] = float(zs.max())
    return out
