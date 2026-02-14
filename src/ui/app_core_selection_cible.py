# ======================================================
# src/ui/app_core_selection_cible.py  (COMPLET)
# ✅ Micro-modif demandée : on stocke AUSSI la couleur de chaque zone/coupe
#    (sans changer le design ni le fonctionnement)
#
# ✅ MODIF (JSON-only) :
# - CoupesManager ne prend plus workbook_path => on ne lui passe plus jamais.
# - render_selection_cibles garde cst_workbook_path en param pour compat legacy,
#   mais il est ignoré pour le manager.
#
# ✅ PATCH IDENTIQUE à app_core_topo (BIEN appliqué) :
# - on lit LES POINTS depuis la source active (cst_workbook_path si valide),
#   sinon fallback Mesures Completes
# - cache busting avec `dummy` : mtime = stat().st_mtime + dummy
#
# ✅ FIX IMPORTANT (pour éviter le "pas à jour" côté sélection) :
# - on propage `mtime` dans list_targets_in_mesures / read_targets_timeseries
#   afin d'invalider aussi le cache LRU du reader.
# ======================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.io.coupes_manager import CoupesManager, Coupe
from src.pipeline.mesures_completes_reader import list_targets_in_mesures, read_targets_timeseries


# ------------------------------------------------------
# Component import (robuste)
# ------------------------------------------------------
def _import_selection_component():
    tries = [
        ("src.ui.selection_cibles_html", "selection_cibles_component"),
        ("src.ui.selection_cibles_component", "selection_cibles_component"),
    ]
    last: Exception | None = None
    for mod, fn in tries:
        try:
            m = __import__(mod, fromlist=[fn])
            return getattr(m, fn)
        except Exception as e:
            last = e
    raise ImportError("Impossible d'importer selection_cibles_component") from last


# ------------------------------------------------------
# Project paths
# ------------------------------------------------------
def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "data" / "common_data").exists():
            return p
        if (p / "app.py").exists():
            return p
    return here.parents[2]


def _common_data_dir() -> Path:
    d = _project_root() / "data" / "common_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_mesures_completes_xlsx(common_data_dir: Path) -> Optional[Path]:
    if not common_data_dir.exists():
        return None
    needles = ["mesures completes", "mesures complètes", "mesures complete", "mesures compl"]
    cands: list[Path] = []
    for p in common_data_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".xlsx" and any(n in p.stem.lower() for n in needles):
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def _resolve_points_workbook(cst_workbook_path: str | None) -> Optional[Path]:
    """
    ✅ PATCH (comme topo) :
    - si cst_workbook_path est fourni et pointe vers un .xlsx existant -> on l'utilise
    - sinon -> fallback Mesures Completes dans data/common_data
    """
    root = _project_root()

    if cst_workbook_path:
        p = Path(cst_workbook_path).expanduser()
        if not p.is_absolute():
            p = (root / p).resolve()
        else:
            p = p.resolve()

        if p.exists() and p.is_file() and p.suffix.lower() == ".xlsx":
            return p

    common = _common_data_dir()
    mp = _find_mesures_completes_xlsx(common)
    if mp and mp.exists():
        return mp

    return None


# ------------------------------------------------------
# Fit-to-canvas mapping
# ------------------------------------------------------
def _robust_bounds(vals: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> Tuple[float, float]:
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(v, q_low))
    hi = float(np.percentile(v, q_high))
    if hi - lo < 1e-12:
        lo = float(np.min(v))
        hi = float(np.max(v))
        if hi - lo < 1e-12:
            hi = lo + 1.0
    return lo, hi


def _make_mapper(points: pd.DataFrame, width: int, height: int):
    w = float(width)
    h = float(height)
    pad = 35.0
    inner_w = max(w - 2 * pad, 1.0)
    inner_h = max(h - 2 * pad, 1.0)

    xs = points["x"].to_numpy(dtype=float)
    ys = points["y"].to_numpy(dtype=float)

    bx0, bx1 = _robust_bounds(xs)
    by0, by1 = _robust_bounds(ys)

    span_x = max(bx1 - bx0, 1e-12)
    span_y = max(by1 - by0, 1e-12)

    s = min(inner_w / span_x, inner_h / span_y)

    content_w = span_x * s
    content_h = span_y * s
    off_x = (w - content_w) / 2.0
    off_y = (h - content_h) / 2.0

    def clamp(v: float, a: float, b: float) -> float:
        return a if v < a else (b if v > b else v)

    def to_px(x: float, y: float) -> Tuple[float, float]:
        cx = clamp(x, bx0, bx1)
        cy = clamp(y, by0, by1)
        px = off_x + (cx - bx0) * s
        py = off_y + (by1 - cy) * s
        return float(px), float(py)

    return to_px


def _last_xyz(df: pd.DataFrame) -> Optional[Tuple[float, float, float]]:
    if df is None or df.empty:
        return None
    if not {"x", "y"}.issubset(df.columns):
        return None
    sub = df.dropna(subset=["x", "y"], how="any")
    if sub.empty:
        return None
    last = sub.iloc[-1]
    x = float(last["x"])
    y = float(last["y"])
    z = float(last["z"]) if ("z" in sub.columns and pd.notna(last.get("z"))) else 0.0
    return x, y, z


@st.cache_data(show_spinner=False)
def _load_points_fit_cached(workbook_path: str, mtime: float, W: int, H: int) -> list[dict[str, Any]]:
    # ✅ `mtime` est volontairement dans la signature => clé de cache Streamlit
    # ✅ IMPORTANT : on propage aussi mtime au reader (sinon lru_cache peut garder l'ancien excel)
    names = list_targets_in_mesures(workbook_path, sheet_name=None, mtime=mtime)
    if not names:
        return []
    ts = read_targets_timeseries(workbook_path, names, sheet_name=None, mtime=mtime)

    rows: list[tuple[str, float, float, float]] = []
    for name, df in ts.items():
        xyz = _last_xyz(df)
        if xyz is None:
            continue
        x, y, z = xyz
        rows.append((str(name), x, y, z))

    if not rows:
        return []

    raw = pd.DataFrame(rows, columns=["name", "x", "y", "z"])
    to_px = _make_mapper(raw, W, H)

    out: list[dict[str, Any]] = []
    for _, r in raw.iterrows():
        px, py = to_px(float(r["x"]), float(r["y"]))
        out.append({"name": str(r["name"]), "px": px, "py": py, "z": float(r["z"])})
    return out


# ------------------------------------------------------
# Zone colors (inchangé côté design: même palette)
# ------------------------------------------------------
_ZONE_PALETTE = ["#146EFF", "#22C55E", "#F97316", "#A855F7", "#EF4444", "#06B6D4", "#EAB308", "#EC4899"]


def _zone_color(i: int) -> str:
    return _ZONE_PALETTE[i % len(_ZONE_PALETTE)]


# ------------------------------------------------------
# Snapshot -> JSON (avec couleur)
# ------------------------------------------------------
def _apply_snapshot(
    mgr: CoupesManager,
    coupes: list[Coupe],
    snapshot: list[dict[str, Any]],
    zone_color_by_ui: dict[int, str],
) -> int:
    by_ui: dict[int, Coupe] = {}
    for c in coupes:
        try:
            by_ui[int(getattr(c, "ui_idx", 0))] = c
        except Exception:
            continue

    updated = 0
    for s in snapshot:
        try:
            ui = int(s.get("idx"))
        except Exception:
            continue

        c = by_ui.get(ui)
        if c is None:
            continue

        alpha = s.get("alpha", None)
        targets = s.get("targets", None)

        angle_final = float(alpha) if alpha is not None else float(getattr(c, "angle_deg", 0.0) or 0.0)

        targets_final = [str(x).strip() for x in (targets or []) if str(x).strip()]
        targets_final = sorted(set(targets_final), key=lambda x: x.lower())

        # ✅ la couleur de zone telle qu'affichée dans le front
        color_final = str(zone_color_by_ui.get(ui, "") or "").strip()

        angle_changed = float(getattr(c, "angle_deg", 0.0) or 0.0) != float(angle_final)
        targets_changed = list(getattr(c, "targets", []) or []) != list(targets_final)

        prev_color = str(getattr(c, "color", "") or "").strip()
        color_changed = prev_color != color_final and color_final != ""

        if not angle_changed and not targets_changed and not color_changed:
            continue

        cnew = Coupe(
            name=getattr(c, "name", ""),
            angle_deg=angle_final,
            targets=targets_final,
            ui_idx=ui,
        )

        try:
            setattr(cnew, "color", color_final)
        except Exception:
            pass

        mgr.update_coupe(getattr(c, "name", ""), cnew)
        updated += 1

    return updated


# ======================================================
# Public entrypoint
# ======================================================
def render_selection_cibles(
    zone_name: str = "GLOBAL",
    cst_workbook_path: str | None = None,
    dummy: int = 0,
) -> None:
    W, H = 1200, 680

    # ✅ JSON-only : cst_workbook_path ignoré pour le manager
    _ = zone_name
    mgr = CoupesManager()
    coupes = mgr.list_coupes()
    if not coupes:
        st.warning("Aucune coupe dans le JSON.")
        return

    # ✅ Zones + map couleur
    zones: list[dict[str, Any]] = []
    zone_color_by_ui: dict[int, str] = {}

    for i, c in enumerate(coupes):
        ui = int(getattr(c, "ui_idx", i))
        zid = str(ui)
        col = _zone_color(i)

        zone_color_by_ui[ui] = col

        zones.append(
            {
                "id": zid,
                "name": getattr(c, "name", ""),
                "idx": ui,
                "color": col,
                "mem_quad": f"quad_{zid}",
            }
        )

    # ✅ PATCH “comme topo” : points = fichier source actif, pas auto-find systématique
    points_workbook = _resolve_points_workbook(cst_workbook_path)

    points: list[dict[str, Any]] = []
    mtime_used: float | None = None
    if points_workbook and points_workbook.exists():
        # ✅ PATCH cache busting identique topo
        mtime_used = float(points_workbook.stat().st_mtime) + float(dummy or 0)
        points = _load_points_fit_cached(str(points_workbook), mtime_used, W, H)

    data = {"w": W, "h": H, "zones": zones, "points": points}

    selection_cibles_component = _import_selection_component()
    ret = selection_cibles_component(data=data, key="selection_cibles_component_main")

    if isinstance(ret, dict) and ret.get("type") == "front_button":
        snapshot = ret.get("snapshot")
        if not isinstance(snapshot, list):
            st.error("Le front n'a pas envoyé le snapshot (idx/name/alpha/targets).")
            return

        try:
            updated = _apply_snapshot(mgr, coupes, snapshot, zone_color_by_ui)
            if updated == 0:
                st.toast("Aucune modification à écrire.", icon="ℹ️")
            else:
                st.toast(f"JSON mis à jour : {updated} coupe(s) ✅", icon="✅")
                st.rerun()
        except Exception as e:
            st.error(f"Erreur écriture JSON: {type(e).__name__}: {e}")

    with st.expander("Debug", expanded=False):
        st.write(
            {
                "ret": ret,
                "cst_workbook_path": cst_workbook_path,
                "points_workbook_used": str(points_workbook) if points_workbook else None,
                "mtime_used": mtime_used,
                "dummy": dummy,
                "nb_points": len(points),
                "nb_coupes": len(coupes),
            }
        )
