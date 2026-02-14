from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ======================================================
# Utils: locate project + Mesures Completes
# ======================================================
def _project_root() -> Path:
    """
    Robust project root finder:
    - Walk upward until we find data/common_data OR app.py
    - Fallback to previous behavior if not found
    """
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "data" / "common_data").exists():
            return p
        if (p / "app.py").exists():
            return p
    return here.parents[2]


def _find_mesures_completes_xlsx(common_data_dir: Path) -> Optional[Path]:
    if not common_data_dir.exists():
        return None

    needles = ["mesures completes", "mesures complètes", "mesures complete", "mesures compl"]
    candidates: List[Path] = []
    for p in common_data_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".xlsx":
            stem = p.stem.lower()
            if any(n in stem for n in needles):
                candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _detect_data_sheet(workbook_path: str) -> str:
    """
    ✅ NOUVEAU COMPORTEMENT :
    On lit TOUJOURS la première feuille du classeur, quel que soit son nom.
    """
    xls = pd.ExcelFile(workbook_path, engine="openpyxl")
    if not xls.sheet_names:
        raise ValueError("Classeur Excel sans onglets.")
    return xls.sheet_names[0]


# ======================================================
# Parsing conventions
# - Row 1 = names on columns 2,5,8,... (triplets)
# - Col 1 = date
# - Triplet: X=j, Y=j+1, Z=j+2
# ======================================================
def _clean_name(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip()
    return s if s else None


def _fmt_date(dt: Optional[pd.Timestamp]) -> str:
    if dt is None or pd.isna(dt):
        return "—"
    return dt.strftime("%d/%m/%Y")


def _fmt_delta_mm(v: float) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v * 1000.0:.1f} mm"


def _fmt_pythag_mm(dx: float, dy: float, dz: Optional[float] = None) -> str:
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return "—"
    if dz is None:
        v = float(np.sqrt(dx * dx + dy * dy))
        return _fmt_delta_mm(v)
    if not np.isfinite(dz):
        return "—"
    v = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    return _fmt_delta_mm(v)


# ======================================================
# Header / date range
# ======================================================
@st.cache_data(show_spinner=False)
def _read_header_and_groups(workbook_path: str, mtime: float) -> Tuple[str, List[Tuple[str, int, int, int]]]:
    sheet = _detect_data_sheet(workbook_path)

    header_df = pd.read_excel(
        workbook_path,
        sheet_name=sheet,
        header=None,
        nrows=1,
        engine="openpyxl",
    )
    header = header_df.iloc[0, :]
    ncols = header_df.shape[1]

    starts: List[Tuple[str, int]] = []
    for j in range(1, ncols):
        name = _clean_name(header.iloc[j])
        if name is not None:
            starts.append((name, j))

    groups: List[Tuple[str, int, int, int]] = []
    for name, jx in starts:
        jy = jx + 1
        jz = jx + 2
        if jz < ncols:
            groups.append((name, jx, jy, jz))

    return sheet, groups


@st.cache_data(show_spinner=False)
def _global_date_range(workbook_path: str, mtime: float) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    sheet = _detect_data_sheet(workbook_path)
    df = pd.read_excel(
        workbook_path,
        sheet_name=sheet,
        header=None,
        skiprows=1,
        usecols=[0],
        engine="openpyxl",
    )
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce", dayfirst=True).dropna()
    if dates.empty:
        return None, None
    return dates.min(), dates.max()


def _window_bounds_from_last(last_date: Optional[pd.Timestamp], days: int) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if last_date is None or pd.isna(last_date):
        return None, None
    end = last_date
    start = last_date - pd.Timedelta(days=days)
    return start, end


# ======================================================
# Data extraction (fast: only needed columns)
# ======================================================
@st.cache_data(show_spinner=False)
def _load_target_xy_medians(workbook_path: str, mtime: float) -> pd.DataFrame:
    sheet, groups = _read_header_and_groups(workbook_path, mtime)
    if not groups:
        return pd.DataFrame(columns=["name", "x", "y"])

    usecols = {0}
    for _, jx, jy, _ in groups:
        usecols.add(jx)
        usecols.add(jy)
    usecols = sorted(usecols)

    data = pd.read_excel(
        workbook_path,
        sheet_name=sheet,
        header=None,
        skiprows=1,
        usecols=usecols,
        engine="openpyxl",
    )

    data_num = data.apply(pd.to_numeric, errors="coerce")
    colpos = {c: i for i, c in enumerate(usecols)}

    rows = []
    for name, jx, jy, _ in groups:
        x_vals = data_num.iloc[:, colpos[jx]].to_numpy(dtype=float, copy=False)
        y_vals = data_num.iloc[:, colpos[jy]].to_numpy(dtype=float, copy=False)
        if np.isfinite(x_vals).sum() == 0 or np.isfinite(y_vals).sum() == 0:
            continue
        rows.append({"name": name, "x": float(np.nanmedian(x_vals)), "y": float(np.nanmedian(y_vals))})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.dropna(subset=["name"]).drop_duplicates(subset=["name"], keep="last")
    out = out.sort_values("name").reset_index(drop=True)
    return out


def _compute_deltas_from_df(df: pd.DataFrame, groups: List[Tuple[str, int, int, int]], usecols: List[int]) -> pd.DataFrame:
    df_num = df.apply(pd.to_numeric, errors="coerce")
    colpos = {c: i for i, c in enumerate(usecols)}

    rows = []
    for name, jx, jy, jz in groups:
        x = df_num.iloc[:, colpos[jx]].to_numpy(dtype=float, copy=False)
        y = df_num.iloc[:, colpos[jy]].to_numpy(dtype=float, copy=False)
        z = df_num.iloc[:, colpos[jz]].to_numpy(dtype=float, copy=False)

        idx_x = np.where(np.isfinite(x))[0]
        idx_y = np.where(np.isfinite(y))[0]
        idx_z = np.where(np.isfinite(z))[0]
        if idx_x.size == 0 or idx_y.size == 0 or idx_z.size == 0:
            continue

        dx = float(x[idx_x[-1]] - x[idx_x[0]])
        dy = float(y[idx_y[-1]] - y[idx_y[0]])
        dz = float(z[idx_z[-1]] - z[idx_z[0]])
        rows.append({"name": name, "dx": dx, "dy": dy, "dz": dz})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.dropna(subset=["name"]).drop_duplicates(subset=["name"], keep="last")
    out = out.sort_values("name").reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def _load_target_deltas_first_last(workbook_path: str, mtime: float) -> pd.DataFrame:
    sheet, groups = _read_header_and_groups(workbook_path, mtime)
    if not groups:
        return pd.DataFrame(columns=["name", "dx", "dy", "dz"])

    usecols = {0}
    for _, jx, jy, jz in groups:
        usecols.add(jx)
        usecols.add(jy)
        usecols.add(jz)
    usecols = sorted(usecols)

    df = pd.read_excel(
        workbook_path,
        sheet_name=sheet,
        header=None,
        skiprows=1,
        usecols=usecols,
        engine="openpyxl",
    )

    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce", dayfirst=True)
    if dates.notna().sum() > 0:
        order = np.argsort(dates.fillna(pd.Timestamp.min).to_numpy())
        df = df.iloc[order].reset_index(drop=True)

    return _compute_deltas_from_df(df, groups, usecols)


@st.cache_data(show_spinner=False)
def _load_target_deltas_last_n_days(workbook_path: str, mtime: float, days: int) -> pd.DataFrame:
    sheet, groups = _read_header_and_groups(workbook_path, mtime)
    if not groups:
        return pd.DataFrame(columns=["name", "dx", "dy", "dz"])

    usecols = {0}
    for _, jx, jy, jz in groups:
        usecols.add(jx)
        usecols.add(jy)
        usecols.add(jz)
    usecols = sorted(usecols)

    df = pd.read_excel(
        workbook_path,
        sheet_name=sheet,
        header=None,
        skiprows=1,
        usecols=usecols,
        engine="openpyxl",
    )

    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce", dayfirst=True)
    if dates.notna().sum() == 0:
        return pd.DataFrame(columns=["name", "dx", "dy", "dz"])

    last_date = dates.max()
    start_date = last_date - pd.Timedelta(days=days)

    mask = (dates >= start_date) & (dates <= last_date)
    dfw = df.loc[mask].copy()
    datesw = dates.loc[mask].copy()
    if dfw.empty:
        return pd.DataFrame(columns=["name", "dx", "dy", "dz"])

    order = np.argsort(datesw.fillna(pd.Timestamp.min).to_numpy())
    dfw = dfw.iloc[order].reset_index(drop=True)

    return _compute_deltas_from_df(dfw, groups, usecols)


# ======================================================
# Ultra-optimized custom period: load once, compute in JS while dragging
# ======================================================
@st.cache_data(show_spinner=False)
def _load_timeseries_xyz(
    workbook_path: str, mtime: float
) -> Tuple[pd.Series, List[Tuple[str, int, int, int]], List[int], pd.DataFrame]:
    sheet, groups = _read_header_and_groups(workbook_path, mtime)
    if not groups:
        return pd.Series(dtype="datetime64[ns]"), [], [0], pd.DataFrame()

    usecols = {0}
    for _, jx, jy, jz in groups:
        usecols.add(jx)
        usecols.add(jy)
        usecols.add(jz)
    usecols = sorted(usecols)

    df = pd.read_excel(
        workbook_path,
        sheet_name=sheet,
        header=None,
        skiprows=1,
        usecols=usecols,
        engine="openpyxl",
    )

    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce", dayfirst=True)

    mask = dates.notna()
    df = df.loc[mask].reset_index(drop=True)
    dates = dates.loc[mask].reset_index(drop=True)

    order = np.argsort(dates.to_numpy())
    df = df.iloc[order].reset_index(drop=True)
    dates = dates.iloc[order].reset_index(drop=True)

    df_num = df.apply(pd.to_numeric, errors="coerce")
    return dates, groups, usecols, df_num


# ======================================================
# Drawing helpers
# ======================================================
def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _robust_bounds(vals: np.ndarray, q_low=1.0, q_high=99.0) -> Tuple[float, float]:
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
    pad = 30.0
    inner_w = max(w - 2 * pad, 1.0)
    inner_h = max(h - 2 * pad, 1.0)

    xs = points["x"].to_numpy(dtype=float)
    ys = points["y"].to_numpy(dtype=float)
    bx0, bx1 = _robust_bounds(xs, 1.0, 99.0)
    by0, by1 = _robust_bounds(ys, 1.0, 99.0)

    span_x = max(bx1 - bx0, 1e-12)
    span_y = max(by1 - by0, 1e-12)

    sx = inner_w / span_x
    sy = inner_h / span_y
    s = min(sx, sy)

    content_w = span_x * s
    content_h = span_y * s
    off_x = (w - content_w) / 2.0
    off_y = (h - content_h) / 2.0

    def clamp(v: float, v0: float, v1: float) -> float:
        return v0 if v < v0 else (v1 if v > v1 else v)

    def to_px(x: float, y: float) -> Tuple[float, float]:
        cx = clamp(x, bx0, bx1)
        cy = clamp(y, by0, by1)
        px = off_x + (cx - bx0) * s
        py = off_y + (by1 - cy) * s
        return px, py

    return to_px, float(s)


def _auto_vector_scale_px(points: pd.DataFrame, deltas: pd.DataFrame, to_px, desired_med_px: float = 25.0) -> float:
    if deltas.empty or points.empty:
        return 1.0

    dd = deltas.drop_duplicates(subset=["name"], keep="last").set_index("name", drop=False)
    lengths = []

    for _, r in points.iterrows():
        name = str(r["name"])
        if name not in dd.index:
            continue
        dx = float(dd.loc[name, "dx"])
        dy = float(dd.loc[name, "dy"])
        if not np.isfinite(dx) or not np.isfinite(dy):
            continue
        if abs(dx) < 1e-15 and abs(dy) < 1e-15:
            continue

        x = float(r["x"])
        y = float(r["y"])
        px, py = to_px(x, y)
        pex, pey = to_px(x + dx, y + dy)
        L = float(np.hypot(pex - px, pey - py))
        if np.isfinite(L) and L > 0:
            lengths.append(L)

    if not lengths:
        return 1.0

    med = float(np.median(lengths))
    if med < 1e-9:
        return 1.0

    scale = desired_med_px / med
    return max(min(scale, 500.0), 0.02)


# ======================================================
# HTML wrapper
# ======================================================
def _wrap_svg_html(
    width: int,
    height: int,
    svg_inner: str,
    with_vectors: bool,
    base_px_per_m: float,
    vec_base_scale: float,
    tooltip_mode: str,
    custom_period_payload: Optional[dict] = None,
    hide_scale_v: bool = False,
    force_show_xy_scale: bool = True,
) -> str:
    defs = ""
    vec_css = ""
    vec_slider = ""

    if with_vectors:
        vec_slider = """<input id="vecScale" class="rng rng-orange" type="range" min="0.1" max="20" value="1" step="0.05" />"""
        defs = """
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(255,145,0,0.70)"></path>
          </marker>
        </defs>
        """
        vec_css = """
        .vec{
          stroke: rgba(255,145,0,0.55);
          stroke-width: 2.2px;
          marker-end: url(#arrow);
          stroke-linecap: round;
          cursor: default;
        }
        .vec:hover{ stroke: rgba(255,145,0,0.88); }
        """

    controls_html = f"""
    <div id="controls" class="{'' if with_vectors else 'hidden'}">
      {vec_slider}
    </div>
    """

    scale_v_class = "hidden" if (hide_scale_v or not with_vectors) else ""
    scale_xy_class = "" if force_show_xy_scale else "hidden"

    custom_data_json = "null"
    custom_top_html = ""
    custom_top_h = 0
    if custom_period_payload is not None:
        custom_data_json = json.dumps(custom_period_payload, ensure_ascii=False)
        custom_top_h = 88
        custom_top_html = """
        <div id="customTop" class="customTop">
          <div class="customTitle">
            <span class="t">Période custom</span>
            <span class="d" id="customDates">—</span>
          </div>

          <div class="dualWrap">
            <div class="trackBase" id="trackBase"></div>
            <div class="trackSel"  id="trackSel"></div>

            <input id="rangeA" class="range rangeA" type="range" min="0" max="100" value="20" step="1" />
            <input id="rangeB" class="range rangeB" type="range" min="0" max="100" value="80" step="1" />
          </div>
        </div>
        """

    view_h = int(height - custom_top_h) if height - custom_top_h > 160 else max(int(height), 260)

    tpl = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html,body{margin:0;padding:0;background:white;}

    #wrap{
      width:__W__px;
      height:__H__px;
      background:white;
      overflow:hidden;
      display:flex;
      flex-direction:column;
    }

    .customTop{
      margin: 12px 12px 0 12px;
      padding: 10px 12px 10px;
      border-radius: 14px;
      background: rgba(255,255,255,0.86);
      box-shadow: 0 6px 18px rgba(0,0,0,0.08);
      backdrop-filter: blur(6px);
      user-select:none;
      flex: 0 0 auto;
      z-index: 60;
    }
    .customTitle{
      display:flex;
      align-items:baseline;
      gap: 10px;
      margin-bottom: 8px;
      font: 12px/1.1 system-ui, -apple-system, Segoe UI, Roboto, Arial;
      color: rgba(0,0,0,0.82);
    }
    .customTitle .t{ font-weight: 750; }
    .customTitle .d{ opacity: 0.9; }

    .dualWrap{ position:relative; height: 28px; margin-top: 2px; }

    .trackBase{
      position:absolute; left: 0; right: 0;
      top: 50%; transform: translateY(-50%);
      height: 6px; border-radius: 999px;
      background: rgba(122,14,14,0.18);
    }
    .trackSel{
      position:absolute;
      top: 50%; transform: translateY(-50%);
      height: 6px; border-radius: 999px;
      background: rgba(122,14,14,0.92);
      left: 20%; width: 60%;
    }
    .range{
      position:absolute; left:0; right:0; top:0;
      width:100%; height: 28px; margin:0;
      background: transparent;
      pointer-events: none;
      -webkit-appearance:none;
      appearance:none;
      outline:none;
    }
    .range::-webkit-slider-runnable-track{ height: 6px; background: transparent; }
    .range::-webkit-slider-thumb{
      -webkit-appearance:none;
      appearance:none;
      width: 16px; height: 16px;
      border-radius: 50%;
      border: 2px solid rgba(255,255,255,0.96);
      box-shadow: 0 4px 10px rgba(0,0,0,0.18);
      background: rgba(122,14,14,0.95);
      cursor: pointer;
      pointer-events: auto;
      margin-top: -5px; /* align thumbs with the 6px line */
    }
    .range::-moz-range-track{ height: 6px; background: transparent; }
    .range::-moz-range-thumb{
      width: 16px; height: 16px;
      border-radius: 50%;
      border: 2px solid rgba(255,255,255,0.96);
      box-shadow: 0 4px 10px rgba(0,0,0,0.18);
      background: rgba(122,14,14,0.95);
      cursor: pointer;
      pointer-events: auto;
    }

    #viewport{
      position:relative;
      width: __W__px;
      height: __VIEW_H__px;
      background:white;
      flex: 1 1 auto;
      z-index:0;
    }

    svg{display:block;background:white; touch-action:none;}

    .target{
      fill: rgba(255,0,0,0.88);
      stroke: rgba(0,0,0,0.92);
      stroke-width: 1.2px;
      shape-rendering: geometricPrecision;
      cursor: default;
    }
    .target:hover{ fill: rgba(255,0,0,0.98); }

    __VEC_CSS__

    .tooltip{
      position:absolute;
      pointer-events:none;
      background: rgba(20,20,20,0.92);
      color: white;
      font: 12px/1.2 system-ui, -apple-system, Segoe UI, Roboto, Arial;
      padding: 8px 10px;
      border-radius: 10px;
      white-space: nowrap;
      transform: translate(10px, 10px);
      opacity: 0;
      transition: opacity 60ms linear;
      z-index: 80;
    }
    .tooltip .title{font-weight:700;margin-bottom:4px;}
    .tooltip .row{opacity:0.95;}

    #controls{
      position:absolute;
      top: 12px;
      right: 12px;
      width: 170px;
      padding: 10px;
      border-radius: 14px;
      background: rgba(255,255,255,0.86);
      box-shadow: 0 6px 18px rgba(0,0,0,0.08);
      backdrop-filter: blur(6px);
      display:flex;
      flex-direction:column;
      gap:10px;
      user-select:none;
      z-index: 70;
    }
    #controls.hidden{ display:none; }

    .rng{
      width: 100%;
      margin: 0;
      -webkit-appearance: none;
      appearance: none;
      height: 4px;
      border-radius: 999px;
      outline: none;
    }
    .rng::-webkit-slider-thumb{
      -webkit-appearance: none;
      appearance: none;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      border: 2px solid rgba(255,255,255,0.96);
      box-shadow: 0 4px 10px rgba(0,0,0,0.15);
      cursor: pointer;
    }
    .rng::-moz-range-thumb{
      width: 14px;
      height: 14px;
      border-radius: 50%;
      border: 2px solid rgba(255,255,255,0.96);
      box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .rng-orange{ background: rgba(255,145,0,0.35); }
    .rng-orange::-webkit-slider-thumb{ background: rgba(255,145,0,0.95); }
    .rng-orange::-moz-range-thumb{ background: rgba(255,145,0,0.95); }

    .scalebox{
      position:absolute;
      bottom: 12px;
      padding: 8px 10px 7px;
      border-radius: 12px;
      background: rgba(255,255,255,0.86);
      box-shadow: 0 6px 18px rgba(0,0,0,0.08);
      backdrop-filter: blur(6px);
      user-select:none;
      z-index: 75;
      pointer-events: none;
    }
    #scaleXY{ left: 12px; }
    #scaleV { right: 12px; }
    .scalebox.hidden{ display:none; }

    .bar{
      height: 8px;
      border-radius: 999px;
      overflow: hidden;
      display: inline-block;
      background: transparent;
    }
    .bar-fill{
      height: 100%;
      width: 100%;
      background: rgba(0,0,0,0.85);
    }
    .bar.orange .bar-fill{ background: rgba(255,145,0,0.95); }

    .labelsbar{
      position: relative;
      margin-top: 6px;
      height: 14px;
      font: 12px/1.1 system-ui, -apple-system, Segoe UI, Roboto, Arial;
      color: rgba(0,0,0,0.78);
    }
    #scaleV .labelsbar{ color: rgba(255,145,0,0.95); }
    .labelsbar .mid{
      position:absolute;
      left: 50%;
      transform: translateX(-50%);
      white-space: nowrap;
    }
    .labelsbar .end{
      position:absolute;
      right: 0;
      white-space: nowrap;
    }
  </style>
</head>
<body>
  <div id="wrap">
    __CUSTOM_TOP__

    <div id="viewport"
         data-base-px-per-m="__BASE_PX_PER_M__"
         data-vec-base-scale="__VEC_BASE_SCALE__"
         data-tooltip-mode="__TOOLTIP_MODE__">

      <div id="tip" class="tooltip"></div>

      __CONTROLS__

      <div id="scaleXY" class="scalebox __SCALE_XY_CLASS__">
        <div class="bar" id="barXY"><div class="bar-fill"></div></div>
        <div class="labelsbar" id="labelsXY">
          <span class="mid" id="labXYa">—</span>
          <span class="end" id="labXYb">—</span>
        </div>
      </div>

      <div id="scaleV" class="scalebox __SCALE_V_CLASS__">
        <div class="bar orange" id="barV"><div class="bar-fill"></div></div>
        <div class="labelsbar" id="labelsV">
          <span class="mid" id="labVa">—</span>
          <span class="end" id="labVb">—</span>
        </div>
      </div>

      <svg id="svg" width="__W__" height="__VIEW_H__"
           viewBox="0 0 __W__ __VIEW_H__"
           xmlns="http://www.w3.org/2000/svg">
        __DEFS__
        <g id="scene">__SVG_INNER__</g>
      </svg>
    </div>
  </div>

  <script>
    const CUSTOM = __CUSTOM_DATA_JSON__;

    const viewport = document.getElementById('viewport');
    const tip = document.getElementById('tip');
    const svg = document.getElementById('svg');
    const scene = document.getElementById('scene');

    const tooltipMode = (viewport.getAttribute('data-tooltip-mode') || 'name').toLowerCase();
    const vecScaleEl = document.getElementById('vecScale');

    const basePxPerM = parseFloat(viewport.getAttribute('data-base-px-per-m')) || 1.0;
    const vecBaseScale = parseFloat(viewport.getAttribute('data-vec-base-scale')) || 1.0;

    const barXY  = document.getElementById('barXY');
    const barV   = document.getElementById('barV');
    const labelsXY = document.getElementById('labelsXY');
    const labelsV  = document.getElementById('labelsV');
    const labXYa = document.getElementById('labXYa');
    const labXYb = document.getElementById('labXYb');
    const labVa  = document.getElementById('labVa');
    const labVb  = document.getElementById('labVb');

    function esc(s){
      return String(s)
        .split('&').join('&amp;')
        .split('<').join('&lt;')
        .split('>').join('&gt;')
        .split('"').join('&quot;')
        .split("'").join('&#39;');
    }

    function showTipNameOnly(name, x, y){
      tip.innerHTML = '<div class="title">' + esc(name) + '</div>';
      tip.style.left = x + 'px';
      tip.style.top  = y + 'px';
      tip.style.opacity = '1';
    }

    function showTipDeltas(name, dx, dy, dz, dplan, dspatial, x, y){
      tip.innerHTML =
        '<div class="title">' + esc(name) + '</div>' +
        '<div class="row">ΔX : ' + esc(dx) + '</div>' +
        '<div class="row">ΔY : ' + esc(dy) + '</div>' +
        '<div class="row">ΔZ : ' + esc(dz) + '</div>' +
        '<div class="row">Δ plan : ' + esc(dplan) + '</div>' +
        '<div class="row">Δ spatial : ' + esc(dspatial) + '</div>';
      tip.style.left = x + 'px';
      tip.style.top  = y + 'px';
      tip.style.opacity = '1';
    }

    function hideTip(){ tip.style.opacity = '0'; }

    svg.addEventListener('mousemove', (e) => {
      const t = e.target;
      if (t && t.classList) {
        if (t.classList.contains('target') || t.classList.contains('vec')) {
          const name = t.getAttribute('data-name') || '';
          if (!name) { hideTip(); return; }

          const rect = viewport.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;

          if (tooltipMode === 'deltas') {
            const dx = t.getAttribute('data-dx') || '—';
            const dy = t.getAttribute('data-dy') || '—';
            const dz = t.getAttribute('data-dz') || '—';
            const dp = t.getAttribute('data-dp') || '—';
            const ds = t.getAttribute('data-ds') || '—';
            showTipDeltas(name, dx, dy, dz, dp, ds, x, y);
          } else {
            showTipNameOnly(name, x, y);
          }
          return;
        }
      }
      hideTip();
    });
    svg.addEventListener('mouseleave', hideTip);

    function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }

    function vecMult(){
      if (!vecScaleEl) return 1.0;
      const v = parseFloat(vecScaleEl.value);
      return Number.isFinite(v) ? v : 1.0;
    }
    function applyVectorScale(mult){
      if (!vecScaleEl) return;
      const lines = svg.querySelectorAll('.vec');
      lines.forEach(line => {
        const x1 = parseFloat(line.getAttribute('data-x1'));
        const y1 = parseFloat(line.getAttribute('data-y1'));
        const vx = parseFloat(line.getAttribute('data-vx'));
        const vy = parseFloat(line.getAttribute('data-vy'));
        if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(vx) || !Number.isFinite(vy)) return;
        line.setAttribute('x2', (x1 + vx * mult).toFixed(3));
        line.setAttribute('y2', (y1 + vy * mult).toFixed(3));
      });
    }

    let panX = 0.0;
    let panY = 0.0;
    let wheelZoom = 1.0;

    function currentZoom(){ return wheelZoom; }
    function applyTransform(){
      const z = currentZoom();
      scene.setAttribute('transform',
        'translate(' + panX.toFixed(3) + ',' + panY.toFixed(3) + ') scale(' + z.toFixed(6) + ')'
      );
    }

    function niceStep(x){
      if (!Number.isFinite(x) || x <= 0) return 1;
      const e = Math.floor(Math.log10(x));
      if (!Number.isFinite(e)) return 1;
      const k = Math.pow(10, e);
      if (!Number.isFinite(k) || k <= 0) return 1;
      const u = x / k;
      if (!Number.isFinite(u)) return 1;
      let n = 1;
      if (u <= 1) n = 1;
      else if (u <= 2) n = 2;
      else if (u <= 5) n = 5;
      else n = 10;
      const out = n * k;
      return (Number.isFinite(out) && out > 0) ? out : 1;
    }

    function fmtLen(v, unit){
      if (!Number.isFinite(v)) return '—';
      if (v >= 10) return Math.round(v) + ' ' + unit;
      if (v >= 1)  return (Math.round(v*10)/10).toFixed(1).replace('.0','') + ' ' + unit;
      let s = (Math.round(v*100)/100).toFixed(2);
      while (s.indexOf('.') >= 0 && (s.endsWith('0') || s.endsWith('.'))) {
        s = s.slice(0, -1);
        if (s.endsWith('.')) { s = s.slice(0, -1); break; }
      }
      return s + ' ' + unit;
    }

    function updateScaleBars(){
      if (!barXY || !labelsXY || !labXYa || !labXYb) return;

      const z = currentZoom();
      const pxPerM = basePxPerM * z;

      const targetPx = 170;
      const minPx = 110;
      const maxPx = 230;

      let Lm = niceStep(targetPx / Math.max(pxPerM, 1e-12));
      let wLm = Lm * pxPerM;

      for (let i=0; i<24 && Number.isFinite(wLm) && wLm < minPx; i++){
        Lm = niceStep(Lm * 2.0);
        wLm = Lm * pxPerM;
      }
      for (let i=0; i<24 && Number.isFinite(wLm) && wLm > maxPx && Lm > 0; i++){
        Lm = niceStep(Lm / 2.0);
        wLm = Lm * pxPerM;
      }
      if (!Number.isFinite(wLm) || wLm <= 0){
        Lm = 1;
        wLm = clamp(targetPx, minPx, maxPx);
      }

      barXY.style.width = wLm.toFixed(1) + 'px';
      labelsXY.style.width = wLm.toFixed(1) + 'px';
      labXYa.textContent = fmtLen(Lm/2, 'm');
      labXYb.textContent = fmtLen(Lm, 'm');

      if (barV && labelsV && labVa && labVb){
        let pxPerMm = (pxPerM * vecBaseScale * vecMult()) / 1000.0;
        if (!Number.isFinite(pxPerMm) || pxPerMm <= 0) pxPerMm = 1e-12;

        let Lmm = niceStep(targetPx / pxPerMm);
        let wLmm = Lmm * pxPerMm;

        for (let i=0; i<24 && Number.isFinite(wLmm) && wLmm < minPx; i++){
          Lmm = niceStep(Lmm * 2.0);
          wLmm = Lmm * pxPerMm;
        }
        for (let i=0; i<24 && Number.isFinite(wLmm) && wLmm > maxPx && Lmm > 0; i++){
          Lmm = niceStep(Lmm / 2.0);
          wLmm = Lmm * pxPerMm;
        }
        if (!Number.isFinite(wLmm) || wLmm <= 0){
          Lmm = 1;
          wLmm = clamp(targetPx, minPx, maxPx);
        }

        barV.style.width = wLmm.toFixed(1) + 'px';
        labelsV.style.width = wLmm.toFixed(1) + 'px';
        labVa.textContent = fmtLen(Lmm/2, 'mm');
        labVb.textContent = fmtLen(Lmm, 'mm');
      }
    }

    if (vecScaleEl){
      applyVectorScale(vecMult());
      vecScaleEl.addEventListener('input', () => {
        applyVectorScale(vecMult());
        updateScaleBars();
      });
    }

    svg.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = svg.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;

      const z0 = currentZoom();
      const factor = Math.exp(-e.deltaY * 0.0012);
      const z1 = clamp(wheelZoom * factor, 0.08, 25.0);

      const sx = (cx - panX) / z0;
      const sy = (cy - panY) / z0;

      wheelZoom = z1;
      panX = cx - sx * z1;
      panY = cy - sy * z1;

      applyTransform();
      updateScaleBars();
      if (CUSTOM && CUSTOM.enabled) requestCustomRecompute();
    }, { passive: false });

    let isPanning = false;
    let startX = 0.0, startY = 0.0;
    let startPanX = 0.0, startPanY = 0.0;

    svg.addEventListener('pointerdown', (e) => {
      const path = e.composedPath ? e.composedPath() : [];
      if (path.some(el => el && el.id === 'controls')) return;
      if (path.some(el => el && el.id === 'customTop')) return;

      isPanning = true;
      startX = e.clientX;
      startY = e.clientY;
      startPanX = panX;
      startPanY = panY;
      svg.setPointerCapture(e.pointerId);
    });

    svg.addEventListener('pointermove', (e) => {
      if (!isPanning) return;
      panX = startPanX + (e.clientX - startX);
      panY = startPanY + (e.clientY - startY);
      applyTransform();
    });

    svg.addEventListener('pointerup', (e) => {
      isPanning = false;
      try { svg.releasePointerCapture(e.pointerId); } catch (_) {}
    });
    svg.addEventListener('pointercancel', () => { isPanning = false; });

    svg.addEventListener('dblclick', () => {
      panX = 0.0; panY = 0.0; wheelZoom = 1.0;
      applyTransform();
      updateScaleBars();
      if (CUSTOM && CUSTOM.enabled) requestCustomRecompute();
    });

    let customRAF = 0;

    function mmFmt(m){
      if (!Number.isFinite(m)) return '—';
      return (m*1000).toFixed(1) + ' mm';
    }
    function hypot2(a,b){
      if (!Number.isFinite(a) || !Number.isFinite(b)) return NaN;
      return Math.sqrt(a*a + b*b);
    }
    function hypot3(a,b,c){
      if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) return NaN;
      return Math.sqrt(a*a + b*b + c*c);
    }
    function dstrFromEpochMs(ms){
      const d = new Date(ms);
      const dd = String(d.getDate()).padStart(2,'0');
      const mm = String(d.getMonth()+1).padStart(2,'0');
      const yy = d.getFullYear();
      return dd + '/' + mm + '/' + yy;
    }

    function requestCustomRecompute(){
      if (!CUSTOM || !CUSTOM.enabled) return;
      if (customRAF) cancelAnimationFrame(customRAF);
      customRAF = requestAnimationFrame(() => {
        customRAF = 0;
        recomputeCustomVectors();
      });
    }

    function recomputeCustomVectors(){
      if (!CUSTOM || !CUSTOM.enabled) return;

      const a = document.getElementById('rangeA');
      const b = document.getElementById('rangeB');
      if (!a || !b) return;

      let ia = parseInt(a.value, 10);
      let ib = parseInt(b.value, 10);
      if (!Number.isFinite(ia)) ia = 0;
      if (!Number.isFinite(ib)) ib = 0;
      if (ia > ib){ const t = ia; ia = ib; ib = t; }

      const dates = CUSTOM.dates_ms;
      const N = dates.length;
      ia = clamp(ia, 0, Math.max(N-1,0));
      ib = clamp(ib, 0, Math.max(N-1,0));

      const startMs = dates[ia];
      const endMs   = dates[ib];

      const label = document.getElementById('customDates');
      if (label){
        label.textContent = 'du ' + dstrFromEpochMs(startMs) + ' au ' + dstrFromEpochMs(endMs);
      }

      const sel = document.getElementById('trackSel');
      const denom = Math.max(N-1, 1);
      const p0 = (100 * ia / denom);
      const p1 = (100 * ib / denom);
      if (sel){
        sel.style.left = p0.toFixed(2) + '%';
        sel.style.width = Math.max(p1 - p0, 0).toFixed(2) + '%';
      }

      const mult = vecMult();
      const lines = svg.querySelectorAll('.vec');
      lines.forEach(line => {
        const name = line.getAttribute('data-name') || '';
        if (!name) return;

        const arrx = CUSTOM.x[name];
        const arry = CUSTOM.y[name];
        const arrz = CUSTOM.z[name];
        if (!arrx || !arry || !arrz) return;

        let fx = -1, lx = -1;
        let fy = -1, ly = -1;
        let fz = -1, lz = -1;

        for (let i = ia; i <= ib; i++){
          const vx0 = arrx[i];
          if (fx < 0 && Number.isFinite(vx0)) fx = i;
          const vy0 = arry[i];
          if (fy < 0 && Number.isFinite(vy0)) fy = i;
          const vz0 = arrz[i];
          if (fz < 0 && Number.isFinite(vz0)) fz = i;
          if (fx >= 0 && fy >= 0 && fz >= 0) break;
        }
        for (let i = ib; i >= ia; i--){
          const vx0 = arrx[i];
          if (lx < 0 && Number.isFinite(vx0)) lx = i;
          const vy0 = arry[i];
          if (ly < 0 && Number.isFinite(vy0)) ly = i;
          const vz0 = arrz[i];
          if (lz < 0 && Number.isFinite(vz0)) lz = i;
          if (lx >= 0 && ly >= 0 && lz >= 0) break;
        }

        if (fx < 0 || lx < 0 || fy < 0 || ly < 0 || fz < 0 || lz < 0){
          line.setAttribute('data-vx','0');
          line.setAttribute('data-vy','0');
          line.setAttribute('data-dx','—');
          line.setAttribute('data-dy','—');
          line.setAttribute('data-dz','—');
          line.setAttribute('data-dp','—');
          line.setAttribute('data-ds','—');
          const x1 = parseFloat(line.getAttribute('data-x1'));
          const y1 = parseFloat(line.getAttribute('data-y1'));
          if (Number.isFinite(x1) && Number.isFinite(y1)){
            line.setAttribute('x2', x1.toFixed(3));
            line.setAttribute('y2', y1.toFixed(3));
          }
          return;
        }

        const dx = arrx[lx] - arrx[fx];
        const dy = arry[ly] - arry[fy];
        const dz = arrz[lz] - arrz[fz];

        const ux = CUSTOM.vx_unit[name];
        const uy = CUSTOM.vy_unit[name];

        const vx = (dx * ux) * vecBaseScale;
        const vy = (dy * uy) * vecBaseScale;

        line.setAttribute('data-vx', Number.isFinite(vx) ? vx.toFixed(3) : '0');
        line.setAttribute('data-vy', Number.isFinite(vy) ? vy.toFixed(3) : '0');

        const dp = hypot2(dx, dy);
        const ds = hypot3(dx, dy, dz);

        line.setAttribute('data-dx', mmFmt(dx));
        line.setAttribute('data-dy', mmFmt(dy));
        line.setAttribute('data-dz', mmFmt(dz));
        line.setAttribute('data-dp', mmFmt(dp));
        line.setAttribute('data-ds', mmFmt(ds));

        const circ = svg.querySelector('.target[data-name="' + CSS.escape(name) + '"]');
        if (circ){
          circ.setAttribute('data-dx', mmFmt(dx));
          circ.setAttribute('data-dy', mmFmt(dy));
          circ.setAttribute('data-dz', mmFmt(dz));
          circ.setAttribute('data-dp', mmFmt(dp));
          circ.setAttribute('data-ds', mmFmt(ds));
        }
      });

      applyVectorScale(mult);
      updateScaleBars();
    }

    function initCustomUI(){
      if (!CUSTOM || !CUSTOM.enabled) return;

      const a = document.getElementById('rangeA');
      const b = document.getElementById('rangeB');
      const N = (CUSTOM.dates_ms || []).length;
      if (!a || !b || N <= 1) return;

      a.min = '0'; a.max = String(N-1);
      b.min = '0'; b.max = String(N-1);

      a.value = String(clamp(CUSTOM.default_i0 || 0, 0, N-1));
      b.value = String(clamp(CUSTOM.default_i1 || (N-1), 0, N-1));

      const normalize = () => {
        let ia = parseInt(a.value,10), ib = parseInt(b.value,10);
        if (ia > ib){
          const t = ia; ia = ib; ib = t;
          a.value = String(ia);
          b.value = String(ib);
        }
      };

      const onAny = () => {
        normalize();
        requestCustomRecompute();
      };

      a.addEventListener('input', onAny);
      b.addEventListener('input', onAny);

      requestCustomRecompute();
    }

    applyTransform();
    updateScaleBars();
    initCustomUI();
  </script>
</body>
</html>
""".strip()

    html = (
        tpl.replace("__W__", str(int(width)))
        .replace("__H__", str(int(height)))
        .replace("__VIEW_H__", str(int(view_h)))
        .replace("__SVG_INNER__", svg_inner)
        .replace("__DEFS__", defs)
        .replace("__VEC_CSS__", vec_css)
        .replace("__CONTROLS__", controls_html)
        .replace("__BASE_PX_PER_M__", f"{base_px_per_m:.10f}")
        .replace("__VEC_BASE_SCALE__", f"{vec_base_scale:.10f}")
        .replace("__TOOLTIP_MODE__", tooltip_mode)
        .replace("__SCALE_V_CLASS__", scale_v_class)
        .replace("__SCALE_XY_CLASS__", scale_xy_class)
        .replace("__CUSTOM_DATA_JSON__", custom_data_json)
        .replace("__CUSTOM_TOP__", custom_top_html)
    )
    return html


# ======================================================
# SVG builders (these names MUST exist => fixes your NameError)
# ======================================================
def _build_svg_targets(points: pd.DataFrame, width: int = 950, height: int = 700) -> str:
    if points.empty:
        return "<div style='padding:12px;font-family:system-ui'>Aucune cible détectée.</div>"

    to_px, base_px_per_m = _make_mapper(points, width, height)

    circles = []
    for _, row in points.iterrows():
        name = _escape_xml(str(row["name"]))
        px, py = to_px(float(row["x"]), float(row["y"]))
        circles.append(f"""<circle class="target" cx="{px:.3f}" cy="{py:.3f}" r="7" data-name="{name}"></circle>""")

    return _wrap_svg_html(
        width=width,
        height=height,
        svg_inner="".join(circles),
        with_vectors=False,
        base_px_per_m=base_px_per_m,
        vec_base_scale=1.0,
        tooltip_mode="name",
        custom_period_payload=None,
        hide_scale_v=True,
        force_show_xy_scale=True,
    )


def _build_svg_mouvement(points: pd.DataFrame, deltas: pd.DataFrame, width: int = 950, height: int = 700) -> str:
    if points.empty:
        return "<div style='padding:12px;font-family:system-ui'>Aucune cible détectée.</div>"
    if deltas.empty:
        return "<div style='padding:12px;font-family:system-ui'>Aucun mouvement calculé.</div>"

    points_u = points.drop_duplicates(subset=["name"], keep="last").reset_index(drop=True)
    deltas_u = deltas.drop_duplicates(subset=["name"], keep="last").reset_index(drop=True)
    dd = deltas_u.set_index("name")

    to_px, base_px_per_m = _make_mapper(points_u, width, height)
    vec_base_scale = _auto_vector_scale_px(points_u, deltas_u, to_px, desired_med_px=25.0)

    circles = []
    vectors = []

    for _, r in points_u.iterrows():
        name = str(r["name"])
        x = float(r["x"])
        y = float(r["y"])
        px, py = to_px(x, y)

        circles.append(
            f"""<circle class="target" cx="{px:.3f}" cy="{py:.3f}" r="7" data-name="{_escape_xml(name)}"></circle>"""
        )

        if name not in dd.index:
            continue

        dx = float(dd.loc[name, "dx"])
        dy = float(dd.loc[name, "dy"])
        dz = float(dd.loc[name, "dz"])
        if not (np.isfinite(dx) and np.isfinite(dy) and np.isfinite(dz)):
            continue
        if abs(dx) < 1e-15 and abs(dy) < 1e-15 and abs(dz) < 1e-15:
            continue

        dplan = _fmt_pythag_mm(dx, dy, None)
        dspat = _fmt_pythag_mm(dx, dy, dz)

        pex, pey = to_px(x + dx, y + dy)
        vx = (pex - px) * vec_base_scale
        vy = (pey - py) * vec_base_scale

        circles[-1] = circles[-1].replace(
            "></circle>",
            (
                f' data-dx="{_fmt_delta_mm(dx)}"'
                f' data-dy="{_fmt_delta_mm(dy)}"'
                f' data-dz="{_fmt_delta_mm(dz)}"'
                f' data-dp="{dplan}"'
                f' data-ds="{dspat}"></circle>'
            ),
        )

        vectors.append(
            f"""<line class="vec"
                      x1="{px:.3f}" y1="{py:.3f}" x2="{px:.3f}" y2="{py:.3f}"
                      data-x1="{px:.3f}" data-y1="{py:.3f}"
                      data-vx="{vx:.3f}" data-vy="{vy:.3f}"
                      data-name="{_escape_xml(name)}"
                      data-dx="{_fmt_delta_mm(dx)}"
                      data-dy="{_fmt_delta_mm(dy)}"
                      data-dz="{_fmt_delta_mm(dz)}"
                      data-dp="{dplan}"
                      data-ds="{dspat}"></line>"""
        )

    svg_inner = "".join(circles) + "".join(vectors)

    return _wrap_svg_html(
        width=width,
        height=height,
        svg_inner=svg_inner,
        with_vectors=True,
        base_px_per_m=base_px_per_m,
        vec_base_scale=vec_base_scale,
        tooltip_mode="deltas",
        custom_period_payload=None,
        hide_scale_v=False,
        force_show_xy_scale=True,
    )


def _build_svg_mouvement_custom_live(
    points: pd.DataFrame,
    workbook_path: str,
    mtime: float,
    width: int = 950,
    height: int = 700,
) -> str:
    if points.empty:
        return "<div style='padding:12px;font-family:system-ui'>Aucune cible détectée.</div>"

    dates, groups, usecols, df_num = _load_timeseries_xyz(workbook_path, mtime)
    if dates.empty or not groups or df_num.empty:
        return "<div style='padding:12px;font-family:system-ui'>Aucune série temporelle exploitable.</div>"

    points_u = points.drop_duplicates(subset=["name"], keep="last").reset_index(drop=True)
    to_px, base_px_per_m = _make_mapper(points_u, width, height)

    try:
        deltas_full = _load_target_deltas_first_last(workbook_path, mtime)
    except Exception:
        deltas_full = pd.DataFrame(columns=["name", "dx", "dy", "dz"])
    vec_base_scale = _auto_vector_scale_px(points_u, deltas_full, to_px, desired_med_px=25.0)

    colpos = {c: i for i, c in enumerate(usecols)}

    dates_ms = (dates.astype("int64") // 1_000_000).to_list()
    N = len(dates_ms)

    x_map: Dict[str, List[Optional[float]]] = {}
    y_map: Dict[str, List[Optional[float]]] = {}
    z_map: Dict[str, List[Optional[float]]] = {}
    vx_unit: Dict[str, float] = {}
    vy_unit: Dict[str, float] = {}

    for name, jx, jy, jz in groups:
        if name is None:
            continue
        if jx not in colpos or jy not in colpos or jz not in colpos:
            continue

        ax = df_num.iloc[:, colpos[jx]].to_numpy(dtype=float, copy=False)
        ay = df_num.iloc[:, colpos[jy]].to_numpy(dtype=float, copy=False)
        az = df_num.iloc[:, colpos[jz]].to_numpy(dtype=float, copy=False)

        x_map[name] = [None if not np.isfinite(v) else float(v) for v in ax]
        y_map[name] = [None if not np.isfinite(v) else float(v) for v in ay]
        z_map[name] = [None if not np.isfinite(v) else float(v) for v in az]

        p = points_u[points_u["name"] == name]
        if p.empty:
            continue
        x0 = float(p.iloc[0]["x"])
        y0 = float(p.iloc[0]["y"])
        px0, py0 = to_px(x0, y0)
        px1, _ = to_px(x0 + 1.0, y0)
        _, py2 = to_px(x0, y0 + 1.0)
        vx_unit[name] = float(px1 - px0)
        vy_unit[name] = float(py2 - py0)

    if N <= 1:
        i0, i1 = 0, 0
    else:
        last_ms = dates_ms[-1]
        start_ms = last_ms - int(30 * 24 * 3600 * 1000)
        i0 = int(np.searchsorted(np.array(dates_ms, dtype=np.int64), start_ms, side="left"))
        i0 = max(min(i0, N - 1), 0)
        i1 = N - 1

    circles = []
    vectors = []
    for _, r in points_u.iterrows():
        name = str(r["name"])
        x = float(r["x"])
        y = float(r["y"])
        px, py = to_px(x, y)

        circles.append(
            f"""<circle class="target" cx="{px:.3f}" cy="{py:.3f}" r="7" data-name="{_escape_xml(name)}"></circle>"""
        )
        vectors.append(
            f"""<line class="vec"
                      x1="{px:.3f}" y1="{py:.3f}" x2="{px:.3f}" y2="{py:.3f}"
                      data-x1="{px:.3f}" data-y1="{py:.3f}"
                      data-vx="0" data-vy="0"
                      data-name="{_escape_xml(name)}"
                      data-dx="—" data-dy="—" data-dz="—" data-dp="—" data-ds="—"></line>"""
        )

    payload = {
        "enabled": True,
        "dates_ms": dates_ms,
        "default_i0": i0,
        "default_i1": i1,
        "x": x_map,
        "y": y_map,
        "z": z_map,
        "vx_unit": vx_unit,
        "vy_unit": vy_unit,
    }

    svg_inner = "".join(circles) + "".join(vectors)
    return _wrap_svg_html(
        width=width,
        height=height,
        svg_inner=svg_inner,
        with_vectors=True,
        base_px_per_m=base_px_per_m,
        vec_base_scale=vec_base_scale,
        tooltip_mode="deltas",
        custom_period_payload=payload,
        hide_scale_v=False,
        force_show_xy_scale=True,
    )


# ======================================================
# Streamlit page entrypoint (signature preserved)
# ======================================================
def render_topo_projections(selected_xlsx: str, xlsx_abs_path: str, force_key: int):
    # IMPORTANT: keep this module UI-neutral.
    # The caller (app.py) is responsible for the page title/subheader.

    root = _project_root()

    # 1) ✅ Priorité: fichier fourni par la page (xlsx_abs_path)
    workbook_path: Optional[Path] = None

    if xlsx_abs_path:
        p = Path(xlsx_abs_path).expanduser()
        if not p.is_absolute():
            p = (root / p).resolve()
        else:
            p = p.resolve()
        if p.exists() and p.is_file() and p.suffix.lower() == ".xlsx":
            workbook_path = p

    # 2) ✅ Sinon, tentative via selected_xlsx (si la page passe juste un nom/chemin relatif)
    if workbook_path is None and selected_xlsx:
        q = Path(selected_xlsx).expanduser()
        if not q.is_absolute():
            q1 = (root / q).resolve()
            if q1.exists() and q1.is_file() and q1.suffix.lower() == ".xlsx":
                workbook_path = q1
            else:
                # emplacements classiques dans GeoDashboard
                for base in [
                    root,
                    root / "data",
                    root / "data" / "common_data",
                    root / "data" / "topo",
                ]:
                    qq = (base / q.name).resolve()
                    if qq.exists() and qq.is_file() and qq.suffix.lower() == ".xlsx":
                        workbook_path = qq
                        break
        else:
            q = q.resolve()
            if q.exists() and q.is_file() and q.suffix.lower() == ".xlsx":
                workbook_path = q

    # 3) ✅ Fallback historique: Mesures Completes dans data/common_data
    if workbook_path is None:
        common_data_dir = root / "data" / "common_data"
        mesures_path = _find_mesures_completes_xlsx(common_data_dir)

        if mesures_path is None or not mesures_path.exists():
            st.error(
                "Fichier Excel introuvable.\n\n"
                f"- Chemin demandé (xlsx_abs_path) : {xlsx_abs_path or '—'}\n"
                f"- Sélection (selected_xlsx)      : {selected_xlsx or '—'}\n"
                f"- Fallback attendu               : {common_data_dir}/Mesures Completes*.xlsx"
            )
            return

        workbook_path = mesures_path

    # ✅ cache busting with force_key (kept)
    mtime = float(workbook_path.stat().st_mtime) + float(force_key or 0)

    try:
        dmin, dmax = _global_date_range(str(workbook_path), mtime)
    except Exception:
        dmin, dmax = None, None

    try:
        points = _load_target_xy_medians(str(workbook_path), mtime)
    except Exception as e:
        st.error("Impossible de lire le classeur (médianes).")
        st.caption(str(e))
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Cibles", "Mouvement connu", "Mois passé", "Semaine passée", "Période custom"]
    )

    with tab1:
        st.markdown("### Ensemble des cibles répertoriées")
        components.html(_build_svg_targets(points, width=950, height=700), height=720, scrolling=False)

        with st.expander("Diagnostic", expanded=False):
            st.write("Classeur utilisé :", str(workbook_path))
            st.write("selected_xlsx :", selected_xlsx or "—")
            st.write("xlsx_abs_path :", xlsx_abs_path or "—")
            try:
                st.write("Onglet (1er onglet lu) :", _detect_data_sheet(str(workbook_path)))
            except Exception as e:
                st.write("Onglet : (erreur)", str(e))
            st.write("Plage de dates globale :", _fmt_date(dmin), "→", _fmt_date(dmax))
            st.write("Nb cibles (médianes) :", int(points.shape[0]))
            st.dataframe(points, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown(f"### Mouvements connus au {_fmt_date(dmax)} depuis le {_fmt_date(dmin)}")
        try:
            deltas_all = _load_target_deltas_first_last(str(workbook_path), mtime)
        except Exception as e:
            st.error("Impossible de calculer le mouvement connu (first/last).")
            st.caption(str(e))
            return
        components.html(_build_svg_mouvement(points, deltas_all, width=950, height=700), height=720, scrolling=False)

    with tab3:
        start_30, end_30 = _window_bounds_from_last(dmax, 30)
        st.markdown(f"### Mouvements connus au cours du mois passé (du {_fmt_date(start_30)} au {_fmt_date(end_30)})")
        try:
            deltas_30 = _load_target_deltas_last_n_days(str(workbook_path), mtime, days=30)
        except Exception as e:
            st.error("Impossible de calculer le mouvement sur les 30 derniers jours.")
            st.caption(str(e))
            return
        components.html(_build_svg_mouvement(points, deltas_30, width=950, height=700), height=720, scrolling=False)

    with tab4:
        start_7, end_7 = _window_bounds_from_last(dmax, 7)
        st.markdown(f"### Mouvements connus au cours de la semaine passée (du {_fmt_date(start_7)} au {_fmt_date(end_7)})")
        try:
            deltas_7 = _load_target_deltas_last_n_days(str(workbook_path), mtime, days=7)
        except Exception as e:
            st.error("Impossible de calculer le mouvement sur les 7 derniers jours.")
            st.caption(str(e))
            return
        components.html(_build_svg_mouvement(points, deltas_7, width=950, height=700), height=720, scrolling=False)

    with tab5:
        st.markdown("### Mouvements sur une période personnalisée")
        try:
            html = _build_svg_mouvement_custom_live(points, str(workbook_path), mtime, width=950, height=700)
        except Exception as e:
            st.error("Impossible de préparer le mode live.")
            st.caption(str(e))
            return
        components.html(html, height=780, scrolling=False)
