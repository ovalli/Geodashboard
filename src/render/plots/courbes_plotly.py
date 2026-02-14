from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple
import math

import pandas as pd
import plotly.graph_objects as go

from src.render.schema import utils_schema as dessinfonctions


# ============================================================
# META
# ============================================================

@dataclass(frozen=True)
class GraphMeta:
    idx: int
    title: str
    yunit: str


def graph_meta_list() -> List[GraphMeta]:
    return [
        GraphMeta(1, "Terrassement / Fond de fouille / PZ", Unite_Graph_1()),
        GraphMeta(2, "Tirants", Unite_Graph_2()),
        GraphMeta(3, "Butons", Unite_Graph_3()),
        GraphMeta(4, "Tête inclino", Unite_Graph_4()),
        GraphMeta(5, "Max tangentiel inclino", Unite_Graph_5()),
        GraphMeta(6, "Topographie : Mouvements Normaux", Unite_Graph_6()),
        GraphMeta(7, "Topographie : Mouvements Tangentiels", Unite_Graph_7()),
        GraphMeta(8, "Topographie : Mouvements Verticaux", Unite_Graph_8()),
    ]


# ============================================================
# UNITÉS (placeholders)
# ============================================================

def Unite_Graph_1() -> str: return "unité inconnue"
def Unite_Graph_2() -> str: return "unité inconnue"
def Unite_Graph_3() -> str: return "unité inconnue"
def Unite_Graph_4() -> str: return "unité inconnue"
def Unite_Graph_5() -> str: return "unité inconnue"
def Unite_Graph_6() -> str: return "unité inconnue"
def Unite_Graph_7() -> str: return "unité inconnue"
def Unite_Graph_8() -> str: return "unité inconnue"


# ============================================================
# PARAMS
# ============================================================

def SeuilATopo():
    return dessinfonctions.LignePara(17, 2)


def SeuiliTopo():
    return dessinfonctions.LignePara(18, 2)


# ============================================================
# COULEURS (doivent matcher le schéma)
# ============================================================

INCLINO_COLOR_AVAL = "red"
INCLINO_COLOR_AMONT = "rgb(80,150,255)"   # bleu clair
INCLINO_COLOR_AMONT2 = "rgb(20,110,60)"   # vert foncé


def CouleurInclino(sheet_name: str) -> str:
    low = (sheet_name or "").strip().lower()
    if low == "inclinoamont2":
        return INCLINO_COLOR_AMONT2
    if low == "inclinoamont":
        return INCLINO_COLOR_AMONT
    if low == "inclino":
        return INCLINO_COLOR_AVAL
    return "black"


def CouleurMaxTan(sheet_name: str) -> str:
    """
    D'après ta structure Excel :
      - MaxTan  -> rouge
      - MaxTan2 -> bleu clair
      - MaxTan3 -> vert foncé
    """
    low = (sheet_name or "").strip().lower()

    if low == "maxtan3" or low.endswith("maxtan3"):
        return INCLINO_COLOR_AMONT2
    if low == "maxtan2" or low.endswith("maxtan2"):
        return INCLINO_COLOR_AMONT
    if low == "maxtan" or low.endswith("maxtan"):
        return INCLINO_COLOR_AVAL

    # fallback (si variantes)
    if "maxtan3" in low or low.endswith("3"):
        return INCLINO_COLOR_AMONT2
    if "maxtan2" in low or low.endswith("2"):
        return INCLINO_COLOR_AMONT
    return INCLINO_COLOR_AVAL


# ============================================================
# STYLES
# ============================================================

STYLE_TERRASSEMENT = dict(color="rgb(133,83,15)", width=2)
STYLE_FF = dict(color="red", width=3, dash="dot")
STYLE_PZ = dict(color="rgb(38,196,236)", width=2)
STYLE_SEUIL_ROUGE = dict(color="rgb(255,0,0)", width=2.25, dash="dot")
STYLE_SEUIL_ORANGE = dict(color="rgb(255,140,0)", width=2.25, dash="dot")


# ============================================================
# LECTURE EXCEL (cache)
# ============================================================

@lru_cache(maxsize=128)
def _read_sheet(workbook_path: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(
        workbook_path,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl",
    )


def _sheet_exists(workbook_path: str, sheet_name: str) -> bool:
    try:
        _read_sheet(workbook_path, sheet_name)
        return True
    except Exception:
        return False


@lru_cache(maxsize=64)
def list_sheets_cached(workbook_path: str) -> List[str]:
    xls = pd.ExcelFile(workbook_path, engine="openpyxl")
    return list(xls.sheet_names)


# ============================================================
# OUTILS
# ============================================================

def _is_blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return str(v).strip() == ""


def _to_datetime_series(s: pd.Series) -> pd.Series:
    if s.empty:
        return pd.to_datetime(s)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime("1899-12-30") + pd.to_timedelta(s, unit="D")
    return pd.to_datetime(s, errors="coerce")


def _clean_xy(x: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    df = pd.DataFrame({"x": x, "y": y}).dropna(subset=["x", "y"])
    return df["x"], df["y"]


def _last_non_empty_row_in_colA(df: pd.DataFrame) -> int:
    if df.shape[1] < 1:
        return 0
    last = df.iloc[:, 0].last_valid_index()
    return int(last) if last is not None else 0


# ============================================================
# BORNES TEMPORELLES GLOBALES
# ============================================================

def _collect_all_time_series(workbook_path: str) -> List[pd.Series]:
    xs: List[pd.Series] = []

    if _sheet_exists(workbook_path, "Excav"):
        df = _read_sheet(workbook_path, "Excav")
        if df.shape[1] >= 1:
            xs.append(_to_datetime_series(df.iloc[1:, 0]))

    for sh in list_sheets_cached(workbook_path):
        if sh.lower().startswith("pz"):
            df = _read_sheet(workbook_path, sh)
            if df.shape[1] >= 1:
                xs.append(_to_datetime_series(df.iloc[1:, 0]))

    for sh in ("Tirants 2", "Butons 2"):
        if _sheet_exists(workbook_path, sh):
            df = _read_sheet(workbook_path, sh)
            if df.shape[1] >= 1:
                xs.append(_to_datetime_series(df.iloc[1:, 0]))

    if _sheet_exists(workbook_path, "Topo"):
        df = _read_sheet(workbook_path, "Topo")
        if df.shape[1] >= 1:
            xs.append(_to_datetime_series(df.iloc[1:, 0]))

    for sh in list_sheets_cached(workbook_path):
        low = sh.lower()
        if low in ("inclino", "inclinoamont", "inclinoamont2"):
            df = _read_sheet(workbook_path, sh)
            if df.shape[0] >= 1 and df.shape[1] >= 2:
                xs.append(_to_datetime_series(pd.Series(df.iloc[0, 1:])))

    for sh in list_sheets_cached(workbook_path):
        if sh.lower().startswith("maxtan"):
            df = _read_sheet(workbook_path, sh)
            if df.shape[0] >= 1 and df.shape[1] >= 1:
                xs.append(_to_datetime_series(pd.Series(df.iloc[0, :])))

    return xs


def compute_global_time_bounds(workbook_path: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    xs = _collect_all_time_series(workbook_path)
    if not xs:
        return None, None
    allx = pd.concat(xs, ignore_index=True).dropna()
    if allx.empty:
        return None, None
    return allx.min(), allx.max()


# ============================================================
# LAYOUT PLOTLY
# ============================================================

def _apply_layout(fig: go.Figure, title: str, yunit: str, tmin, tmax) -> go.Figure:
    fig.update_layout(
        title=title,
        margin=dict(l=30, r=20, t=40, b=70),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.25,
            yanchor="top",
        ),
    )
    fig.update_xaxes(title_text="Date", range=[tmin, tmax] if (tmin is not None and tmax is not None) else None)
    fig.update_yaxes(title_text=yunit or "unité inconnue")
    return fig


# ============================================================
# HARMONISATION ECHELLE Y (Graphs 4..8)
# ============================================================

def _floor5(v: float) -> float:
    return float(math.floor(v / 5.0) * 5)


def _ceil5(v: float) -> float:
    return float(math.ceil(v / 5.0) * 5)


def _fig_min_max_y(fig: go.Figure) -> Tuple[Optional[float], Optional[float]]:
    ys = []
    for tr in fig.data:
        if not hasattr(tr, "y") or tr.y is None:
            continue
        s = pd.to_numeric(pd.Series(list(tr.y)), errors="coerce").dropna()
        if not s.empty:
            ys.append(s)
    if not ys:
        return None, None
    all_y = pd.concat(ys, ignore_index=True).dropna()
    if all_y.empty:
        return None, None
    return float(all_y.min()), float(all_y.max())


def _snap_range_same_span(min_raw: float, max_raw: float, span: float) -> Tuple[float, float]:
    """
    Fenêtre [y0,y1] de largeur span, multiples de 5,
    déplacée par pas de 5 si nécessaire pour contenir toutes les valeurs.
    """
    c = 0.5 * (min_raw + max_raw)
    y0 = _floor5(c - 0.5 * span)
    y1 = y0 + span

    step = 5.0
    for _ in range(2000):
        if min_raw < y0:
            y0 -= step
            y1 -= step
            continue
        if max_raw > y1:
            y0 += step
            y1 += step
            continue
        break

    return float(y0), float(y1)


def _harmonize_y_scales_4_to_8(figs: List[go.Figure]) -> List[go.Figure]:
    """
    Applique une même "échelle" verticale (span commun) à toutes les figs fournies,
    avec bornes arrondies à 5, sans couper de valeurs.
    """
    # calc mins/maxs + spans
    ranges = []
    spans = []
    for f in figs:
        mn, mx = _fig_min_max_y(f)
        if mn is None or mx is None:
            ranges.append((None, None))
            continue
        ranges.append((mn, mx))
        spans.append(mx - mn)

    if not spans:
        return figs

    span_common = _ceil5(max(spans))
    if span_common <= 0:
        span_common = 5.0

    out = []
    for f, (mn, mx) in zip(figs, ranges):
        if mn is None or mx is None:
            out.append(f)
            continue

        y0, y1 = _snap_range_same_span(mn, mx, span_common)
        y0 = _floor5(y0)
        y1 = y0 + span_common

        f.update_yaxes(range=[y0, y1])
        out.append(f)

    return out


# ============================================================
# BUILDERS
# ============================================================

def build_graph_1(workbook_path: str, tmin, tmax, meta: GraphMeta) -> Optional[go.Figure]:
    fig = go.Figure()
    has_any = False

    if _sheet_exists(workbook_path, "Excav"):
        df = _read_sheet(workbook_path, "Excav")
        if df.shape[1] >= 3 and df.shape[0] >= 3:
            x = _to_datetime_series(df.iloc[1:, 0])
            y_terr = pd.to_numeric(df.iloc[1:, 1], errors="coerce")
            y_ff = pd.to_numeric(df.iloc[1:, 2], errors="coerce")

            xx, yy = _clean_xy(x, y_terr)
            if not xx.empty:
                fig.add_trace(go.Scatter(
                    x=xx, y=yy,
                    mode="lines+markers",
                    name="Niveau terrassement",
                    line=STYLE_TERRASSEMENT,
                    marker=dict(size=6),
                ))
                has_any = True

            xx, yy = _clean_xy(x, y_ff)
            if not xx.empty:
                fig.add_trace(go.Scatter(
                    x=xx, y=yy,
                    mode="lines",
                    name="Fond de fouille déf.",
                    line=STYLE_FF,
                ))
                has_any = True

    for sh in list_sheets_cached(workbook_path):
        if sh.lower().startswith("pz"):
            df = _read_sheet(workbook_path, sh)
            if df.shape[1] >= 2 and df.shape[0] >= 2:
                x = _to_datetime_series(df.iloc[1:, 0])
                y = pd.to_numeric(df.iloc[1:, 1], errors="coerce")
                xx, yy = _clean_xy(x, y)
                if not xx.empty:
                    fig.add_trace(go.Scatter(
                        x=xx, y=yy,
                        mode="lines+markers",
                        name=sh,
                        line=STYLE_PZ,
                        marker=dict(size=5),
                    ))
                    has_any = True

    if not has_any:
        return None
    return _apply_layout(fig, meta.title, meta.yunit, tmin, tmax)


def build_graph_2(workbook_path: str, tmin, tmax, meta: GraphMeta) -> Optional[go.Figure]:
    sheet = "Tirants 2"
    if not _sheet_exists(workbook_path, sheet):
        return None

    df = _read_sheet(workbook_path, sheet)
    if df.shape[0] < 3 or df.shape[1] < 2:
        return None

    last = _last_non_empty_row_in_colA(df)
    if last < 1:
        return None

    x = _to_datetime_series(df.iloc[1:last + 1, 0])
    fig = go.Figure()
    has_any = False

    col = 1
    while col < df.shape[1]:
        name = df.iat[0, col]
        if _is_blank(name):
            break

        y = pd.to_numeric(df.iloc[1:last + 1, col], errors="coerce")
        xx, yy = _clean_xy(x, y)
        if not xx.empty:
            fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name=str(name), line=dict(width=2)))
            has_any = True
        col += 1

    if not has_any:
        return None
    return _apply_layout(fig, meta.title, meta.yunit, tmin, tmax)


def build_graph_3(workbook_path: str, tmin, tmax, meta: GraphMeta) -> Optional[go.Figure]:
    sheet = "Butons 2"
    if not _sheet_exists(workbook_path, sheet):
        return None

    df = _read_sheet(workbook_path, sheet)
    if df.shape[0] < 3 or df.shape[1] < 2:
        return None

    last = _last_non_empty_row_in_colA(df)
    if last < 1:
        return None

    x = _to_datetime_series(df.iloc[1:last + 1, 0])
    fig = go.Figure()
    has_any = False

    col = 1
    while col < df.shape[1]:
        name = df.iat[0, col]
        if _is_blank(name):
            break

        y = pd.to_numeric(df.iloc[1:last + 1, col], errors="coerce")
        xx, yy = _clean_xy(x, y)
        if not xx.empty:
            fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name=str(name), line=dict(width=2)))
            has_any = True
        col += 1

    if not has_any:
        return None
    return _apply_layout(fig, meta.title, meta.yunit, tmin, tmax)


def build_graph_4(workbook_path: str, tmin, tmax, meta: GraphMeta) -> Optional[go.Figure]:
    fig = go.Figure()
    has_any = False

    for sh in list_sheets_cached(workbook_path):
        low = sh.lower().strip()
        if low not in ("inclino", "inclinoamont", "inclinoamont2"):
            continue

        df = _read_sheet(workbook_path, sh)
        if df.shape[0] < 2 or df.shape[1] < 2:
            continue

        x = _to_datetime_series(pd.Series(df.iloc[0, 1:]))
        y = pd.to_numeric(pd.Series(df.iloc[1, 1:]), errors="coerce")
        xx, yy = _clean_xy(x, y)
        if xx.empty:
            continue

        fig.add_trace(go.Scatter(
            x=xx, y=yy,
            mode="lines+markers",
            name=sh,
            line=dict(color=CouleurInclino(sh), width=2),
            marker=dict(size=6),
        ))
        has_any = True

    if not has_any:
        return None
    return _apply_layout(fig, meta.title, meta.yunit, tmin, tmax)


def build_graph_5(workbook_path: str, tmin, tmax, meta: GraphMeta) -> Optional[go.Figure]:
    fig = go.Figure()
    has_any = False

    for sh in list_sheets_cached(workbook_path):
        low = sh.strip().lower()
        if not low.startswith("maxtan"):
            continue

        df = _read_sheet(workbook_path, sh)
        if df.shape[0] < 2 or df.shape[1] < 1:
            continue

        x = _to_datetime_series(pd.Series(df.iloc[0, :]))
        y = pd.to_numeric(pd.Series(df.iloc[1, :]), errors="coerce")
        xx, yy = _clean_xy(x, y)
        if xx.empty:
            continue

        fig.add_trace(go.Scatter(
            x=xx, y=yy,
            mode="lines+markers",
            name=sh,
            line=dict(color=CouleurMaxTan(sh), width=2),
            marker=dict(size=6),
        ))
        has_any = True

    if not has_any:
        return None

    return _apply_layout(fig, meta.title, meta.yunit, tmin, tmax)


def build_graph_6_7_8(
    workbook_path: str,
    tmin,
    tmax,
    meta6: GraphMeta,
    meta7: GraphMeta,
    meta8: GraphMeta
) -> List[go.Figure]:
    """
    Graphs topo 6/7/8.
    Ici on NE force PAS les ranges Y : on laisse l'harmonisation 4..8 le faire ensuite.
    """
    if not _sheet_exists(workbook_path, "Topo"):
        return []

    df = _read_sheet(workbook_path, "Topo")
    if df.shape[0] < 3 or df.shape[1] < 12:
        return []

    last = _last_non_empty_row_in_colA(df)
    if last < 1:
        return []

    x = _to_datetime_series(df.iloc[1:last + 1, 0])

    def build_one(component_idx: int, meta: GraphMeta) -> Optional[go.Figure]:
        fig = go.Figure()
        has_any = False

        k = 5
        while k < df.shape[1]:
            nom = df.iat[0, k]
            if _is_blank(nom):
                break

            col_val = k + 3 + component_idx
            if col_val >= df.shape[1]:
                break

            y = pd.to_numeric(df.iloc[1:last + 1, col_val], errors="coerce")
            xx, yy = _clean_xy(x, y)
            if not xx.empty:
                fig.add_trace(go.Scatter(
                    x=xx, y=yy,
                    mode="lines+markers",
                    name=str(nom),
                    marker=dict(size=5),
                ))
                has_any = True

            k += 6

        if not has_any:
            return None

        return _apply_layout(fig, meta.title, meta.yunit, tmin, tmax)

    figs: List[go.Figure] = []
    for f in (build_one(0, meta6), build_one(1, meta7), build_one(2, meta8)):
        if f is not None:
            figs.append(f)
    return figs


# ============================================================
# API
# ============================================================

def build_all_figures(workbook_path: Optional[str] = None) -> List[go.Figure]:
    """
    Renvoie les figures Plotly dans l'ordre 1..8 (mais certaines peuvent manquer).
    On harmonise l'échelle Y de manière intelligente sur les graphs 4..8 disponibles.
    """
    if workbook_path is None:
        workbook_path = dessinfonctions.classeurCoupe()

    tmin, tmax = compute_global_time_bounds(workbook_path)
    metas = graph_meta_list()

    figs: List[go.Figure] = []

    f1 = build_graph_1(workbook_path, tmin, tmax, metas[0])
    if f1 is not None:
        figs.append(f1)

    f2 = build_graph_2(workbook_path, tmin, tmax, metas[1])
    if f2 is not None:
        figs.append(f2)

    f3 = build_graph_3(workbook_path, tmin, tmax, metas[2])
    if f3 is not None:
        figs.append(f3)

    # --- Construire 4..8, puis harmoniser ensemble ---
    f4 = build_graph_4(workbook_path, tmin, tmax, metas[3])
    f5 = build_graph_5(workbook_path, tmin, tmax, metas[4])
    topo_figs = build_graph_6_7_8(workbook_path, tmin, tmax, metas[5], metas[6], metas[7])

    figs_4_to_8: List[go.Figure] = []
    if f4 is not None:
        figs_4_to_8.append(f4)
    if f5 is not None:
        figs_4_to_8.append(f5)
    figs_4_to_8.extend(topo_figs)

    if figs_4_to_8:
        figs_4_to_8 = _harmonize_y_scales_4_to_8(figs_4_to_8)

    # garder l'ordre "logique" : 4, 5, puis 6/7/8
    if f4 is not None:
        figs.append(figs_4_to_8.pop(0) if figs_4_to_8 else f4)
    if f5 is not None:
        figs.append(figs_4_to_8.pop(0) if figs_4_to_8 else f5)
    # reste = topo
    figs.extend(figs_4_to_8)

    return figs
