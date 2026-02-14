import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.pipeline.pipeline import build_shapes
from src.render.render_html import compute_canvas_bounds, render_shapes_html
from src.render.plots.courbes_plotly import build_all_figures
from src.render.schema import utils_schema as dessinfonctions


# ======================================================
# Helpers
# ======================================================
def audit_sheets(abs_xlsx_path: str) -> dict:
    try:
        xls = pd.ExcelFile(abs_xlsx_path, engine="openpyxl")
        sheets = set(xls.sheet_names)
    except Exception:
        return {"_error": True, "sheets": set()}

    return {
        "_error": False,
        "sheets": sheets,
        "geol": "Geol" in sheets,
        "inclino": any(s.lower().startswith("inclino") for s in sheets),
        "topo": "Topo" in sheets,
        "excav": "Excav" in sheets,
        "pz": any(s.lower().startswith("pz") for s in sheets),
    }


def _code_mtime() -> float:
    """
    Invalide le cache quand on modifie le code (dessin.py, render_html.py, etc.).
    """
    base_dir = Path(__file__).resolve().parents[2]  # racine projet
    candidates = [
        base_dir / "src" / "render" / "schema" / "dessin.py",
        base_dir / "src" / "render" / "render_html.py",
        base_dir / "src" / "pipeline" / "pipeline.py",
        base_dir / "src" / "render" / "plots" / "courbes_plotly.py",
    ]
    mtimes = []
    for p in candidates:
        try:
            if p.exists():
                mtimes.append(p.stat().st_mtime)
        except Exception:
            pass
    return max(mtimes) if mtimes else 0.0


@st.cache_data(show_spinner=False)
def build_shapes_cached(xlsx_path: str, mtime: float, code_mtime: float, force_key: int):
    # code_mtime est là pour invalider le cache
    return build_shapes()


@st.cache_data(show_spinner=False)
def build_figs_cached(xlsx_path: str, mtime: float, code_mtime: float, force_key: int):
    return build_all_figures(workbook_path=xlsx_path)


def _resolve_xlsx_path(selected_xlsx: str) -> str:
    if not selected_xlsx:
        return ""

    if os.path.isabs(selected_xlsx) and os.path.exists(selected_xlsx):
        return selected_xlsx

    base_dir = Path(__file__).resolve().parents[2]
    excel_dir = base_dir / "data" / "excel_data"
    return str((excel_dir / selected_xlsx).resolve())


# ======================================================
# Main
# ======================================================
def render_coupes(selected_xlsx: str, force_key: int):
    xlsx_path = _resolve_xlsx_path(selected_xlsx)
    if not xlsx_path or not os.path.exists(xlsx_path):
        return  # silence

    mtime = os.path.getmtime(xlsx_path)
    code_mtime = _code_mtime()

    status = audit_sheets(xlsx_path)
    if status.get("_error"):
        return  # silence

    # Dire au moteur quel classeur utiliser
    dessinfonctions.set_classeur(xlsx_path)

    # ==================================================
    # SCHÉMA (uniquement si Geol présente)
    # ==================================================
    if status.get("geol", False):
        try:
            shapes = build_shapes_cached(xlsx_path, mtime, code_mtime, force_key)
            offset_x, offset_y, w, h = compute_canvas_bounds(
                shapes, margin=24, min_width=640, min_height=360
            )
            html_doc = render_shapes_html(
                shapes, width=w, height=h, offset_x=offset_x, offset_y=offset_y
            )
            components.html(html_doc, height=h + 10, scrolling=False)
        except Exception:
            # Si jamais le schéma plante sur un fichier, on n'empêche pas les courbes
            st.info("Schéma non disponible pour ce fichier.")
    else:
        # ✅ On ne coupe PLUS les courbes : on masque seulement le schéma
        st.info("Feuille Geol absente : schéma non disponible pour ce fichier.")

    # ==================================================
    # COURBES (toujours, même si Geol absente)
    # ==================================================
    try:
        figs = build_figs_cached(xlsx_path, mtime, code_mtime, force_key)
    except Exception:
        figs = None

    if not figs:
        return

    for fig in figs:
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
