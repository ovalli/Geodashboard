from __future__ import annotations

from pathlib import Path
import os
import zipfile
from datetime import datetime

import streamlit as st
from streamlit_option_menu import option_menu

from src.io.excel_utils import list_xlsx_in_folder
from src.ui.app_core_coupes import render_coupes
from src.ui.app_core_topo import render_topo_projections
from src.ui.app_core_selection_cible import render_selection_cibles
from src.ui.app_core_import import render_import
from src.io.import_utils import find_mesures_completes_xlsx
from src.ui.app_core_parametrage import render_parametrage
from src.ui.app_core_terrassements import render_terrassements


# ======================================================
# PATHS & CONFIG
# ======================================================
ROOT = Path(__file__).parent.resolve()
ASSETS = ROOT / "assets"
STYLES = ROOT / "styles"
DATA_DIR = ROOT / "data"
COMMON_DATA = DATA_DIR / "common_data"

print("APP:", Path(__file__).resolve())

GEODASH_ENV = os.getenv("GEODASH_ENV", "local").strip().lower()
IS_LOCAL = GEODASH_ENV == "local"

st.set_page_config(
    page_title="GeoDashBoard",
    page_icon=str(ASSETS / "favicon.png") if (ASSETS / "favicon.png").exists() else "⚫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================
# STARTUP: BACKUP LOCAL SILENCIEUX
# ======================================================
EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".venv", "venv", "env", "node_modules",
    ".idea", ".vscode",
    "backup", "_backups",
}


def _should_exclude(rel_path: str) -> bool:
    return any(p in EXCLUDE_DIRS for p in rel_path.split(os.sep))


def _startup_backup_local(project_root: Path, keep_last: int = 30) -> None:
    if not IS_LOCAL or st.session_state.get("did_startup_backup"):
        return
    st.session_state["did_startup_backup"] = True

    try:
        backups_dir = Path(os.path.expanduser("~/Desktop/backup")).resolve()
        backups_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = backups_dir / f"GeoDashboard_FULL_{ts}.zip"

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for folder, subfolders, filenames in os.walk(project_root):
                rel_folder = os.path.relpath(folder, project_root)
                rel_folder = "" if rel_folder == "." else rel_folder

                subfolders[:] = [d for d in subfolders if not _should_exclude(d)]

                for fn in filenames:
                    abs_path = os.path.join(folder, fn)
                    rel_path = os.path.normpath(os.path.join(rel_folder, fn))
                    if _should_exclude(rel_path):
                        continue
                    zf.write(abs_path, rel_path)

        # prune
        zips = sorted(
            backups_dir.glob("GeoDashboard_FULL_*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in zips[keep_last:]:
            try:
                old.unlink()
            except Exception:
                pass
    except Exception:
        pass


_startup_backup_local(ROOT)


# ======================================================
# CSS
# ======================================================
def _inject_css(styles_dir: Path) -> None:
    css_path = styles_dir / "main.css"
    if not css_path.exists():
        return
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


_inject_css(STYLES)


# ======================================================
# COUPES DEPUIS EXCEL: LISTE FICHIERS
# ======================================================
def _pretty_xlsx_name(filename: str) -> str:
    name = filename[:-5] if filename.lower().endswith(".xlsx") else filename
    return name[8:] if len(name) > 8 else name


def _build_display_map(xlsx_files: list[str]) -> dict[str, str]:
    display_map: dict[str, str] = {}
    counts: dict[str, int] = {}
    for fn in xlsx_files:
        disp = _pretty_xlsx_name(fn)
        n = counts.get(disp, 0) + 1
        counts[disp] = n
        if n > 1:
            disp = f"{disp} ({n})"
        display_map[disp] = fn
    return display_map


def _abs_xlsx_path(display_map: dict[str, str], display_name: str) -> str:
    fn = display_map.get(display_name)
    return str((DATA_DIR / fn).resolve()) if fn else ""


XLSX_FILES = list_xlsx_in_folder(DATA_DIR)
DISPLAY_MAP = _build_display_map(XLSX_FILES)
DISPLAY_NAMES = list(DISPLAY_MAP.keys())


# ======================================================
# NAV
# ======================================================
NAV = {
    "Coupes": "file-earmark-text",
    "2D": "layers",
    "3D": "badge-3d",
    "Terrassements": "truck-flatbed",
    "Paramètres chantier": "sliders",
    "Projections topographiques": "map",
    "Sélection des zones": "bounding-box",
    "Inclinomètres": "activity",
    "Import": "cloud-upload",
    "Export": "download",
    "Profil": "person-circle",
    "Coupes depuis Excel": "file-earmark-spreadsheet",
}

SIDEBAR_STYLES = {
    "container": {"padding": "0!important"},
    "icon": {"font-size": "0.95rem", "opacity": "0.85"},
    "nav-link": {
        "font-size": "0.92rem",
        "padding": "6px 10px",
        "border-radius": "0.5rem",
        "transition": "background-color 120ms ease",
    },
    "nav-link-hover": {"background-color": "rgba(0,0,0,0.06)"},
    "nav-link-selected": {
        "background-color": "rgba(0,0,0,0.10)",
        "font-weight": "700",
        "border-radius": "0.5rem",
    },
}

TABS_STYLES = {
    "container": {"padding": "0!important"},
    "icon": {"display": "none"},
    "nav-link": {
        "font-size": "0.90rem",
        "padding": "8px 10px",
        "margin": "0 4px 0 0",
        "border-radius": "0.6rem",
        "white-space": "nowrap",
        "transition": "background-color 120ms ease",
    },
    "nav-link-hover": {"background-color": "rgba(0,0,0,0.06)"},
    "nav-link-selected": {
        "background-color": "rgba(0,0,0,0.10)",
        "font-weight": "700",
        "border-radius": "0.6rem",
    },
}


def _sidebar_nav() -> str:
    with st.sidebar:
        logo = ASSETS / "logo.png"
        if logo.exists():
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image(str(logo), width=140)
        else:
            st.markdown("## ⚫")

        st.divider()

        return option_menu(
            menu_title=None,
            options=list(NAV.keys()),
            icons=list(NAV.values()),
            orientation="vertical",
            key="main_nav",
            styles=SIDEBAR_STYLES,
        )


# ======================================================
# PAGES (✅ sans subheader)
# ======================================================
def page_import() -> None:
    render_import(COMMON_DATA)


def page_export() -> None:
    pass


def page_profil() -> None:
    pass


def page_coupes_excel() -> None:
    if not DISPLAY_NAMES:
        st.error("Aucun fichier .xlsx trouvé dans data/")
        return

    selected_coupe = option_menu(
        menu_title=None,
        options=DISPLAY_NAMES,
        icons=[""] * len(DISPLAY_NAMES),
        orientation="horizontal",
        key="coupes_tabs",
        styles=TABS_STYLES,
    )

    xlsx_abs = _abs_xlsx_path(DISPLAY_MAP, selected_coupe)
    if not xlsx_abs:
        st.error("Fichier introuvable.")
        return

    render_coupes(xlsx_abs, 0)


def page_coupes_cst() -> None:
    try:
        from src.ui.app_core_coupes_cst import render_coupes_cst
        render_coupes_cst()
        return
    except Exception as e:
        st.info("Cette page doit désormais lire le JSON (plus de Charges sur Trame.xlsx).")
        st.error("💥 Page 'Coupes depuis CST' cassée (erreur réelle ci-dessous) :")
        st.exception(e)

    st.divider()
    try:
        render_parametrage()
    except Exception as e2:
        st.error("💥 Fallback Paramètrage cassé :")
        st.exception(e2)


def page_2d() -> None:
    # ✅ Vide pour l’instant (placeholder)
    # (On remplira plus tard)
    pass


def page_3d() -> None:
    try:
        import src.ui.app_core_3d as app_core_3d
    except Exception as e:
        st.error("💥 3D est cassé (erreur réelle ci-dessous) :")
        st.exception(e)
        return

    if not hasattr(app_core_3d, "render_3d"):
        st.error("💥 3D est cassé : render_3d() introuvable dans src/ui/app_core_3d.py")
        return

    try:
        app_core_3d.render_3d()
    except Exception as e:
        st.error("💥 3D est cassé (erreur réelle ci-dessous) :")
        st.exception(e)


def page_terrassements() -> None:
    try:
        render_terrassements()
    except Exception as e:
        st.error("💥 Terrassements est cassé (erreur réelle ci-dessous) :")
        st.exception(e)


def page_parametres_chantier() -> None:
    render_parametrage()


def page_topo() -> None:
    mesures_path = find_mesures_completes_xlsx(COMMON_DATA)
    if mesures_path is None or not mesures_path.exists():
        st.error(f"Fichier Mesures Completes introuvable dans : {COMMON_DATA}")
        return
    render_topo_projections("GLOBAL", str(mesures_path), 0)


def page_selection_zones() -> None:
    render_selection_cibles("GLOBAL", None, 0)


def page_inclino() -> None:
    st.info("Page en cours de construction")


ROUTES: dict[str, callable] = {
    "Import": page_import,
    "Export": page_export,
    "Profil": page_profil,
    "Coupes": page_coupes_cst,
    "2D": page_2d,  # ✅ nouveau placeholder
    "3D": page_3d,
    "Terrassements": page_terrassements,
    "Paramètres chantier": page_parametres_chantier,
    "Projections topographiques": page_topo,
    "Sélection des zones": page_selection_zones,
    "Inclinomètres": page_inclino,
    "Coupes depuis Excel": page_coupes_excel,
}


# ======================================================
# RUN
# ======================================================
selected_page = _sidebar_nav()
handler = ROUTES.get(selected_page)

if not handler:
    st.error("Navigation inconnue.")
else:
    handler()
    st.stop()
