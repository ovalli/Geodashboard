import pathlib
print("APP:", pathlib.Path(__file__).resolve())

from pathlib import Path
import os
import zipfile
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu

from src.io.excel_utils import list_xlsx_in_folder
from src.ui.app_core_coupes import render_coupes
from src.ui.app_core_topo import render_topo_projections
from src.ui.app_core_selection_cible import render_selection_cibles

from src.ui.app_core_import import render_import
from src.io.import_utils import find_mesures_completes_xlsx

# ✅ nouveau core paramétrage
from src.ui.app_core_parametrage import render_parametrage


# ======================================================
# PATHS & CONFIG
# ======================================================
ROOT = Path(__file__).parent.resolve()
ASSETS = ROOT / "assets"
STYLES = ROOT / "styles"

DATA_DIR = ROOT / "data"
COMMON_DATA = DATA_DIR / "common_data"


# ======================================================
# ENV (local vs cloud)
# ======================================================
GEODASH_ENV = os.getenv("GEODASH_ENV", "local").strip().lower()
IS_LOCAL = GEODASH_ENV == "local"

st.set_page_config(
    page_title="GeoDashBoard",
    page_icon=str(ASSETS / "favicon.png") if (ASSETS / "favicon.png").exists() else "⚫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================
# BACKUP SILENCIEUX (LOCAL ONLY)
# ======================================================
def _should_exclude_path(rel_path: str) -> bool:
    parts = rel_path.split(os.sep)
    exclude_dirs = {
        ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
        ".venv", "venv", "env", "node_modules",
        ".idea", ".vscode",
        "backup", "_backups",
    }
    return any(p in exclude_dirs for p in parts)


def _create_project_backup_zip(project_root: Path, backups_dir: Path, keep_last: int = 30):
    backups_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = backups_dir / f"GeoDashboard_FULL_{ts}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder, subfolders, filenames in os.walk(project_root):
            rel_folder = os.path.relpath(folder, project_root)
            if rel_folder == ".":
                rel_folder = ""

            subfolders[:] = [d for d in subfolders if not _should_exclude_path(d)]

            for fn in filenames:
                abs_path = os.path.join(folder, fn)
                rel_path = os.path.normpath(os.path.join(rel_folder, fn))
                if _should_exclude_path(rel_path):
                    continue
                zf.write(abs_path, rel_path)

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


if IS_LOCAL and ("did_startup_backup" not in st.session_state):
    st.session_state["did_startup_backup"] = True
    try:
        DESKTOP_BACKUP_DIR = Path(os.path.expanduser("~/Desktop/backup")).resolve()
        _create_project_backup_zip(ROOT, DESKTOP_BACKUP_DIR)
    except Exception:
        pass


# ======================================================
# CSS
# ======================================================
css_path = STYLES / "main.css"
if css_path.exists():
    components.html(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        height=0,
        scrolling=False,
    )


# ======================================================
# UTILS
# ======================================================
def pretty_xlsx_name(filename: str) -> str:
    name = filename[:-5] if filename.lower().endswith(".xlsx") else filename
    return name[8:] if len(name) > 8 else name


def _get_abs_path(display_map: dict, display_name: str) -> str:
    fn = display_map.get(display_name)
    if not fn:
        return ""
    return str((DATA_DIR / fn).resolve())


# ======================================================
# DATA : LISTE EXCEL (utilisée pour "Coupes depuis Excel")
# ======================================================
xlsx_files = list_xlsx_in_folder(DATA_DIR)

display_map: dict[str, str] = {}
counter: dict[str, int] = {}
for fn in xlsx_files:
    disp = pretty_xlsx_name(fn)
    if disp in counter:
        counter[disp] += 1
        disp = f"{disp} ({counter[disp]})"
    else:
        counter[disp] = 1
    display_map[disp] = fn

display_names = list(display_map.keys())


# ======================================================
# NAV PRINCIPALE
# ======================================================
NAV_COUPES_EXCEL = "Coupes depuis Excel"
NAV_CST = "Coupes"  # ✅ on garde le menu
NAV_3D = "3D"
NAV_PARAM = "Paramétrage"
NAV_TOPO = "Projections topographiques"
NAV_SEL_CIBLES = "Sélection des zones"
NAV_INCL = "Inclinomètres"
NAV_IMPORT = "Import"
NAV_EXPORT = "Export"
NAV_PROFIL = "Profil"

main_nav_items = [
    NAV_COUPES_EXCEL,
    NAV_CST,          # ✅ rétabli
    NAV_3D,
    NAV_PARAM,
    NAV_TOPO,
    NAV_SEL_CIBLES,
    NAV_INCL,
    NAV_IMPORT,
    NAV_EXPORT,
    NAV_PROFIL,
]

main_nav_icons = [
    "file-earmark-spreadsheet",
    "file-earmark-text",   # ✅ comme avant
    "badge-3d",
    "sliders",
    "map",
    "bounding-box",
    "activity",
    "cloud-upload",
    "download",
    "person-circle",
]

if "main_nav" not in st.session_state:
    st.session_state["main_nav"] = NAV_COUPES_EXCEL


# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    logo_path = ASSETS / "logo.png"
    if logo_path.exists():
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(str(logo_path), width=140)
    else:
        st.markdown("## ⚫")

    st.divider()

    selected_page = option_menu(
        menu_title=None,
        options=main_nav_items,
        icons=main_nav_icons,
        orientation="vertical",
        key="main_nav",
        styles={
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
        },
    )


# ======================================================
# ROUTING
# ======================================================
if selected_page == NAV_IMPORT:
    render_import(COMMON_DATA)
    st.stop()

if selected_page == NAV_EXPORT:
    st.subheader("Export")
    st.stop()

if selected_page == NAV_PROFIL:
    st.subheader("Profil")
    st.stop()

# --- Coupes depuis Excel
if selected_page == NAV_COUPES_EXCEL:
    if not display_names:
        st.error("Aucun fichier .xlsx trouvé dans data/")
        st.stop()

    st.subheader("Coupes depuis Excel")

    selected_coupe = option_menu(
        menu_title=None,
        options=display_names,
        icons=[""] * len(display_names),
        orientation="horizontal",
        key="coupes_tabs",
        styles={
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
        },
    )

    xlsx_abs_path = _get_abs_path(display_map, selected_coupe)
    if not xlsx_abs_path:
        st.error("Fichier introuvable.")
        st.stop()

    render_coupes(xlsx_abs_path, 0)
    st.stop()

# --- ✅ Coupes depuis CST (menu conservé, backend JSON-only)
if selected_page == NAV_CST:
    # On conserve la page/label, mais on ne dépend plus du fichier CST.
    st.subheader("Coupes depuis CST")

    try:
        from src.ui.app_core_coupes_cst import render_coupes_cst
        render_coupes_cst()  # ⚠️ doit être JSON-only côté module
    except Exception as e:
        # fallback propre : on n'empêche pas l'app de tourner
        st.info("Cette page doit désormais lire le JSON (plus de Charges sur Trame.xlsx).")
        st.error("💥 Page 'Coupes depuis CST' cassée (erreur réelle ci-dessous) :")
        st.exception(e)

        # fallback utile : on t'amène vers le paramétrage (JSON)
        st.divider()
        st.caption("Fallback : affichage Paramètrage (JSON).")
        try:
            render_parametrage()
        except Exception as e2:
            st.error("💥 Fallback Paramètrage cassé :")
            st.exception(e2)

    st.stop()

# --- 3D
if selected_page == NAV_3D:
    try:
        import src.ui.app_core_3d as app_core_3d
    except Exception as e:
        st.error("💥 3D est cassé (erreur réelle ci-dessous) :")
        st.exception(e)
        st.stop()

    if not hasattr(app_core_3d, "render_3d"):
        st.error("💥 3D est cassé : render_3d() introuvable dans src/ui/app_core_3d.py")
        st.stop()

    try:
        app_core_3d.render_3d()
    except Exception as e:
        st.error("💥 3D est cassé (erreur réelle ci-dessous) :")
        st.exception(e)
        st.stop()

    st.stop()

# --- ✅ Paramètrage (JSON-only)
if selected_page == NAV_PARAM:
    render_parametrage()
    st.stop()

# --- Topo
if selected_page == NAV_TOPO:
    st.subheader("Projections topographiques")

    mesures_path = find_mesures_completes_xlsx(COMMON_DATA)
    if mesures_path is None or not mesures_path.exists():
        st.error(f"Fichier Mesures Completes introuvable dans : {COMMON_DATA}")
        st.stop()

    render_topo_projections("GLOBAL", str(mesures_path), 0)
    st.stop()

# --- Sélection des zones (JSON-only)
if selected_page == NAV_SEL_CIBLES:
    st.subheader("Sélection des zones")
    render_selection_cibles("GLOBAL", None, 0)
    st.stop()

# --- Inclino
if selected_page == NAV_INCL:
    st.subheader("Inclinomètres")
    st.info("Page en cours de construction")
    st.stop()

st.error("Navigation inconnue.")
