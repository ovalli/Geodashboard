from pathlib import Path
import streamlit as st

from streamlit_electron.streamlit_app.src.render.ui.ui import inject_css, sidebar_brand, cache_button
from streamlit_electron.streamlit_app.src.io.excel_utils import list_xlsx_in_folder
from streamlit_electron.streamlit_app.src.render.ui.app_core_topo import render_topo_projections


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
STYLES = ROOT / "styles"

st.set_page_config(
    page_title="GeoDashBoard • Projections topographiques",
    page_icon=str(ASSETS / "favicon.png") if (ASSETS / "favicon.png").exists() else "⚫",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css(STYLES)
sidebar_brand(ASSETS, subtitle="Projections topographiques")
cache_button()

xlsx_files = list_xlsx_in_folder(ROOT)
if not xlsx_files:
    st.error("Aucun fichier .xlsx trouvé dans le dossier de l'application.")
    st.stop()

if "force_key" not in st.session_state:
    st.session_state.force_key = 0

selected_xlsx = st.selectbox("Choisir le fichier Excel", xlsx_files, index=0)

xlsx_abs_path = str((ROOT / selected_xlsx).resolve())

if st.button("Lancer / Mettre à jour", type="primary"):
    st.session_state.force_key += 1
    st.rerun()

render_topo_projections(selected_xlsx, xlsx_abs_path, st.session_state.force_key)
