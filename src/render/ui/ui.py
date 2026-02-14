from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components


def inject_css(styles_dir: Path):
    """
    Injecte le CSS global (styles/main.css) dans le DOM sans affichage texte.
    """
    css_path = styles_dir / "main.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        components.html(f"<style>{css}</style>", height=0, scrolling=False)
    else:
        st.warning("Fichier CSS introuvable : styles/main.css")


def sidebar_brand(assets_dir: Path, subtitle: str = ""):
    """
    Sidebar avec logo + nom + sous-titre.
    """
    logo = assets_dir / "logo.png"
    if logo.exists():
        st.sidebar.image(str(logo), use_container_width=True)
    else:
        st.sidebar.markdown("## ⚫")

    st.sidebar.markdown("### GeoDashBoard")
    if subtitle:
        st.sidebar.caption(subtitle)
    st.sidebar.divider()


def cache_button():
    """
    Bouton utilitaire pour vider les caches Streamlit.
    """
    if st.sidebar.button("Vider le cache"):
        st.cache_data.clear()
        if "force_key" in st.session_state:
            st.session_state.force_key += 1
        st.rerun()
