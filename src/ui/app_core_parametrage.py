from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu

from src.io.coupes_manager import CoupesManager
from src.ui.app_core_parametrage_lithologie import render_parametrage_lithologie
from src.ui.app_core_parametrage_geologie import render_parametrage_geologie
from src.ui.app_core_parametrage_tirants import render_parametrage_tirants
from src.ui.app_core_parametrage_butons import render_parametrage_butons
from src.ui.app_core_parametrage_parois import render_parametrage_parois
from src.ui.app_core_parametrage_planchers import render_parametrage_planchers  # ✅ AJOUT


# ------------------------------------------------------
# Styles option_menu horizontal (identiques à tes autres pages)
# ------------------------------------------------------
_OPT_MENU_STYLES = {
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

# ------------------------------------------------------
# Couleurs -> emoji (uniquement pour les sous-onglets "coupes")
# ------------------------------------------------------
_ZONE_EMOJI = ["🔵", "🟢", "🟠", "🟣", "🔴", "🟦", "🟡", "🩷"]
_COLOR_TO_EMOJI = {
    "#146EFF": "🔵",
    "#22C55E": "🟢",
    "#F97316": "🟠",
    "#A855F7": "🟣",
    "#EF4444": "🔴",
    "#06B6D4": "🟦",
    "#EAB308": "🟡",
    "#EC4899": "🩷",
}


def _zone_emoji(i: int) -> str:
    return _ZONE_EMOJI[i % len(_ZONE_EMOJI)]


def _emoji_from_color(col: str, fallback_i: int) -> str:
    col = (col or "").strip()
    return _COLOR_TO_EMOJI.get(col, _zone_emoji(fallback_i))


# ------------------------------------------------------
# Project paths (robuste, JSON-only)
# ------------------------------------------------------
def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "data" / "common_data").exists():
            return p
        if (p / "app.py").exists():
            return p
    return here.parents[2]


def _default_common_data_dir() -> Path:
    d = _project_root() / "data" / "common_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_coupes():
    # ✅ JSON-only
    mgr = CoupesManager()
    coupes = mgr.list_coupes()
    return sorted(
        coupes,
        key=lambda c: (
            int(getattr(c, "ui_idx", 0) or 0),
            str(getattr(c, "name", "") or ""),
        ),
    )


def _render_coupe_subtabs_placeholder(section_key: str, coupes) -> None:
    if not coupes:
        st.warning("Aucune coupe trouvée dans le JSON.")
        st.info("(Contenu vide)")
        return

    display_to_name: dict[str, str] = {}
    display_options: list[str] = []

    for i, c in enumerate(coupes):
        name = str(getattr(c, "name", "") or "").strip()
        if not name:
            continue
        col = str(getattr(c, "color", "") or "").strip()
        disp = f"{_emoji_from_color(col, i)} {name}"
        display_to_name[disp] = name
        display_options.append(disp)

    if not display_options:
        st.warning("Aucune coupe valide.")
        st.info("(Contenu vide)")
        return

    ss_key = f"param_{section_key}_selected_coupe"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = display_to_name[display_options[0]]

    default_name = str(st.session_state[ss_key] or "")
    default_disp = next((d for d, n in display_to_name.items() if n == default_name), display_options[0])

    selected_disp = option_menu(
        menu_title=None,
        options=display_options,
        icons=[""] * len(display_options),
        orientation="horizontal",
        key=f"param_{section_key}_coupes_tabs",
        default_index=display_options.index(default_disp) if default_disp in display_options else 0,
        styles=_OPT_MENU_STYLES,
    )

    selected_name = display_to_name.get(selected_disp, display_to_name[display_options[0]])
    st.session_state[ss_key] = selected_name

    st.caption(f"{section_key} • {selected_name}")
    st.info("Contenu à venir.")


# ======================================================
# Butons list builder (basé sur les coupes)
# ======================================================
def _build_butons_from_coupes(coupes):
    from types import SimpleNamespace

    out = []
    for c in (coupes or []):
        name = str(getattr(c, "name", "") or "").strip()
        if not name:
            continue
        col = str(getattr(c, "color", "") or "").strip()
        out.append(SimpleNamespace(name=name, color=col))
    return out


# ======================================================
# Planchers list builder (simple)
# ======================================================
def _build_planchers_default(n: int = 1):
    """
    Pas encore de PlanchersManager => on crée une liste stable d'onglets.
    Par défaut: 1 onglet "Plancher 1".
    Tu peux monter à 2/3 plus tard si tu veux.
    """
    from types import SimpleNamespace

    n = int(n or 1)
    n = max(1, min(20, n))
    return [SimpleNamespace(name=f"Plancher {i+1}", color="") for i in range(n)]


# ======================================================
# Public API
# ======================================================
def render_parametrage(workbook_path: str | Path | None = None, common_data_dir: str | Path | None = None) -> None:
    _ = workbook_path  # compat legacy
    st.subheader("Paramètrage")

    common_data = Path(common_data_dir) if common_data_dir is not None else _default_common_data_dir()

    try:
        coupes = _load_coupes()
    except Exception as e:
        st.error("Impossible de charger les coupes (via CoupesManager).")
        st.exception(e)
        coupes = []

    tab_litho, tab_geo, tab_tir, tab_but, tab_par, tab_pla = st.tabs(
        ["Lithologie", "Geologie", "Tirants", "Butons", "Parois", "Planchers"]
    )

    with tab_litho:
        render_parametrage_lithologie(common_data)

    with tab_geo:
        render_parametrage_geologie(common_data_dir=common_data, coupes=coupes)

    with tab_tir:
        render_parametrage_tirants(common_data_dir=common_data, coupes=coupes)

    with tab_but:
        butons = _build_butons_from_coupes(coupes)
        if not butons:
            st.warning("Aucun buton trouvé.")
            st.info("(Contenu vide)")
        else:
            render_parametrage_butons(common_data_dir=common_data, butons=butons)

    with tab_par:
        render_parametrage_parois(common_data_dir=common_data, coupes=coupes)

    with tab_pla:
        # ✅ Intégration Planchers (onglets planchers + tableau Niveau/Cote)
        planchers = _build_planchers_default(n=1)
        render_parametrage_planchers(common_data_dir=common_data, planchers=planchers)
