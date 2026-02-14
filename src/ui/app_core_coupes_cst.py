# ======================================================
# src/ui/app_core_coupes_cst.py  (COMPLET)
# - Onglets coupes (streamlit-option-menu) + bouton +
# - 🗑️ supprimer / ✏️ renommer la coupe sélectionnée
# - Affiche les figures topo de la coupe sélectionnée
#
# ✅ NEW: chaque onglet affiche un indicateur de couleur (emoji) basé sur
#         la couleur stockée dans le JSON (champ "color")
#         -> PAS d'HTML dans les labels (option_menu échappe le HTML)
#
# IMPORTANT: aucun changement de fonctionnement ni de design,
#            juste un préfixe emoji dans le texte de l’onglet.
#
# ✅ MODIF (JSON-only):
# - CoupesManager ne prend plus workbook_path => on ne lui passe plus jamais.
# - On ne dépend plus de mgr.pg_path (attribut interne) : on utilise common_data_dir.
# ======================================================

from __future__ import annotations

from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu

from src.io.coupes_manager import CoupesManager
from src.pipeline.mesures_completes_reader import read_targets_timeseries, compute_deltas_vs_first_known
from src.render.plots.topo_targets_plotly import build_topo_figures_for_zone


# ======================================================
# Palette (fallback) + mapping emoji (safe)
# (même palette que Sélection des cibles)
# ======================================================
_ZONE_PALETTE = ["#146EFF", "#22C55E", "#F97316", "#A855F7", "#EF4444", "#06B6D4", "#EAB308", "#EC4899"]
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


def _zone_color(i: int) -> str:
    return _ZONE_PALETTE[i % len(_ZONE_PALETTE)]


def _zone_emoji(i: int) -> str:
    return _ZONE_EMOJI[i % len(_ZONE_EMOJI)]


def _emoji_from_color(col: str, fallback_i: int) -> str:
    col = (col or "").strip()
    return _COLOR_TO_EMOJI.get(col, _zone_emoji(fallback_i))


# ======================================================
# Project paths (robuste, JSON-only)
# ======================================================
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


# ======================================================
# Helpers
# ======================================================
def _find_mesures_completes_xlsx(common_data_dir: Path) -> Path | None:
    if not common_data_dir.exists():
        return None
    needles = ["mesures completes", "mesures complètes", "mesures complete", "mesures compl"]
    candidates: list[Path] = []
    for p in common_data_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".xlsx":
            stem = p.stem.lower()
            if any(n in stem for n in needles):
                candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _css_buttons() -> None:
    st.markdown(
        """
<style>
div[data-testid="stButton"] button[kind="secondary"] { line-height: 1; }

div.cst-trash div[data-testid="stButton"] button {
  background: rgba(239,68,68,0.12) !important;
  border: 1px solid rgba(239,68,68,0.35) !important;
  color: rgb(239,68,68) !important;
  padding: 0.25rem 0.55rem !important;
  border-radius: 0.55rem !important;
}

div.cst-edit div[data-testid="stButton"] button {
  background: rgba(59,130,246,0.12) !important;
  border: 1px solid rgba(59,130,246,0.35) !important;
  color: rgb(59,130,246) !important;
  padding: 0.25rem 0.55rem !important;
  border-radius: 0.55rem !important;
}

div.cst-plus div[data-testid="stButton"] button {
  background: rgba(16,185,129,0.14) !important;
  border: 1px solid rgba(16,185,129,0.35) !important;
  color: rgb(16,185,129) !important;
  padding: 0.30rem 0.60rem !important;
  border-radius: 0.60rem !important;
  font-weight: 700 !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_coupe_selector(coupes) -> tuple[str, bool]:
    """
    Onglets coupes (option_menu) + bouton +
    ✅ NEW: préfixe emoji de couleur (safe, pas d'HTML)
    Retourne (selected_coupe_name, did_plus)
    """
    coupe_names = [c.name for c in coupes]

    if "cst_selected_coupe" not in st.session_state:
        st.session_state["cst_selected_coupe"] = coupe_names[0]

    # map "display label" -> true name (option_menu renvoie le label)
    display_to_name: dict[str, str] = {}
    display_options: list[str] = []

    for i, c in enumerate(coupes):
        col = (getattr(c, "color", "") or "").strip() or _zone_color(i)
        emo = _emoji_from_color(col, i)
        disp = f"{emo} {c.name}"
        display_to_name[disp] = c.name
        display_options.append(disp)

    # valeur par défaut (display label) depuis le name en session
    default_name = st.session_state["cst_selected_coupe"]
    default_disp = next((d for d, n in display_to_name.items() if n == default_name), display_options[0])

    c_tabs, c_plus = st.columns([20, 1], vertical_alignment="center")

    with c_tabs:
        selected_disp = option_menu(
            menu_title=None,
            options=display_options,
            icons=[""] * len(display_options),
            orientation="horizontal",
            key="cst_coupe_tabs",
            default_index=display_options.index(default_disp) if default_disp in display_options else 0,
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

    with c_plus:
        st.markdown('<div class="cst-plus">', unsafe_allow_html=True)
        did_plus = st.button("+", key="cst_add_plus", help="Ajouter une coupe", type="secondary")
        st.markdown("</div>", unsafe_allow_html=True)

    selected_name = display_to_name.get(selected_disp, coupe_names[0])
    st.session_state["cst_selected_coupe"] = selected_name
    return selected_name, did_plus


def _render_add_coupe_inline(mgr: CoupesManager) -> bool:
    if "cst_add_mode" not in st.session_state:
        st.session_state["cst_add_mode"] = False
    if "cst_new_coupe_name" not in st.session_state:
        st.session_state["cst_new_coupe_name"] = ""

    if not st.session_state["cst_add_mode"]:
        return False

    c1, c2, c3 = st.columns([5, 1, 1], vertical_alignment="center")
    with c1:
        st.session_state["cst_new_coupe_name"] = st.text_input(
            "Nom nouvelle coupe",
            value=st.session_state["cst_new_coupe_name"],
            key="cst_new_coupe_input",
            label_visibility="collapsed",
            placeholder="Nom de la coupe…",
        )
    with c2:
        if st.button("Créer", type="primary", key="cst_create_coupe"):
            name = str(st.session_state["cst_new_coupe_name"]).strip()
            if not name:
                st.error("Nom vide.")
                return False
            try:
                mgr.add_coupe(name)
                st.session_state["cst_add_mode"] = False
                st.session_state["cst_new_coupe_name"] = ""
                st.session_state["cst_selected_coupe"] = name
                return True
            except Exception as e:
                st.error(f"Impossible de créer la coupe: {type(e).__name__}: {e}")
    with c3:
        if st.button("Annuler", key="cst_cancel_add"):
            st.session_state["cst_add_mode"] = False
            st.session_state["cst_new_coupe_name"] = ""
    return False


def _render_title_actions(mgr: CoupesManager, coupe_name: str) -> tuple[bool, bool, str | None]:
    """
    Returns: (did_delete, did_rename, new_name_or_None)
    """
    if "cst_delete_confirm" not in st.session_state:
        st.session_state["cst_delete_confirm"] = None

    if "cst_rename_mode" not in st.session_state:
        st.session_state["cst_rename_mode"] = False
    if "cst_rename_value" not in st.session_state:
        st.session_state["cst_rename_value"] = ""

    did_delete = False
    did_rename = False
    new_name: str | None = None

    c_left, c_edit, c_trash = st.columns([20, 1, 1], vertical_alignment="center")
    with c_left:
        st.markdown(f"### {coupe_name}")

    with c_edit:
        st.markdown('<div class="cst-edit">', unsafe_allow_html=True)
        if st.button("✏️", key=f"cst_edit_{coupe_name}", help="Renommer cette coupe", type="secondary"):
            st.session_state["cst_rename_mode"] = True
            st.session_state["cst_rename_value"] = coupe_name
        st.markdown("</div>", unsafe_allow_html=True)

    with c_trash:
        st.markdown('<div class="cst-trash">', unsafe_allow_html=True)
        if st.button("🗑️", key=f"cst_trash_{coupe_name}", help="Supprimer cette coupe", type="secondary"):
            st.session_state["cst_delete_confirm"] = coupe_name
        st.markdown("</div>", unsafe_allow_html=True)

    # rename inline
    if st.session_state.get("cst_rename_mode", False):
        r1, r2, r3 = st.columns([5, 1, 1], vertical_alignment="center")
        with r1:
            st.session_state["cst_rename_value"] = st.text_input(
                "Nouveau nom",
                value=st.session_state["cst_rename_value"],
                key="cst_rename_input",
                label_visibility="collapsed",
                placeholder="Nouveau nom…",
            )
        with r2:
            if st.button("✅", type="primary", key=f"cst_rename_ok_{coupe_name}", help="Valider"):
                candidate = str(st.session_state["cst_rename_value"]).strip()
                if not candidate:
                    st.error("Nom vide.")
                elif candidate == coupe_name:
                    st.session_state["cst_rename_mode"] = False
                    st.session_state["cst_rename_value"] = ""
                else:
                    try:
                        mgr.rename_coupe(coupe_name, candidate)
                        st.session_state["cst_rename_mode"] = False
                        st.session_state["cst_rename_value"] = ""
                        did_rename = True
                        new_name = candidate
                    except Exception as e:
                        st.error(f"Impossible de renommer: {type(e).__name__}: {e}")
        with r3:
            if st.button("Annuler", key=f"cst_rename_cancel_{coupe_name}"):
                st.session_state["cst_rename_mode"] = False
                st.session_state["cst_rename_value"] = ""

    # delete confirm
    pending = st.session_state.get("cst_delete_confirm")
    if pending == coupe_name:
        st.warning(f"Confirmer la suppression de **{coupe_name}** ?")
        a, b = st.columns([1, 1])
        with a:
            if st.button("✅ Supprimer", type="primary", key=f"cst_confirm_del_{coupe_name}"):
                try:
                    mgr.delete_coupe(coupe_name)
                    st.session_state["cst_delete_confirm"] = None
                    did_delete = True
                except Exception as e:
                    st.error(f"Impossible de supprimer: {type(e).__name__}: {e}")
        with b:
            if st.button("Annuler", key=f"cst_cancel_del_{coupe_name}"):
                st.session_state["cst_delete_confirm"] = None

    return did_delete, did_rename, new_name


# ======================================================
# Main
# ======================================================
def render_coupes_cst() -> None:
    _css_buttons()

    # ✅ JSON-only
    mgr = CoupesManager()

    # Mesures Completes (dans data/common_data)
    common = _common_data_dir()
    mesures_path = _find_mesures_completes_xlsx(common)
    if mesures_path is None or not mesures_path.exists():
        st.error(f"Fichier Mesures Completes introuvable dans {common}")
        return

    # Lecture coupes
    try:
        coupes = mgr.list_coupes()
    except Exception as e:
        st.error(f"Impossible de lire les coupes: {type(e).__name__}: {e}")
        return

    # État: aucune coupe => création inline
    if not coupes:
        st.info("Aucune coupe pour l’instant. Clique sur **+** pour en créer une, ou crée-la ci-dessous.")
        st.session_state["cst_add_mode"] = True
        if _render_add_coupe_inline(mgr):
            st.toast("Coupe créée ✅", icon="✅")
            st.rerun()
        return

    # Selector + plus (✅ onglets préfixés emoji couleur)
    selected_coupe, did_plus = _render_coupe_selector(coupes)
    if did_plus:
        st.session_state["cst_add_mode"] = True
        st.session_state["cst_new_coupe_name"] = ""

    # inline add
    if _render_add_coupe_inline(mgr):
        st.toast("Coupe créée ✅", icon="✅")
        st.rerun()

    # reload list
    coupes = mgr.list_coupes()
    coupe_names = [c.name for c in coupes]
    if selected_coupe not in coupe_names:
        selected_coupe = coupe_names[0]
        st.session_state["cst_selected_coupe"] = selected_coupe

    # header actions
    did_delete, did_rename, new_name = _render_title_actions(mgr, selected_coupe)
    if did_delete:
        st.toast("Coupe supprimée ✅", icon="✅")
        st.session_state["cst_rename_mode"] = False
        st.session_state["cst_rename_value"] = ""
        st.rerun()

    if did_rename and new_name:
        st.toast("Coupe renommée ✅", icon="✅")
        st.session_state["cst_selected_coupe"] = new_name
        st.rerun()

    # plot for selected coupe
    coupe = mgr.get_coupe(st.session_state["cst_selected_coupe"])

    if not coupe.targets:
        st.info("Aucune cible enregistrée pour cette coupe (targets vide).")
        return

    ts_by_target = read_targets_timeseries(str(mesures_path), coupe.targets, sheet_name=None)

    deltas_by_target = {}
    for name, df in ts_by_target.items():
        d = compute_deltas_vs_first_known(df)
        if d is not None and not d.empty:
            deltas_by_target[name] = d

    if not deltas_by_target:
        st.warning("Cibles trouvées dans le JSON, mais aucune série exploitable dans Mesures Completes/Data.")
        return

    figs = build_topo_figures_for_zone(
        zone_name=coupe.name,
        angle_deg=coupe.angle_deg,
        deltas_by_target=deltas_by_target,
    )
    for fig in figs:
        st.plotly_chart(fig, use_container_width=True)
