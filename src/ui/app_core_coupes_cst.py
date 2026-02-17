# ======================================================
# src/ui/app_core_coupes_cst.py  (COMPACT + FACTORISÉ)
# - Onglets coupes (streamlit-option-menu) + bouton ajout
# - 🗑️ supprimer / ✏️ renommer la coupe sélectionnée
# - Affiche les figures topo de la coupe sélectionnée
#
# ✅ UX Streamlit-native:
# - Rename via st.dialog + st.form => Enter valide, croix ferme
# - Delete via st.dialog simplifié:
#     - UNE seule ligne: "Confirmer la suppression de la coupe XXXX"
#     - UN seul gros bouton rouge "🗑 Supprimer"
#     - croix pour fermer/annuler
#
# ✅ FIX landing after rename/add/delete:
# - option_menu key versionnée (cst_tabs_version)
#
# ✅ IMPORTANT:
# - AUCUNE CSS globale sur nav (ne casse pas les autres menus)
#
# ✅ PERF:
# - read_targets_timeseries(..., mtime=cache_buster mtime+size) => invalide cache si xlsx change
# - permet la voie "ultra rapide" openpyxl streaming côté reader
# ======================================================

from __future__ import annotations

from pathlib import Path
import math
from typing import Iterable

import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go

from src.io.coupes_manager import CoupesManager
from src.pipeline.mesures_completes_reader import read_targets_timeseries, compute_deltas_vs_first_known
from src.render.plots.topo_targets_plotly import build_topo_figures_for_zone

# ======================================================
# Palette + mapping emoji
# ======================================================
_ZONE_PALETTE = ["#146EFF", "#22C55E", "#F97316", "#A855F7", "#EF4444", "#06B6D4", "#EAB308", "#EC4899"]
_ZONE_EMOJI = ["🔵", "🟢", "🟠", "🟣", "🔴", "🟦", "🟡", "🩷"]
_COLOR_TO_EMOJI = dict(zip(_ZONE_PALETTE, _ZONE_EMOJI))


def _emoji(color: str, i: int) -> str:
    c = (color or "").strip()
    return _COLOR_TO_EMOJI.get(c, _ZONE_EMOJI[i % len(_ZONE_EMOJI)])


def _fallback_color(i: int) -> str:
    return _ZONE_PALETTE[i % len(_ZONE_PALETTE)]


# ======================================================
# Project paths (robuste, JSON-only)
# ======================================================
def _common_data_dir() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "data" / "common_data").exists() or (p / "app.py").exists():
            d = p / "data" / "common_data"
            d.mkdir(parents=True, exist_ok=True)
            return d
    d = here.parents[2] / "data" / "common_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ======================================================
# Helpers
# ======================================================
def _find_mesures_completes_xlsx(common_data_dir: Path) -> Path | None:
    if not common_data_dir.exists():
        return None
    needles = ("mesures completes", "mesures complètes", "mesures complete", "mesures compl")
    files = [
        p
        for p in common_data_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".xlsx" and any(n in p.stem.lower() for n in needles)
    ]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _cache_buster_mtime_size(p: Path) -> float:
    """
    Cache-buster robuste (aligné 3D topo):
    - combine mtime + size pour invalider le cache même si mtime peu fiable
    """
    try:
        stt = p.stat()
        return float(stt.st_mtime) + float(getattr(stt, "st_size", 0)) * 1e-9
    except Exception:
        try:
            return float(p.stat().st_mtime)
        except Exception:
            return 0.0


def _ss_defaults(**defaults) -> None:
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _bump_tabs() -> None:
    st.session_state["cst_tabs_version"] = int(st.session_state.get("cst_tabs_version", 0)) + 1


def _rerun_after(toast: str | None = None, icon: str | None = None) -> None:
    if toast:
        st.toast(toast, icon=icon)
    st.rerun()


def _next_default_coupe_name(existing: Iterable[str]) -> str:
    s = set(existing)
    n = len(s) + 1
    while True:
        cand = f"Coupe {n}"
        if cand not in s:
            return cand
        n += 1


def _css_safe() -> None:
    st.markdown(
        """
<style>
div[data-testid="column"] { padding-top: 0 !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _strip_fig(fig: go.Figure) -> go.Figure:
    """Fix 'undefined' + pas de légende + X sans titre + Y = mm."""
    try:
        fig.update_layout(title_text="", showlegend=False, legend_title_text="")
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="mm")  # ✅ unité verticale
        # enlever annotation "undefined"
        try:
            fig.layout.annotations = tuple(
                a
                for a in (fig.layout.annotations or [])
                if (getattr(a, "text", "") or "").strip().lower() != "undefined"
            )
        except Exception:
            pass
    except Exception:
        pass
    return fig


def _legend_only(figs: list[go.Figure], items_per_row: int = 4) -> go.Figure:
    """Figure Plotly qui n'affiche QUE la légende (hauteur dynamique)."""

    def _colors(t):
        line_c = getattr(getattr(t, "line", None), "color", None)
        m = getattr(t, "marker", None)
        return line_c, getattr(m, "color", None) if m else None, getattr(m, "symbol", None) if m else None

    seen, items = set(), []
    for f in figs:
        for t in (getattr(f, "data", None) or []):
            name = getattr(t, "name", None)
            if not name:
                continue
            lc, mc, ms = _colors(t)
            key = (str(name), str(lc), str(mc), str(ms))
            if key in seen:
                continue
            seen.add(key)
            items.append((str(name), lc, mc, ms))

    n = len(items)
    items_per_row = max(1, int(items_per_row))
    rows = max(1, int(math.ceil(n / items_per_row))) if n else 1
    height = 40 + rows * 28

    lf = go.Figure()
    for name, lc, mc, ms in items:
        lf.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines+markers",
                name=name,
                line={"color": lc} if lc else None,
                marker=({"color": mc, "symbol": ms} if (mc or ms) else None),
                hoverinfo="skip",
                showlegend=True,
            )
        )
    lf.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title_text="",
        legend_title_text="",
        showlegend=True,
        legend=dict(orientation="h", x=0.0, y=1.0, xanchor="left", yanchor="top"),
    )
    lf.update_xaxes(showgrid=False, zeroline=False)
    lf.update_yaxes(showgrid=False, zeroline=False)
    return lf


def _add_coupe_and_select(mgr: CoupesManager, name: str) -> None:
    mgr.add_coupe(name)
    st.session_state["cst_selected_coupe"] = name
    _bump_tabs()
    _rerun_after("Coupe créée ✅", icon="✅")


# ======================================================
# Dialogs
# ======================================================
@st.dialog("Renommer la coupe")
def _dialog_rename(mgr: CoupesManager, old_name: str) -> None:
    _ss_defaults(cst_rename_value=old_name)

    with st.form("cst_rename_form", clear_on_submit=False):
        new_val = st.text_input(
            "Nouveau nom",
            value=st.session_state.get("cst_rename_value", old_name),
            key="cst_rename_input",
            placeholder="Nouveau nom…",
        )
        c1, c2 = st.columns([1, 1])
        ok = c1.form_submit_button("✅ Valider", use_container_width=True)
        cancel = c2.form_submit_button("Annuler", use_container_width=True)

    if cancel:
        st.session_state["cst_open_rename"] = False
        st.rerun()

    if ok:
        candidate = str(new_val).strip()
        if not candidate:
            st.error("Nom vide.")
            return
        if candidate == old_name:
            st.session_state["cst_open_rename"] = False
            st.rerun()
        try:
            mgr.rename_coupe(old_name, candidate)
        except Exception as e:
            st.error(f"Impossible de renommer: {type(e).__name__}: {e}")
            return

        st.session_state["cst_selected_coupe"] = candidate
        st.session_state["cst_open_rename"] = False
        _bump_tabs()
        _rerun_after("Coupe renommée ✅", icon="✅")


@st.dialog(" ")
def _dialog_delete_simplified(mgr: CoupesManager) -> None:
    name = str(st.session_state.get("cst_delete_candidate", "")).strip()
    if not name:
        st.session_state["cst_open_delete"] = False
        st.rerun()

    st.markdown(f"### Confirmer la suppression de la coupe **{name}**")
    st.markdown(
        """
<style>
div[data-testid="stButton"] button[kind="primary"]{
  background: #dc2626 !important;
  border-color: #dc2626 !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover{ filter: brightness(0.95); }
</style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🗑 Supprimer", type="primary", use_container_width=True, key="cst_delete_red_btn"):
        try:
            mgr.delete_coupe(name)
        except Exception as e:
            st.error(f"Impossible de supprimer: {type(e).__name__}: {e}")
            return
        st.session_state["cst_open_delete"] = False
        _bump_tabs()
        _rerun_after("Coupe supprimée ✅", icon="✅")


# ======================================================
# UI blocks
# ======================================================
def _render_coupe_selector(coupes) -> tuple[str, bool]:
    names = [c.name for c in coupes]
    _ss_defaults(cst_selected_coupe=(names[0] if names else ""), cst_tabs_version=0)

    selected = st.session_state.get("cst_selected_coupe") or (names[0] if names else "")
    if selected not in names and names:
        selected = names[0]
        st.session_state["cst_selected_coupe"] = selected

    display_options, disp2name = [], {}
    for i, c in enumerate(coupes):
        col = (getattr(c, "color", "") or "").strip() or _fallback_color(i)
        disp = f"{_emoji(col, i)} {c.name}"
        display_options.append(disp)
        disp2name[disp] = c.name

    # default index
    def_idx = 0
    for i, d in enumerate(display_options):
        if disp2name.get(d) == selected:
            def_idx = i
            break

    c_tabs, c_plus = st.columns([18, 2], vertical_alignment="center")
    with c_tabs:
        v = int(st.session_state.get("cst_tabs_version", 0))
        selected_disp = option_menu(
            menu_title=None,
            options=display_options,
            icons=[""] * len(display_options),
            orientation="horizontal",
            key=f"cst_coupe_tabs__v{v}",
            default_index=def_idx,
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
        did_plus = st.button("+", key="cst_add_plus", help="Ajouter une coupe", type="secondary")

    chosen = disp2name.get(selected_disp, selected)
    st.session_state["cst_selected_coupe"] = chosen
    return chosen, did_plus


def _render_title_actions(mgr: CoupesManager, coupe_name: str) -> None:
    _ss_defaults(cst_open_rename=False, cst_open_delete=False, cst_delete_candidate="")

    c_left, c_edit, c_trash = st.columns([20, 1, 1], vertical_alignment="center")
    c_left.markdown(f"### {coupe_name}")

    if c_edit.button("✎", key="cst_edit", help="Renommer cette coupe", type="secondary"):
        st.session_state["cst_open_rename"] = True
        st.rerun()

    if c_trash.button("🗑", key="cst_trash", help="Supprimer cette coupe", type="secondary"):
        st.session_state["cst_delete_candidate"] = coupe_name
        st.session_state["cst_open_delete"] = True
        st.rerun()

    if st.session_state.get("cst_open_rename"):
        _dialog_rename(mgr, coupe_name)
    if st.session_state.get("cst_open_delete"):
        _dialog_delete_simplified(mgr)


# ======================================================
# Main
# ======================================================
def render_coupes_cst() -> None:
    _css_safe()
    mgr = CoupesManager()  # JSON-only

    common = _common_data_dir()
    mesures_path = _find_mesures_completes_xlsx(common)
    if not mesures_path or not mesures_path.exists():
        st.error(f"Fichier Mesures Completes introuvable dans {common}")
        return

    try:
        coupes = mgr.list_coupes()
    except Exception as e:
        st.error(f"Impossible de lire les coupes: {type(e).__name__}: {e}")
        return

    # ---- No coupes -> create Coupe 1
    if not coupes:
        st.info("Aucune coupe pour l’instant. Clique sur **+** pour créer automatiquement **Coupe 1**.")
        if st.button("+", key="cst_add_plus_empty", help="Créer Coupe 1", type="secondary"):
            try:
                _add_coupe_and_select(mgr, "Coupe 1")
            except Exception as e:
                st.error(f"Impossible de créer la coupe: {type(e).__name__}: {e}")
        return

    selected, did_plus = _render_coupe_selector(coupes)

    if did_plus:
        try:
            new_name = _next_default_coupe_name([c.name for c in coupes])
            _add_coupe_and_select(mgr, new_name)
        except Exception as e:
            st.error(f"Impossible de créer la coupe: {type(e).__name__}: {e}")
            return

    # Refresh list after any mutation elsewhere
    coupes = mgr.list_coupes()
    names = [c.name for c in coupes]
    if selected not in names and names:
        st.session_state["cst_selected_coupe"] = names[0]
        _bump_tabs()
        st.rerun()

    selected = st.session_state["cst_selected_coupe"]
    _render_title_actions(mgr, selected)

    coupe = mgr.get_coupe(selected)
    if not coupe.targets:
        st.info("Aucune cible enregistrée pour cette coupe (targets vide).")
        return

    # ==================================================
    # ✅ PERF: cache-buster pour déclencher la voie rapide
    # ==================================================
    cache_buster = _cache_buster_mtime_size(mesures_path)

    ts_by_target = read_targets_timeseries(
        str(mesures_path),
        coupe.targets,
        sheet_name=None,
        mtime=cache_buster,  # ✅ clé de cache + invalidation quand le fichier change
    )

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
    if not figs:
        st.warning("Aucune figure topo générée.")
        return

    figs = [_strip_fig(f) for f in list(figs)]
    legend_fig = _legend_only(figs, items_per_row=4)

    st.markdown("### Topographie")
    labels = ["Mouvements Normaux", "Mouvements Tangentiels", "Mouvements Verticaux"]

    for i, fig in enumerate(figs):
        if i < 3:
            st.markdown(f"#### {labels[i]}")
        else:
            st.markdown(f"#### Mouvement {i+1}")
        st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(legend_fig, use_container_width=True)
