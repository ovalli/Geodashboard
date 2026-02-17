# ======================================================
# src/ui/app_core_3d_inclino.py  (COMPLET - squelette additive)
# ✅ Prêt à être ajouté au même fig que la topo
# ✅ Aucune dépendance topo
# ✅ Il te restera à remplir compute_inclino_payload() avec tes vraies données
# ======================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import streamlit as st


@dataclass
class InclinoOptions:
    # ici tu mettras tes options (choix série, projection, etc.)
    dummy: bool = True


@dataclass
class InclinoPayload:
    selected_key: str
    # exemples de structures (à adapter à tes besoins)
    # lignes 3D concaténées avec None:
    x: List[float]
    y: List[float]
    z: List[float]
    caption: str


def inclino_controls(key_prefix: str, *, render_widgets: bool) -> Tuple[bool, InclinoOptions]:
    k_enabled = f"show_inclino_3d_{key_prefix}"

    st.session_state.setdefault(k_enabled, False)

    if render_widgets:
        st.markdown("#### Inclino")
        st.checkbox("Afficher inclino", value=bool(st.session_state[k_enabled]), key=k_enabled)

    enabled = bool(st.session_state[k_enabled])
    return enabled, InclinoOptions()


def compute_inclino_payload(*, selected_key: str, mesures_path, targets: List[str], opts: InclinoOptions) -> InclinoPayload | None:
    """
    TODO: remplace ça par ta vraie lecture inclino.
    Pour l’instant: retourne None (donc pas de traces ajoutées).
    """
    return None


def add_inclino_traces(fig, payload: InclinoPayload) -> None:
    import plotly.graph_objects as go

    fig.add_trace(
        go.Scatter3d(
            x=payload.x,
            y=payload.y,
            z=payload.z,
            mode="lines",
            line=dict(color="cyan", width=6),
            hoverinfo="skip",
            showlegend=False,
            name="",
        )
    )


def inclino_post_render(payload: InclinoPayload) -> None:
    st.caption(payload.caption)
