from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import pandas as pd
import plotly.graph_objects as go


def _apply_layout(fig: go.Figure, title: str, yunit: str = "mm") -> go.Figure:
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
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=yunit)
    return fig


def _angle_rad(angle_deg: float) -> float:
    return float(angle_deg) * math.pi / 180.0


def _rotate_to_normal_tangent(dx: pd.Series, dy: pd.Series, angle_deg: float) -> Tuple[pd.Series, pd.Series]:
    """
    Angle : "Angle orienté (X n), trigo" (hypothèse standard)
    Normal = projection sur l'axe orienté par angle
    Tangentiel = projection sur l'axe orthogonal (trigo)
    """
    a = _angle_rad(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    normal = dx * ca + dy * sa
    tang = -dx * sa + dy * ca
    return normal, tang


def build_topo_figures_for_zone(
    zone_name: str,
    angle_deg: float,
    deltas_by_target: Dict[str, pd.DataFrame],
) -> List[go.Figure]:
    """
    Attend deltas_by_target[name] avec colonnes date, dx,dy,dz (en unités Excel, souvent mm)
    Retourne 3 figures: Normaux, Tangentiels, Verticaux
    """
    fig_n = go.Figure()
    fig_t = go.Figure()
    fig_v = go.Figure()

    # bornes x communes
    all_dates = []
    for d in deltas_by_target.values():
        if d is not None and not d.empty:
            all_dates.append(pd.to_datetime(d["date"], errors="coerce"))
    if all_dates:
        xmin = pd.concat(all_dates).dropna().min()
        xmax = pd.concat(all_dates).dropna().max()
        fig_n.update_xaxes(range=[xmin, xmax])
        fig_t.update_xaxes(range=[xmin, xmax])
        fig_v.update_xaxes(range=[xmin, xmax])

    for name, d in deltas_by_target.items():
        if d is None or d.empty:
            continue

        dates = pd.to_datetime(d["date"], errors="coerce")
        dx = pd.to_numeric(d["dx"], errors="coerce")
        dy = pd.to_numeric(d["dy"], errors="coerce")
        dz = pd.to_numeric(d["dz"], errors="coerce")

        normal, tang = _rotate_to_normal_tangent(dx, dy, angle_deg)

        fig_n.add_trace(go.Scatter(x=dates, y=normal, mode="lines+markers", name=name))
        fig_t.add_trace(go.Scatter(x=dates, y=tang, mode="lines+markers", name=name))
        fig_v.add_trace(go.Scatter(x=dates, y=dz, mode="lines+markers", name=name))

    fig_n = _apply_layout(fig_n, f"Topographie : Mouvements Normaux — {zone_name}", yunit="mm")
    fig_t = _apply_layout(fig_t, f"Topographie : Mouvements Tangentiels — {zone_name}", yunit="mm")
    fig_v = _apply_layout(fig_v, f"Topographie : Mouvements Verticaux — {zone_name}", yunit="mm")

    return [fig_n, fig_t, fig_v]
