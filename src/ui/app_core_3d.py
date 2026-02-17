# ======================================================
# src/ui/app_core_3d.py  (COMPLET - additive + camera normalized + vector n in triad)
# ✅ Un seul go.Figure() -> topo ajoute ses traces
# ✅ Caméra initiale basée sur Coupe.angle_deg (sélection des zones)
# ✅ camera.* en coordonnées NORMALISÉES Plotly => plus de "hors champ"
# ✅ Vecteur "n" intégré AU REPÈRE XYZ (même origine + même longueur que les segments)
# ✅ Si pas d'angle: vue neutre + pas de vecteur n
# ✅ Chantier complet: angle = moyenne circulaire pondérée par nb cibles de chaque coupe
# ✅ MAIS: on N'AFFICHE PAS le vecteur n en "chantier complet" (caméra seulement)
#
# ✅ FIX UPDATE DATA:
# - data_tag (cache-bust) basé sur session_state.data_hash["topo"] si dispo, sinon mtime du xlsx
# - uirevision inclut data_tag => Plotly refresh quand le fichier change
# - compute_topo_payload(..., cache_bust=...) si supporté (fallback sinon)
# - mesures_header(..., mtime=...) si supporté (fallback sinon)
#
# ✅ UI tweak:
# - "Cibles présentes dans plusieurs coupes" est rendu SOUS le plot (hook below_plot_fn)
# ======================================================

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional
import math
import hashlib

import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

from src.io.coupes_manager import CoupesManager

from src.ui.app_core_3d_topo import (
    controls as topo_controls,
    compute_topo_payload,
    add_topo_traces,
    topo_post_render,
    mesures_header,
    zone_color,
    emoji_from_color,
)


# ======================================================
# Paths
# ======================================================
def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "data" / "common_data").exists() or (p / "app.py").exists():
            return p
    return here.parents[2]


def _find_mesures_completes_xlsx(common_data_dir: Path) -> Path | None:
    if not common_data_dir.exists():
        return None
    needles = ("mesures completes", "mesures complètes", "mesures complete", "mesures compl")
    xs = [
        p
        for p in common_data_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".xlsx"
        and any(n in p.stem.lower() for n in needles)
    ]
    if not xs:
        return None
    return max(xs, key=lambda x: x.stat().st_mtime)


# ======================================================
# Cache-bust (cloud safe): prefer session_state.data_hash["topo"], fallback mtime
# ======================================================
def _get_dataset_hash(kind: str) -> str:
    dh = st.session_state.get("data_hash", {})
    if isinstance(dh, dict):
        return str(dh.get(kind, "") or "")
    return ""


def _hash_to_float(sig: str) -> float:
    """
    Convertit un hash (hex ou string) en float stable (pour casser cache_data/LRU).
    """
    if not sig:
        return 0.0
    s = str(sig).strip()
    if not s:
        return 0.0
    try:
        n = int(s[:12], 16)
        return float(n)
    except Exception:
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()
        n = int(h[:12], 16)
        return float(n)


def _data_tag_for_file(p: Path, kind: str = "topo") -> Tuple[str, float]:
    """
    Retourne (data_tag_str, cache_bust_float)
    - si data_hash[kind] dispo => tag="hash:..." + float stable
    - sinon => tag="mtime:..." + float(mtime)
    """
    sig = _get_dataset_hash(kind)
    if sig:
        return f"hash:{sig}", _hash_to_float(sig)
    try:
        mt = float(p.stat().st_mtime)
    except Exception:
        mt = 0.0
    return f"mtime:{mt:.6f}", mt


# ======================================================
# Angle helpers (robuste)
# ======================================================
def _angle_is_valid(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return False
        try:
            v = float(s)
        except Exception:
            return False
    try:
        f = float(v)
    except Exception:
        return False
    return math.isfinite(f)


def _normalize_deg(a: float) -> float:
    x = float(a) % 360.0
    if x < 0:
        x += 360.0
    return x


def _weighted_circular_mean_deg(angles_deg: List[float], weights: List[float]) -> float | None:
    """
    Moyenne circulaire pondérée :
      mean = atan2(sum(w*sin(theta)), sum(w*cos(theta)))
    Retourne un angle en degrés [0,360) ou None si pas défini.
    """
    if not angles_deg or not weights or len(angles_deg) != len(weights):
        return None

    sw = 0.0
    sx = 0.0
    sy = 0.0
    for a, w in zip(angles_deg, weights):
        ww = float(w)
        if ww <= 0:
            continue
        th = math.radians(float(a))
        sx += ww * math.cos(th)
        sy += ww * math.sin(th)
        sw += ww

    if sw <= 0 or (abs(sx) < 1e-12 and abs(sy) < 1e-12):
        return None

    mean = math.degrees(math.atan2(sy, sx))
    return _normalize_deg(mean)


# ======================================================
# Coupes maps + UI selection
# ======================================================
def _build_maps(coupes) -> Tuple[List[str], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    coupe_names: list[str] = []
    coupe_color: Dict[str, str] = {}
    coupe_label: Dict[str, str] = {}
    target_to_color: Dict[str, str] = {}
    duplicates: Dict[str, List[str]] = {}

    for i, c in enumerate(coupes):
        name = str(getattr(c, "name", "") or "").strip()
        if not name:
            continue
        coupe_names.append(name)

        col = (getattr(c, "color", "") or "").strip() or zone_color(i)
        coupe_color[name] = col
        coupe_label[name] = f"{emoji_from_color(col, i)} {name}"

        for t in (getattr(c, "targets", None) or []):
            tt = str(t).strip()
            if not tt:
                continue
            prev = target_to_color.get(tt)
            if prev and prev != col:
                duplicates.setdefault(tt, []).extend([prev, col])
            else:
                target_to_color[tt] = col

    return coupe_names, coupe_color, coupe_label, target_to_color, duplicates


def _select_coupe(coupe_names: List[str], coupe_label: Dict[str, str]) -> str:
    options = [coupe_label[n] for n in coupe_names]
    display_to_name = {coupe_label[n]: n for n in coupe_names}

    st.session_state.setdefault("cst_selected_coupe_3d", coupe_names[0])
    default_name = st.session_state["cst_selected_coupe_3d"]
    if default_name not in coupe_names:
        default_name = coupe_names[0]

    default_disp = coupe_label[default_name]
    selected_disp = option_menu(
        menu_title=None,
        options=options,
        icons=[""] * len(options),
        orientation="horizontal",
        key="cst_coupe_tabs_3d",
        default_index=options.index(default_disp) if default_disp in options else 0,
        styles={
            "container": {"padding": "0!important"},
            "icon": {"display": "none"},
            "nav-link": {
                "font-size": "0.90rem",
                "padding": "8px 10px",
                "margin": "0 4px 0 0",
                "border-radius": "0.6rem",
                "white-space": "nowrap",
            },
            "nav-link-hover": {"background-color": "rgba(0,0,0,0.06)"},
            "nav-link-selected": {"background-color": "rgba(0,0,0,0.10)", "font-weight": "700"},
        },
    )

    selected_name = display_to_name.get(selected_disp, coupe_names[0])
    st.session_state["cst_selected_coupe_3d"] = selected_name
    return selected_name


# ======================================================
# Camera (Plotly normalized coords, safe)
# ======================================================
def _camera_neutral() -> dict:
    return dict(
        eye=dict(x=1.6, y=1.6, z=1.0),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
    )


def _camera_from_angle_deg(angle_deg: float, right_deg: float = 30.0) -> dict:
    """
    Camera en coords normalisées Plotly.
    Objectif: que le vecteur (angle_deg) "sorte de l'écran" légèrement à droite.
    => caméra DEVANT le vecteur (même sens), puis yaw de -right_deg.
    """
    theta = math.radians(float(angle_deg))
    vx, vy = math.cos(theta), math.sin(theta)

    camx, camy = vx, vy

    a = math.radians(-float(right_deg))
    camx, camy = (camx * math.cos(a) - camy * math.sin(a), camx * math.sin(a) + camy * math.cos(a))

    dist = 2.2
    height = 1.0

    return dict(
        eye=dict(x=camx * dist, y=camy * dist, z=height),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
    )


def _init_layout(fig, *, uirevision: str, camera: dict | None) -> None:
    scene = dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="rgba(0,0,0,0)",
        aspectmode="data",
        dragmode="orbit",
    )
    if camera is not None:
        scene["camera"] = camera

    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        scene=scene,
        uirevision=uirevision,
        hovermode="closest",
    )


# ======================================================
# Vecteur "n" intégré au trièdre XYZ (mêmes formules que _add_xyz_triad)
# ======================================================
def _add_n_vector_in_triad(fig, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, angle_deg: float) -> None:
    import plotly.graph_objects as go

    if xs.size == 0 or ys.size == 0 or zs.size == 0:
        return

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    zmin, zmax = float(zs.min()), float(zs.max())
    dx, dy, dz = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0), max(zmax - zmin, 1.0)
    diag = float(np.sqrt(dx * dx + dy * dy + dz * dz))

    L = max(diag * 0.08, 1.0)
    ox, oy, oz = xmin + dx * 0.06, ymin + dy * 0.06, zmin + dz * 0.06

    theta = math.radians(float(angle_deg))
    vx, vy = math.cos(theta), math.sin(theta)

    tip_x = ox + vx * L
    tip_y = oy + vy * L
    tip_z = oz

    fig.add_trace(
        go.Scatter3d(
            x=[ox, tip_x],
            y=[oy, tip_y],
            z=[oz, tip_z],
            mode="lines",
            hoverinfo="skip",
            line=dict(width=6, color="rgba(34,211,238,0.95)"),
            showlegend=False,
            name="",
        )
    )

    head_len = max(L * 0.30, 0.8)
    cone_sizeref = max(L * 0.45, 0.9)

    base_x = tip_x - vx * head_len
    base_y = tip_y - vy * head_len
    base_z = oz

    fig.add_trace(
        go.Cone(
            x=[base_x],
            y=[base_y],
            z=[base_z],
            u=[vx * head_len],
            v=[vy * head_len],
            w=[0.0],
            anchor="tail",
            showscale=False,
            cauto=False,
            cmin=0.0,
            cmax=1.0,
            colorscale=[[0.0, "rgba(34,211,238,0.95)"], [1.0, "rgba(34,211,238,0.95)"]],
            sizemode="absolute",
            sizeref=float(cone_sizeref),
            opacity=0.98,
            hoverinfo="skip",
            name="",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[tip_x + vx * (0.15 * L)],
            y=[tip_y + vy * (0.15 * L)],
            z=[oz],
            mode="text",
            text=["n"],
            textfont=dict(size=14, color="rgba(34,211,238,0.95)"),
            hoverinfo="skip",
            showlegend=False,
            name="",
        )
    )


# ======================================================
# Render composed
# ======================================================
def _render_composed_3d(
    *,
    key_prefix: str,
    selected_key: str,
    targets: List[str],
    mesures_path: Path,
    target_color_map: Dict[str, str] | None,
    default_point_color: str,
    camera_angle_deg: float | None,
    show_n_vector: bool,
    data_tag: str,
    cache_bust: float,
    below_plot_fn: Optional[Callable[[], None]] = None,  # ✅ NEW: content rendered just under the plot
) -> None:
    import plotly.graph_objects as go

    show_surface, show_vectors, scale, motion, auto_filter, outlier_k = topo_controls(key_prefix, render_widgets=False)

    try:
        topo_payload = compute_topo_payload(
            selected_key=selected_key,
            targets=targets,
            mesures_path=mesures_path,
            show_vectors=show_vectors,
            scale=scale,
            auto_filter=auto_filter,
            outlier_k=outlier_k,
            motion_mm_range=motion,
            target_color_map=target_color_map,
            default_point_color=default_point_color,
            cache_bust=cache_bust,
        )
    except TypeError:
        topo_payload = compute_topo_payload(
            selected_key=selected_key,
            targets=targets,
            mesures_path=mesures_path,
            show_vectors=show_vectors,
            scale=scale,
            auto_filter=auto_filter,
            outlier_k=outlier_k,
            motion_mm_range=motion,
            target_color_map=target_color_map,
            default_point_color=default_point_color,
        )

    if topo_payload is None:
        return

    fig = go.Figure()
    add_topo_traces(fig, topo_payload, show_surface=show_surface, show_vectors=show_vectors)

    angle_ok = _angle_is_valid(camera_angle_deg)
    angle_val = float(camera_angle_deg) if angle_ok else None

    if angle_val is not None and show_n_vector:
        _add_n_vector_in_triad(fig, topo_payload.xs, topo_payload.ys, topo_payload.zs, angle_val)

    if angle_val is None:
        cam = _camera_neutral()
        ang_tag = "none"
    else:
        cam = _camera_from_angle_deg(angle_val, right_deg=30.0)
        ang_tag = f"{angle_val:.3f}"

    uirev = f"3d::{selected_key}::cam::{ang_tag}::data::{data_tag}"
    _init_layout(fig, uirevision=uirev, camera=cam)

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"plot3d::{selected_key}",
        config={"displaylogo": False, "scrollZoom": True},
    )

    # ✅ HERE: render extra blocks right below the plot
    if below_plot_fn is not None:
        below_plot_fn()

    topo_post_render(topo_payload, show_surface=show_surface)

    st.markdown("---")
    st.markdown("### Paramètres")
    topo_controls(key_prefix, render_widgets=True)


# ======================================================
# Chantier complet: angle moyen pondéré (par nb cibles)
# ======================================================
def _site_weighted_angle_from_coupes(coupes) -> float | None:
    angles: list[float] = []
    weights: list[float] = []

    for c in coupes:
        a = getattr(c, "angle_deg", None)
        if not _angle_is_valid(a):
            continue

        tgs = getattr(c, "targets", None) or []
        w = 0
        for t in tgs:
            if isinstance(t, str) and t.strip():
                w += 1
        if w <= 0:
            continue

        angles.append(float(a))
        weights.append(float(w))

    return _weighted_circular_mean_deg(angles, weights)


# ======================================================
# Public UI
# ======================================================
def render_3d() -> None:
    root = _project_root()
    common_data = root / "data" / "common_data"

    mgr = CoupesManager()
    coupes = mgr.list_coupes()
    if not coupes:
        st.info("Aucune coupe détectée dans le JSON.")
        return

    mesures_path = _find_mesures_completes_xlsx(common_data)
    if mesures_path is None or not mesures_path.exists():
        st.error(f"Fichier Mesures Completes introuvable dans : {common_data}")
        return

    data_tag, cache_bust = _data_tag_for_file(mesures_path, kind="topo")

    coupe_names, coupe_color, coupe_label, target_to_color, duplicates = _build_maps(coupes)
    if not coupe_names:
        st.info("Aucune coupe détectée dans le JSON.")
        return

    tab_chantier, tab_coupes = st.tabs(["Chantier Complet", "Coupes"])

    with tab_chantier:
        # ✅ IMPORTANT: appel mesures_header avec le bon kw (mtime) pour éviter un cache d'ancien dataset
        try:
            sheet, _, all_targets = mesures_header(str(mesures_path), mtime=cache_bust)
        except TypeError:
            sheet, _, all_targets = mesures_header(str(mesures_path))

        if not all_targets:
            st.error(f"Aucune cible détectée dans l'entête de Mesures Completes (1ère feuille: {sheet}).")
            return

        site_angle = _site_weighted_angle_from_coupes(coupes)

        def _below_plot_duplicates() -> None:
            if not duplicates:
                return
            with st.expander(f"Cibles présentes dans plusieurs coupes ({len(duplicates)})"):
                for t, cols in sorted(duplicates.items(), key=lambda kv: kv[0].lower()):
                    uniq: list[str] = []
                    for c in cols:
                        if c not in uniq:
                            uniq.append(c)
                    st.write(f"- **{t}** : {', '.join(uniq)}")

        # ✅ FIX CRITIQUE: en "chantier complet", on FORCE targets=[]
        _render_composed_3d(
            key_prefix="site",
            selected_key="chantier_complet",
            targets=[],  # ✅ au lieu de all_targets
            mesures_path=mesures_path,
            target_color_map=target_to_color,
            default_point_color="red",
            camera_angle_deg=site_angle,
            show_n_vector=False,
            data_tag=data_tag,
            cache_bust=cache_bust,
            below_plot_fn=_below_plot_duplicates,  # ✅ expander sous le plot
        )

    with tab_coupes:
        selected = _select_coupe(coupe_names, coupe_label)
        coupe = next((c for c in coupes if getattr(c, "name", None) == selected), None)
        if coupe is None:
            st.error("Coupe introuvable.")
            return

        targets = [t for t in (getattr(coupe, "targets", None) or []) if isinstance(t, str) and t.strip()]
        if not targets:
            st.info("Cette coupe ne contient aucune cible.")
            return

        col = coupe_color.get(selected, "red")

        cam_angle = getattr(coupe, "angle_deg", None)
        cam_angle = float(cam_angle) if _angle_is_valid(cam_angle) else None

        _render_composed_3d(
            key_prefix="coupe",
            selected_key=f"coupe::{selected}",
            targets=targets,
            mesures_path=mesures_path,
            target_color_map={str(t): col for t in targets},
            default_point_color="red",
            camera_angle_deg=cam_angle,
            show_n_vector=True,
            data_tag=data_tag,
            cache_bust=cache_bust,
            below_plot_fn=None,
        )
