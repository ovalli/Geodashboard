from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple
import base64
import hashlib
import json
import math  # ✅ LIVE_VEC

import numpy as np
import pandas as pd
import streamlit as st

from src.io.coupes_manager import CoupesManager, Coupe
from src.pipeline.mesures_completes_reader import list_targets_in_mesures, read_targets_timeseries


DATASET_KIND_FOR_POINTS = "topo"


# ------------------------------------------------------
# Component import
# ------------------------------------------------------
def _import_selection_component():
    tries = [
        ("src.ui.selection_cibles_html", "selection_cibles_component"),
        ("src.ui.selection_cibles_component", "selection_cibles_component"),
    ]
    last: Exception | None = None
    for mod, fn in tries:
        try:
            m = __import__(mod, fromlist=[fn])
            return getattr(m, fn)
        except Exception as e:
            last = e
    raise ImportError("Impossible d'importer selection_cibles_component") from last


# ------------------------------------------------------
# Paths
# ------------------------------------------------------
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


def _find_mesures_completes_xlsx(common_data_dir: Path) -> Optional[Path]:
    if not common_data_dir.exists():
        return None
    needles = ["mesures completes", "mesures complètes", "mesures complete", "mesures compl"]
    cands: list[Path] = []
    for p in common_data_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".xlsx" and any(n in p.stem.lower() for n in needles):
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def _resolve_points_workbook(cst_workbook_path: str | None) -> Optional[Path]:
    root = _project_root()

    if cst_workbook_path:
        p = Path(cst_workbook_path).expanduser()
        if not p.is_absolute():
            p = (root / p).resolve()
        else:
            p = p.resolve()
        if p.exists() and p.is_file() and p.suffix.lower() == ".xlsx":
            return p

    common = _common_data_dir()
    mp = _find_mesures_completes_xlsx(common)
    if mp and mp.exists():
        return mp
    return None


# ------------------------------------------------------
# Cache key helpers
# ------------------------------------------------------
def _get_dataset_hash(kind: str) -> str:
    dh = st.session_state.get("data_hash", {})
    if isinstance(dh, dict):
        return str(dh.get(kind, "") or "")
    return ""


def _hash_to_float(sig: str) -> float:
    if not sig:
        return 0.0
    try:
        return float(int(sig[:12], 16))
    except Exception:
        h = hashlib.sha1(sig.encode("utf-8")).hexdigest()
        return float(int(h[:12], 16))


def _file_sig(p: Path) -> str:
    try:
        stt = p.stat()
        return f"{p.resolve()}|{int(stt.st_mtime)}|{int(stt.st_size)}"
    except Exception:
        return f"{p}"


def _ui_tag(points_wb: Optional[Path], dataset_hash: str, plan_src: Optional[Path], dummy: int) -> str:
    """
    Tag utilisé:
    - pour différencier la key du composant (remount)
    - pour éviter des états fantômes quand les données changent
    """
    parts: list[str] = [f"dummy:{int(dummy or 0)}"]
    if dataset_hash:
        parts.append(f"dh:{dataset_hash}")
    if points_wb and points_wb.exists():
        parts.append(f"wb:{_file_sig(points_wb)}")
    if plan_src and plan_src.exists():
        parts.append(f"bg:{_file_sig(plan_src)}")
    raw = "||".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


# ------------------------------------------------------
# Fit-to-canvas mapping
# ------------------------------------------------------
def _robust_bounds(vals: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> Tuple[float, float]:
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(v, q_low))
    hi = float(np.percentile(v, q_high))
    if hi - lo < 1e-12:
        lo = float(np.min(v))
        hi = float(np.max(v))
        if hi - lo < 1e-12:
            hi = lo + 1.0
    return lo, hi


def _make_mapper(points: pd.DataFrame, width: int, height: int):
    w = float(width)
    h = float(height)
    pad = 35.0
    inner_w = max(w - 2 * pad, 1.0)
    inner_h = max(h - 2 * pad, 1.0)

    xs = points["x"].to_numpy(dtype=float)
    ys = points["y"].to_numpy(dtype=float)

    bx0, bx1 = _robust_bounds(xs)
    by0, by1 = _robust_bounds(ys)

    span_x = max(bx1 - bx0, 1e-12)
    span_y = max(by1 - by0, 1e-12)

    s = min(inner_w / span_x, inner_h / span_y)

    content_w = span_x * s
    content_h = span_y * s
    off_x = (w - content_w) / 2.0
    off_y = (h - content_h) / 2.0

    def clamp(v: float, a: float, b: float) -> float:
        return a if v < a else (b if v > b else v)

    def to_px(x: float, y: float) -> Tuple[float, float]:
        cx = clamp(x, bx0, bx1)
        cy = clamp(y, by0, by1)
        px = off_x + (cx - bx0) * s
        py = off_y + (by1 - cy) * s
        return float(px), float(py)

    return to_px


def _last_xyz(df: pd.DataFrame) -> Optional[Tuple[float, float, float]]:
    if df is None or df.empty:
        return None
    if not {"x", "y"}.issubset(df.columns):
        return None
    sub = df.dropna(subset=["x", "y"], how="any")
    if sub.empty:
        return None
    last = sub.iloc[-1]
    x = float(last["x"])
    y = float(last["y"])
    z = float(last["z"]) if ("z" in sub.columns and pd.notna(last.get("z"))) else 0.0
    return x, y, z


@st.cache_data(show_spinner=False)
def _load_points_fit_cached(workbook_path: str, cache_key: str, W: int, H: int) -> list[dict[str, Any]]:
    """
    cache_key sert à invalider le cache Streamlit.
    On le convertit aussi en float "mtime_like" pour invalider les LRU internes du reader.
    """
    mtime_like = _hash_to_float(cache_key)

    names = list_targets_in_mesures(workbook_path, sheet_name=None, mtime=mtime_like)
    if not names:
        return []
    ts = read_targets_timeseries(workbook_path, names, sheet_name=None, mtime=mtime_like)

    rows: list[tuple[str, float, float, float]] = []
    for name, df in ts.items():
        xyz = _last_xyz(df)
        if xyz is None:
            continue
        x, y, z = xyz
        rows.append((str(name), x, y, z))

    if not rows:
        return []

    raw = pd.DataFrame(rows, columns=["name", "x", "y", "z"])
    to_px = _make_mapper(raw, W, H)

    out: list[dict[str, Any]] = []
    for _, r in raw.iterrows():
        px, py = to_px(float(r["x"]), float(r["y"]))
        out.append({"name": str(r["name"]), "px": px, "py": py, "z": float(r["z"])})
    return out


# ------------------------------------------------------
# Background plan: cache sur disque (FAST)
# ------------------------------------------------------
def _pick_latest_plan_file(common: Path) -> Optional[Path]:
    if not common.exists():
        return None
    exts = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    cands = [p for p in common.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _bg_cache_dir(common: Path) -> Path:
    d = common / "_bg_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bg_cache_key(src: Path) -> str:
    stt = src.stat()
    return hashlib.sha1(f"{src.resolve()}|{int(stt.st_mtime)}|{stt.st_size}".encode("utf-8")).hexdigest()[:16]


def _ensure_cached_bg_jpg(src: Path, W: int, H: int) -> Optional[Path]:
    """
    Convertit (si besoin) le PDF en JPEG et l'écrit dans _bg_cache.
    Une seule fois par (path+mtime+size) + taille canvas.
    """
    common = _common_data_dir()
    out_dir = _bg_cache_dir(common)
    key = _bg_cache_key(src)
    out = out_dir / f"fond_{key}_{W}x{H}.jpg"
    if out.exists() and out.stat().st_size > 10_000:
        return out

    ext = src.suffix.lower()

    if ext in {".jpg", ".jpeg"}:
        try:
            if src.resolve() != out.resolve():
                out.write_bytes(src.read_bytes())
            return out
        except Exception:
            return None

    if ext in {".png", ".webp"}:
        try:
            from PIL import Image  # type: ignore
            im = Image.open(src)
            im = im.convert("RGB")
            im.thumbnail((int(W * 1.2), int(H * 1.2)))
            im.save(out, format="JPEG", quality=72, optimize=True)
            return out
        except Exception:
            return None

    if ext == ".pdf":
        try:
            import fitz  # type: ignore
            doc = fitz.open(str(src))
            page = doc.load_page(0)
            rect = page.rect

            zx = (W * 1.2) / max(rect.width, 1.0)
            zy = (H * 1.2) / max(rect.height, 1.0)
            z = float(min(zx, zy))
            z = max(0.5, min(z, 3.0))

            mat = fitz.Matrix(z, z)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out.write_bytes(pix.tobytes("jpeg"))
            return out
        except Exception:
            return None

    return None


@st.cache_data(show_spinner=False)
def _jpg_to_data_url_cached(jpg_path_str: str, mtime_int: int) -> tuple[Optional[str], int, int]:
    p = Path(jpg_path_str)
    if not p.exists():
        return None, 0, 0

    b = p.read_bytes()
    url = "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")

    w = h = 0
    try:
        from PIL import Image  # type: ignore
        im = Image.open(p)
        w, h = im.size
    except Exception:
        pass

    _ = mtime_int  # clé cache
    return url, int(w or 0), int(h or 0)


# ------------------------------------------------------
# BG transform persistence (Option 1 = backend truth)
# ------------------------------------------------------
def _bg_transform_store_path(common: Path) -> Path:
    p = common / "_bg_transform_store.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _bg_transform_key(plan_src: Optional[Path], W: int, H: int) -> str:
    if not plan_src or not plan_src.exists():
        return f"none|{W}x{H}"
    return f"{_file_sig(plan_src)}|{W}x{H}"


def _load_bg_transform_from_disk(common: Path, key: str) -> Optional[dict[str, float]]:
    p = _bg_transform_store_path(common)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        v = data.get(key)
        if not isinstance(v, dict):
            return None
        return {
            "x": float(v.get("x", 0.0)),
            "y": float(v.get("y", 0.0)),
            "scale": float(v.get("scale", 1.0)),
        }
    except Exception:
        return None


def _save_bg_transform_to_disk(common: Path, key: str, tr: dict[str, float]) -> None:
    p = _bg_transform_store_path(common)
    try:
        data: dict[str, Any] = {}
        if p.exists():
            prev = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                data = prev
        data[key] = {
            "x": float(tr.get("x", 0.0)),
            "y": float(tr.get("y", 0.0)),
            "scale": float(tr.get("scale", 1.0)),
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # non bloquant
        return


# ------------------------------------------------------
# Zone colors
# ------------------------------------------------------
_ZONE_PALETTE = ["#146EFF", "#22C55E", "#F97316", "#A855F7", "#EF4444", "#06B6D4", "#EAB308", "#EC4899"]


def _zone_color(i: int) -> str:
    return _ZONE_PALETTE[i % len(_ZONE_PALETTE)]


# ------------------------------------------------------
# ✅ LIVE_VEC: vecState -> alpha_backend
# ------------------------------------------------------
def _alpha_from_vecstate(vs: Any) -> Optional[float]:
    """
    Convertit un vecState front (x0,y0,x1,y1) -> angle deg (-180,180].
    On n'impose pas custom=True ici: si lock passe custom, ok.
    Si deleted => None.
    """
    if not isinstance(vs, dict):
        return None
    try:
        if bool(vs.get("deleted")):
            return None
        x0 = vs.get("x0", None)
        y0 = vs.get("y0", None)
        x1 = vs.get("x1", None)
        y1 = vs.get("y1", None)
        if x0 is None or y0 is None or x1 is None or y1 is None:
            return None
        dx = float(x1) - float(x0)
        dy = float(y1) - float(y0)
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return None
        a = math.degrees(math.atan2(dy, dx))
        # normalize (-180, 180]
        while a <= -180:
            a += 360
        while a > 180:
            a -= 360
        return float(a)
    except Exception:
        return None


# ------------------------------------------------------
# Snapshot -> JSON
# ------------------------------------------------------
def _apply_snapshot(
    mgr: CoupesManager,
    coupes: list[Coupe],
    snapshot: list[dict[str, Any]],
    zone_color_by_ui: dict[int, str],
) -> int:
    by_ui: dict[int, Coupe] = {}
    for c in coupes:
        try:
            by_ui[int(getattr(c, "ui_idx", 0))] = c
        except Exception:
            continue

    updated = 0
    for s in snapshot:
        try:
            ui = int(s.get("idx"))
        except Exception:
            continue

        c = by_ui.get(ui)
        if c is None:
            continue

        alpha = s.get("alpha", None)
        targets = s.get("targets", None)

        angle_final = float(alpha) if alpha is not None else float(getattr(c, "angle_deg", 0.0) or 0.0)
        targets_final = [str(x).strip() for x in (targets or []) if str(x).strip()]
        targets_final = sorted(set(targets_final), key=lambda x: x.lower())

        color_final = str(zone_color_by_ui.get(ui, "") or "").strip()
        prev_color = str(getattr(c, "color", "") or "").strip()

        angle_changed = float(getattr(c, "angle_deg", 0.0) or 0.0) != float(angle_final)
        targets_changed = list(getattr(c, "targets", []) or []) != list(targets_final)
        color_changed = prev_color != color_final and color_final != ""

        if not angle_changed and not targets_changed and not color_changed:
            continue

        cnew = Coupe(
            name=getattr(c, "name", ""),
            angle_deg=angle_final,
            targets=targets_final,
            ui_idx=ui,
        )
        try:
            setattr(cnew, "color", color_final)
        except Exception:
            pass

        mgr.update_coupe(getattr(c, "name", ""), cnew)
        updated += 1

    return updated


# ======================================================
# Public entrypoint
# ======================================================
def render_selection_cibles(
    zone_name: str = "GLOBAL",
    cst_workbook_path: str | None = None,
    dummy: int = 0,
) -> None:
    W, H = 1200, 680
    _ = zone_name

    mgr = CoupesManager()
    coupes = mgr.list_coupes()
    if not coupes:
        st.warning("Aucune coupe dans le JSON.")
        return

    # ✅ LIVE_VEC: store init
    if "live_vec_by_ui" not in st.session_state or not isinstance(st.session_state.get("live_vec_by_ui"), dict):
        st.session_state["live_vec_by_ui"] = {}
    live_vec_by_ui: dict[int, Any] = st.session_state["live_vec_by_ui"]

    # Background (fast path)
    common = _common_data_dir()
    plan_src = _pick_latest_plan_file(common)

    # Points
    points_workbook = _resolve_points_workbook(cst_workbook_path)
    dataset_hash = _get_dataset_hash(DATASET_KIND_FOR_POINTS)

    # Tag UI => remount component if inputs changed
    ui_tag = _ui_tag(points_workbook, dataset_hash, plan_src, dummy)

    points: list[dict[str, Any]] = []
    if points_workbook and points_workbook.exists():
        pth = str(points_workbook)
        if dataset_hash:
            cache_key = dataset_hash
        else:
            stt = points_workbook.stat()
            cache_key = f"mtime:{stt.st_mtime:.6f}|size:{int(stt.st_size)}|dummy:{int(dummy or 0)}|path:{pth}"
        points = _load_points_fit_cached(pth, cache_key, W, H)

    # ---- BG transform (Option 1 = backend truth) ----
    bg_key = _bg_transform_key(plan_src, W, H)

    # 1) disque (persistant)
    bg_transform = _load_bg_transform_from_disk(common, bg_key)

    # 2) fallback session_state
    if not isinstance(bg_transform, dict):
        bg_transform = st.session_state.get("bg_transform")
        if not isinstance(bg_transform, dict):
            bg_transform = {"x": 0.0, "y": 0.0, "scale": 1.0}

    # BG payload
    bg_payload: dict[str, Any] = {"enabled": False}
    if plan_src and plan_src.exists():
        cached_jpg = _ensure_cached_bg_jpg(plan_src, W=W, H=H)
        if cached_jpg and cached_jpg.exists():
            mtime_int = int(cached_jpg.stat().st_mtime)
            url, iw, ih = _jpg_to_data_url_cached(str(cached_jpg), mtime_int)
            if url:
                bg_payload = {
                    "enabled": True,
                    "img_url": url,
                    "img_w": int(iw or W),
                    "img_h": int(ih or H),
                    "transform": {
                        "x": float(bg_transform.get("x", 0.0)),
                        "y": float(bg_transform.get("y", 0.0)),
                        "scale": float(bg_transform.get("scale", 1.0)),
                    },
                }

    # Zones
    zones: list[dict[str, Any]] = []
    zone_color_by_ui: dict[int, str] = {}

    for i, c in enumerate(coupes):
        ui = int(getattr(c, "ui_idx", i))
        zid = str(ui)
        col = _zone_color(i)
        zone_color_by_ui[ui] = col

        # ✅ LIVE_VEC: calc alpha_backend from last live vecstate for this zone (if any)
        a_live = _alpha_from_vecstate(live_vec_by_ui.get(ui))

        zones.append(
            {
                "id": zid,
                "name": getattr(c, "name", ""),
                "idx": ui,
                "color": col,
                "mem_quad": f"quad_{zid}",
                # ✅ LIVE_VEC: we send both, front has priority alpha_backend then angle_deg
                "alpha_backend": a_live,
                "angle_deg": float(getattr(c, "angle_deg", 0.0) or 0.0),
            }
        )

    data = {"w": W, "h": H, "zones": zones, "points": points, "bg": bg_payload, "ui_tag": ui_tag}

    selection_cibles_component = _import_selection_component()

    # ✅ Key includes ui_tag to remount when data/bg changes (prevents sticky component value)
    comp_key = f"selection_cibles_component_main_{ui_tag}"
    ret = selection_cibles_component(data=data, key=comp_key)

    # ✅ LIVE_VEC: handle live vector updates (NO st.rerun here)
    if isinstance(ret, dict) and ret.get("type") == "live_vec":
        click_id = str(ret.get("click_id") or "").strip()
        if click_id:
            last_live = str(st.session_state.get("_sel_cibles_last_live_id") or "")
            if click_id != last_live:
                st.session_state["_sel_cibles_last_live_id"] = click_id

                try:
                    ui = int(ret.get("zone_idx"))
                except Exception:
                    ui = None

                vs = ret.get("vecState")
                if ui is not None and isinstance(vs, dict):
                    # store raw vecState
                    live_vec_by_ui[ui] = vs
                    st.session_state["live_vec_by_ui"] = live_vec_by_ui
        # IMPORTANT: on s'arrête là (pas de JSON write, pas de rerun explicite)
        return

    # ✅ Anti-boucle rerun: handle click once
    if isinstance(ret, dict) and ret.get("type") == "front_button":
        click_id = str(ret.get("click_id") or "").strip()
        if not click_id:
            st.error("Front: click_id manquant (mise à jour impossible).")
            return

        last_click = str(st.session_state.get("_sel_cibles_last_click_id") or "")
        if click_id == last_click:
            # Déjà traité => surtout ne pas rerun en boucle
            return

        st.session_state["_sel_cibles_last_click_id"] = click_id

        snapshot = ret.get("snapshot")
        if not isinstance(snapshot, list):
            st.error("Le front n'a pas envoyé le snapshot.")
            return

        # ✅ Save bg_transform to session + DISK (so it survives full reload)
        bg_t = ret.get("bg_transform")
        if isinstance(bg_t, dict):
            new_tr = {
                "x": float(bg_t.get("x", 0.0)),
                "y": float(bg_t.get("y", 0.0)),
                "scale": float(bg_t.get("scale", 1.0)),
            }
            st.session_state["bg_transform"] = new_tr
            _save_bg_transform_to_disk(common, bg_key, new_tr)

        updated = _apply_snapshot(mgr, coupes, snapshot, zone_color_by_ui)
        if updated == 0:
            st.toast("Aucune modification à écrire.", icon="ℹ️")
        else:
            st.toast(f"JSON mis à jour : {updated} coupe(s) ✅", icon="✅")

        st.rerun()
