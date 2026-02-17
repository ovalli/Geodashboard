from __future__ import annotations
from pathlib import Path
from typing import Any
import base64, hashlib, json, math
import numpy as np
import pandas as pd
import streamlit as st

from src.io.coupes_manager import CoupesManager, Coupe
from src.pipeline.mesures_completes_reader import list_targets_in_mesures, read_targets_timeseries

DATASET_KIND = "topo"
W, H = 1200, 680

_ZONE_PALETTE = ["#146EFF", "#22C55E", "#F97316", "#A855F7", "#EF4444", "#06B6D4", "#EAB308", "#EC4899"]


def project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "data" / "common_data").exists() or (p / "app.py").exists():
            return p
    return here.parents[2]


common_dir = lambda: (d := project_root() / "data" / "common_data") and d.mkdir(parents=True, exist_ok=True) or d


def latest_xlsx() -> Path | None:
    d = common_dir()
    if not d.exists(): return None
    cands = [p for p in d.iterdir() if p.suffix.lower() == ".xlsx" and "mesur" in p.stem.lower()]
    return max(cands, key=lambda p: p.stat().st_mtime, default=None) if cands else None


def resolve_workbook(cst_path: str | None) -> Path | None:
    if cst_path:
        p = Path(cst_path).expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".xlsx": return p
    return latest_xlsx()


def file_sig(p: Path) -> str:
    try:
        s = p.stat()
        return f"{p.resolve()}|{int(s.st_mtime)}|{int(s.st_size)}"
    except:
        return str(p)


def ui_tag(points_wb: Path | None, ds_hash: str, plan: Path | None, dummy: int) -> str:
    parts = [f"dummy:{dummy}"]
    if ds_hash: parts.append(f"dh:{ds_hash}")
    if points_wb: parts.append(f"wb:{file_sig(points_wb)}")
    if plan: parts.append(f"bg:{file_sig(plan)}")
    return hashlib.sha1("||".join(parts).encode()).hexdigest()[:12]


# ─── Mapping canvas ──────────────────────────────────────────────────────────

def robust_bounds(v: np.ndarray, q1=1.0, q99=99.0) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    if len(v) == 0: return 0., 1.
    lo, hi = np.percentile(v, [q1, q99])
    if hi - lo < 1e-12:
        lo, hi = float(v.min()), float(v.max())
    return lo, hi or lo + 1


def make_mapper(df: pd.DataFrame, w: int, h: int):
    pad, W, H = 35.0, float(w), float(h)
    iw = max(W - 2 * pad, 1.0)
    ih = max(H - 2 * pad, 1.0)
    x0, x1 = robust_bounds(df["x"].to_numpy())
    y0, y1 = robust_bounds(df["y"].to_numpy())
    s = min(iw / (x1 - x0 or 1), ih / (y1 - y0 or 1))
    ox = (W - (x1 - x0) * s) / 2
    oy = (H - (y1 - y0) * s) / 2

    def to_px(x: float, y: float) -> tuple[float, float]:
        x = max(x0, min(x1, x))
        y = max(y0, min(y1, y))
        return ox + (x - x0) * s, oy + (y1 - y) * s

    return to_px


def last_xyz(df: pd.DataFrame) -> tuple[float, float, float] | None:
    if df is None or df.empty or not {"x", "y"} <= set(df.columns):
        return None
    row = df.dropna(subset=["x", "y"]).iloc[-1]
    return float(row["x"]), float(row["y"]), float(row.get("z") or 0)


@st.cache_data(show_spinner=False)
def load_points_cached(path: str, cache_key: str, W: int, H: int) -> list[dict]:
    # petite astuce pour avoir un float "temps-like" à partir du hash
    mtime_like = int(hashlib.sha1(cache_key.encode()).hexdigest()[:12], 16) / 1e12

    names = list_targets_in_mesures(path, mtime=mtime_like)
    if not names:
        return []

    series = read_targets_timeseries(path, names, mtime=mtime_like)
    rows = []
    for name, df in series.items():
        xyz = last_xyz(df)
        if xyz:
            rows.append((str(name), *xyz))

    if not rows:
        return []

    df = pd.DataFrame(rows, columns=["name", "x", "y", "z"])
    mapper = make_mapper(df, W, H)

    result = []
    for _, r in df.iterrows():
        px, py = mapper(float(r["x"]), float(r["y"]))
        result.append({
            "name": str(r["name"]),
            "px": px,
            "py": py,
            "z": float(r["z"])
        })

    return result


# ─── Background image caching ────────────────────────────────────────────────

def latest_plan() -> Path | None:
    exts = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    cands = [p for p in common_dir().iterdir() if p.is_file() and p.suffix.lower() in exts]
    return max(cands, key=lambda p: p.stat().st_mtime, default=None) if cands else None


def bg_cache_path(src: Path, W: int, H: int) -> Path:
    key = hashlib.sha1(file_sig(src).encode()).hexdigest()[:16]
    (common_dir() / "_bg_cache").mkdir(parents=True, exist_ok=True)
    return common_dir() / "_bg_cache" / f"fond_{key}_{W}x{H}.jpg"


def ensure_cached_bg(src: Path, W: int, H: int) -> Path | None:
    out = bg_cache_path(src, W, H)
    if out.exists() and out.stat().st_size > 10000:
        return out

    ext = src.suffix.lower()

    if ext in {".jpg", ".jpeg"}:
        out.write_bytes(src.read_bytes())
        return out

    try:
        from PIL import Image

        if ext in {".png", ".webp"}:
            im = Image.open(src).convert("RGB")
            im.thumbnail((int(W * 1.2), int(H * 1.2)))
            im.save(out, "JPEG", quality=72, optimize=True)
            return out

        if ext == ".pdf":
            import fitz
            doc = fitz.open(str(src))
            page = doc[0]
            z = min(W * 1.2 / page.rect.width, H * 1.2 / page.rect.height)
            z = max(0.5, min(z, 3.0))
            pix = page.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
            out.write_bytes(pix.tobytes("jpeg"))
            return out
    except Exception:
        pass

    return None


@st.cache_data(show_spinner=False)
def jpg_to_dataurl(path_str: str, _mtime: int):
    p = Path(path_str)
    if not p.exists():
        return None, 0, 0
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    try:
        from PIL import Image
        w, h = Image.open(p).size
    except:
        w = h = 0
    return f"data:image/jpeg;base64,{b64}", int(w), int(h)


# ─── BG transform persistence ────────────────────────────────────────────────

def bg_transform_path() -> Path:
    p = common_dir() / "_bg_transform_store.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def bg_key(plan: Path | None, W: int, H: int) -> str:
    return f"none|{W}x{H}" if not plan or not plan.exists() else f"{file_sig(plan)}|{W}x{H}"


def load_bg_transform(key: str) -> dict[str, float]:
    p = bg_transform_path()
    if not p.exists():
        return {"x": 0.0, "y": 0.0, "scale": 1.0}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        v = data.get(key, {"x": 0, "y": 0, "scale": 1})
        return {"x": float(v.get("x", 0)), "y": float(v.get("y", 0)), "scale": float(v.get("scale", 1))}
    except:
        return {"x": 0.0, "y": 0.0, "scale": 1.0}


def save_bg_transform(key: str, tr: dict):
    p = bg_transform_path()
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    data[key] = {
        "x": float(tr.get("x", 0.0)),
        "y": float(tr.get("y", 0.0)),
        "scale": float(tr.get("scale", 1.0))
    }
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── Main render ─────────────────────────────────────────────────────────────

def render_selection_cibles(zone_name="GLOBAL", cst_workbook_path=None, dummy=0):
    mgr = CoupesManager()
    coupes = mgr.list_coupes()
    if not coupes:
        st.warning("Aucune coupe dans le JSON.")
        return

    # Live vectors
    if "live_vec_by_ui" not in st.session_state:
        st.session_state.live_vec_by_ui = {}
    live_vec = st.session_state.live_vec_by_ui

    plan = latest_plan()
    wb = resolve_workbook(cst_workbook_path)
    ds_hash = st.session_state.get("data_hash", {}).get(DATASET_KIND, "")
    tag = ui_tag(wb, ds_hash, plan, dummy)

    points = []
    if wb and wb.exists():
        cache_key = ds_hash or f"mtime:{wb.stat().st_mtime:.6f}|sz:{wb.stat().st_size}|d{dummy}"
        points = load_points_cached(str(wb), cache_key, W, H)

    bg_key_str = bg_key(plan, W, H)
    bg_tr = load_bg_transform(bg_key_str)

    bg = {"enabled": False}
    if plan and plan.exists():
        jpg = ensure_cached_bg(plan, W, H)
        if jpg and jpg.exists():
            url, iw, ih = jpg_to_dataurl(str(jpg), int(jpg.stat().st_mtime))
            if url:
                bg = {
                    "enabled": True,
                    "img_url": url,
                    "img_w": iw or W,
                    "img_h": ih or H,
                    "transform": bg_tr
                }

    zones, color_by_ui = [], {}
    for i, c in enumerate(coupes):
        ui = int(getattr(c, "ui_idx", i))
        col = _ZONE_PALETTE[i % len(_ZONE_PALETTE)]
        color_by_ui[ui] = col

        # alpha from live vector (if any)
        a_live = None
        vs = live_vec.get(ui)
        if isinstance(vs, dict) and not vs.get("deleted"):
            try:
                x0, y0 = float(vs["x0"]), float(vs["y0"])
                x1, y1 = float(vs["x1"]), float(vs["y1"])
                dx, dy = x1 - x0, y1 - y0
                if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                    a = math.degrees(math.atan2(dy, dx))
                    a_live = ((a + 180) % 360) - 180
            except:
                pass

        zones.append({
            "id": str(ui),
            "name": getattr(c, "name", ""),
            "idx": ui,
            "color": col,
            "mem_quad": f"quad_{ui}",
            "alpha_backend": a_live,
            "angle_deg": float(getattr(c, "angle_deg", 0.0) or 0.0)
        })

    data = {
        "w": W,
        "h": H,
        "zones": zones,
        "points": points,
        "bg": bg,
        "ui_tag": tag
    }

    # Import du composant
    try:
        from src.ui.selection_cibles_html import selection_cibles_component
    except ImportError:
        from src.ui.selection_cibles_component import selection_cibles_component

    ret = selection_cibles_component(data=data, key=f"sel_cibles_{tag}")

    if not isinstance(ret, dict):
        return

    # ─── Live vector update (pas de rerun) ───────────────────────────────
    if ret.get("type") == "live_vec":
        cid = str(ret.get("click_id", "")).strip()
        if not cid or cid == st.session_state.get("_last_live_id"):
            return
        st.session_state["_last_live_id"] = cid
        ui = ret.get("zone_idx")
        vs = ret.get("vecState")
        if ui is not None and isinstance(vs, dict):
            live_vec[int(ui)] = vs
        return

    # ─── Button click ────────────────────────────────────────────────────
    if ret.get("type") != "front_button":
        return

    cid = str(ret.get("click_id", "")).strip()
    if not cid or cid == st.session_state.get("_last_click_id"):
        return
    st.session_state["_last_click_id"] = cid

    # Sauvegarde transform background
    if isinstance(bg_t := ret.get("bg_transform"), dict):
        tr = {
            "x": float(bg_t.get("x", 0.0)),
            "y": float(bg_t.get("y", 0.0)),
            "scale": float(bg_t.get("scale", 1.0))
        }
        st.session_state["bg_transform"] = tr
        save_bg_transform(bg_key_str, tr)

    snapshot = ret.get("snapshot")
    if not isinstance(snapshot, list):
        st.error("Pas de snapshot reçu du composant")
        return

    updated = 0
    by_ui = {int(getattr(c, "ui_idx", i)): c for i, c in enumerate(coupes)}

    for s in snapshot:
        try:
            ui = int(s["idx"])
            c = by_ui.get(ui)
            if not c:
                continue

            angle = float(s.get("alpha") or getattr(c, "angle_deg", 0.0))
            targets = sorted({str(x).strip() for x in s.get("targets", []) if str(x).strip()})
            color = color_by_ui.get(ui, getattr(c, "color", ""))

            if (abs(float(getattr(c, "angle_deg", 0.0)) - angle) < 1e-5 and
                getattr(c, "targets", []) == targets and
                getattr(c, "color", "") == color):
                continue

            new_c = Coupe(
                name=getattr(c, "name", ""),
                angle_deg=angle,
                targets=targets,
                ui_idx=ui
            )
            setattr(new_c, "color", color)
            mgr.update_coupe(getattr(c, "name", ""), new_c)
            updated += 1

        except Exception:
            continue

    if updated:
        st.toast(f"JSON mis à jour : {updated} coupe(s) modifiée(s)", icon="✅")
    else:
        st.toast("Aucune modification détectée", icon="ℹ️")

    st.rerun()