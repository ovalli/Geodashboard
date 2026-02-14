from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import tempfile
import unicodedata
from typing import Any

import numpy as np


# -----------------------------
# Utils
# -----------------------------
def _norm_key(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")
    s = " ".join(s.strip().split())
    return s.casefold()


def find_project_root(workbook_path: str | None = None) -> Path:
    # Try near workbook
    if workbook_path:
        try:
            wp = Path(workbook_path).resolve()
            base = wp if wp.is_dir() else wp.parent
            for p in [base, *base.parents]:
                if (p / "data" / "common_data").exists():
                    return p
                if (p / "app.py").exists():
                    return p
        except Exception:
            pass

    # Fallback near this file
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "data" / "common_data").exists():
            return p
        if (p / "app.py").exists():
            return p

    return here.parents[2]


def parametres_generaux_json_path(workbook_path: str | None = None) -> Path | None:
    root = find_project_root(workbook_path)
    p = root / "data" / "common_data" / "Parametres_Generaux.json"
    return p if p.exists() else None


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_save_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp.json", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def find_angle_param_key(pg: dict) -> str | None:
    params = pg.get("parametres")
    if not isinstance(params, dict):
        return None

    target_label = "Angle orienté (X n), trigo"

    for k, meta in params.items():
        if isinstance(meta, dict) and str(meta.get("label", "")).strip() == target_label:
            return str(k)

    for k, meta in params.items():
        if not isinstance(meta, dict):
            continue
        lbl = str(meta.get("label", "")).lower()
        if "angle" in lbl and "trigo" in lbl:
            return str(k)

    return None


def list_coupes(pg: dict) -> list[str]:
    coupes = pg.get("coupes")
    if not isinstance(coupes, dict):
        return []
    out = [str(k) for k in coupes.keys() if str(k).strip()]
    out.sort(key=lambda s: s.lower())
    return out


def resolve_existing_coupe_key(pg: dict, coupe_name: str) -> str:
    coupes = pg.get("coupes")
    if not isinstance(coupes, dict):
        raise KeyError("JSON invalide: 'coupes' absent ou pas un dict.")

    if coupe_name in coupes:
        return coupe_name

    wanted = _norm_key(coupe_name)
    if not wanted:
        raise KeyError("Coupe vide.")

    for k in coupes.keys():
        if _norm_key(k) == wanted:
            return str(k)

    raise KeyError(f"Coupe introuvable dans JSON: {coupe_name!r}")


def read_angle(pg: dict, coupe_name: str, angle_key: str) -> float | None:
    coupes = pg.get("coupes")
    if not isinstance(coupes, dict):
        return None
    block = coupes.get(coupe_name)
    if not isinstance(block, dict):
        return None
    cell = block.get(angle_key)
    if not isinstance(cell, dict):
        return None
    v = cell.get("value", None)
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def write_angles_bulk(pg_path: Path, angle_key: str, updates: dict[str, float], source: str = "ui_apply_all") -> int:
    """
    1 seule écriture JSON (atomic).
    updates: coupe_name -> new_value
    """
    pg = json_load(pg_path)
    if "coupes" not in pg or not isinstance(pg.get("coupes"), dict):
        raise KeyError("JSON invalide: 'coupes' absent.")

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    changed = 0
    for coupe_raw, new_val in updates.items():
        coupe_key = resolve_existing_coupe_key(pg, coupe_raw)

        if coupe_key not in pg["coupes"] or not isinstance(pg["coupes"][coupe_key], dict):
            pg["coupes"][coupe_key] = {}

        if angle_key not in pg["coupes"][coupe_key] or not isinstance(pg["coupes"][coupe_key].get(angle_key), dict):
            pg["coupes"][coupe_key][angle_key] = {}

        pg["coupes"][coupe_key][angle_key]["value"] = float(new_val)
        pg["coupes"][coupe_key][angle_key]["last_update"] = now
        pg["coupes"][coupe_key][angle_key]["source"] = source
        changed += 1

    if "meta" not in pg or not isinstance(pg.get("meta"), dict):
        pg["meta"] = {}
    pg["meta"]["updated_at"] = now

    json_save_atomic(pg_path, pg)
    return changed
