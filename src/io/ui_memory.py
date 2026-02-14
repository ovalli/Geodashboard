from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "data" / "common_data").exists():
            return p
        if (p / "app.py").exists():
            return p
    return here.parents[2]


def get_memory_path() -> Path:
    root = get_project_root()
    common = root / "data" / "common_data"
    common.mkdir(parents=True, exist_ok=True)
    return common / "ui_memory.json"


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_memory() -> Dict[str, Any]:
    path = get_memory_path()
    if not path.exists():
        return {"version": 1, "updated_at": _now_iso(), "shapes": {}}

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
        if not isinstance(data, dict):
            raise ValueError("ui_memory.json invalide (root non dict)")
    except Exception:
        return {"version": 1, "updated_at": _now_iso(), "shapes": {}}

    data.setdefault("version", 1)
    data.setdefault("updated_at", _now_iso())
    data.setdefault("shapes", {})
    if not isinstance(data["shapes"], dict):
        data["shapes"] = {}
    return data


def save_memory(mem: Dict[str, Any]) -> None:
    path = get_memory_path()
    mem = dict(mem)
    mem["updated_at"] = _now_iso()

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomique


def get_shape(scope: str, key: str) -> Optional[Dict[str, Any]]:
    mem = load_memory()
    shapes = mem.get("shapes", {})
    return shapes.get(scope, {}).get(key)


def set_shape(scope: str, key: str, shape: Dict[str, Any]) -> None:
    mem = load_memory()
    shapes = mem.setdefault("shapes", {})
    shapes.setdefault(scope, {})[key] = shape
    save_memory(mem)
