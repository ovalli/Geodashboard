# ======================================================
# src/io/coupes_manager.py  (COMPLET)
# ✅ JSON-only : ne lit / ne cherche JAMAIS aucun Excel (donc jamais CST)
# ✅ CRUD complet : list/get/add/update/rename/delete
# ✅ Compatible UI : Coupe(name, angle_deg, targets, ui_idx, color)
# ✅ Conserve les autres clés du JSON (ne réécrit pas tout n'importe comment)
# ======================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import tempfile


@dataclass
class Coupe:
    name: str
    angle_deg: float | None = None
    targets: List[str] | None = None
    ui_idx: int | None = None
    color: str | None = None


class CoupesManager:
    """
    Manager JSON-only.
    Source unique : data/common_data/Parametres_Generaux.json (ou variantes)
    """

    def __init__(self, json_path: Path | None = None, project_root: Path | None = None) -> None:
        self.project_root = Path(project_root) if project_root else self._project_root()
        self.pg_path: Path = Path(json_path) if json_path else self._default_json_path(self.project_root)

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def list_coupes(self) -> List[Coupe]:
        data, key = self._load_with_key()
        items = data.get(key, [])
        if not isinstance(items, list):
            return []
        coupes = [self._coupe_from_obj(o) for o in items]
        coupes = [c for c in coupes if c is not None]
        # tri stable : ui_idx si présent, sinon ordre fichier
        def sort_key(c: Coupe) -> Tuple[int, str]:
            ui = c.ui_idx if isinstance(c.ui_idx, int) else 10**9
            return (ui, (c.name or "").lower())
        return sorted(coupes, key=sort_key)

    def get_coupe(self, name: str) -> Coupe:
        name = str(name).strip()
        for c in self.list_coupes():
            if c.name == name:
                return c
        raise KeyError(f"Coupe introuvable: {name}")

    def add_coupe(self, name: str) -> Coupe:
        name = str(name).strip()
        if not name:
            raise ValueError("Nom de coupe vide.")
        data, key = self._load_with_key()

        items = data.get(key, [])
        if not isinstance(items, list):
            items = []
            data[key] = items

        # unicité nom
        for o in items:
            if isinstance(o, dict) and str(o.get("name", "")).strip() == name:
                raise ValueError(f"Coupe déjà existante: {name}")

        # ui_idx auto (max+1)
        next_ui = self._next_ui_idx(items)

        obj = {
            "name": name,
            "angle_deg": 0.0,
            "targets": [],
            "ui_idx": next_ui,
            "color": None,
        }
        items.append(obj)
        self._save(data)

        return self._coupe_from_obj(obj)  # type: ignore[return-value]

    def update_coupe(self, name: str, new_coupe: Coupe) -> None:
        """
        Remplace le contenu de la coupe 'name' par new_coupe (même nom ou renommage ailleurs).
        """
        name = str(name).strip()
        data, key = self._load_with_key()
        items = data.get(key, [])
        if not isinstance(items, list):
            raise ValueError(f"Structure JSON invalide: {key} n'est pas une liste.")

        idx = self._find_index_by_name(items, name)
        if idx is None:
            raise KeyError(f"Coupe introuvable: {name}")

        # normalisation targets
        targets = []
        if isinstance(new_coupe.targets, list):
            for t in new_coupe.targets:
                ts = str(t).strip()
                if ts:
                    targets.append(ts)
        # dédoublonnage conservatif
        seen = set()
        targets2 = []
        for t in targets:
            tl = t.lower()
            if tl in seen:
                continue
            seen.add(tl)
            targets2.append(t)

        obj = items[idx]
        if not isinstance(obj, dict):
            obj = {}
            items[idx] = obj

        obj["name"] = str(new_coupe.name).strip() or name
        obj["angle_deg"] = float(new_coupe.angle_deg) if new_coupe.angle_deg is not None else None
        obj["targets"] = targets2
        obj["ui_idx"] = int(new_coupe.ui_idx) if isinstance(new_coupe.ui_idx, int) else obj.get("ui_idx", None)

        col = (new_coupe.color or "")
        col = str(col).strip() if isinstance(col, str) else ""
        obj["color"] = col if col else (obj.get("color", None))

        self._save(data)

    def rename_coupe(self, old_name: str, new_name: str) -> None:
        old_name = str(old_name).strip()
        new_name = str(new_name).strip()
        if not new_name:
            raise ValueError("Nouveau nom vide.")
        if old_name == new_name:
            return

        data, key = self._load_with_key()
        items = data.get(key, [])
        if not isinstance(items, list):
            raise ValueError(f"Structure JSON invalide: {key} n'est pas une liste.")

        # vérifie collisions
        for o in items:
            if isinstance(o, dict) and str(o.get("name", "")).strip() == new_name:
                raise ValueError(f"Une coupe s'appelle déjà '{new_name}'.")

        idx = self._find_index_by_name(items, old_name)
        if idx is None:
            raise KeyError(f"Coupe introuvable: {old_name}")

        obj = items[idx]
        if not isinstance(obj, dict):
            obj = {}
            items[idx] = obj
        obj["name"] = new_name

        self._save(data)

    def delete_coupe(self, name: str) -> None:
        name = str(name).strip()
        data, key = self._load_with_key()
        items = data.get(key, [])
        if not isinstance(items, list):
            raise ValueError(f"Structure JSON invalide: {key} n'est pas une liste.")

        idx = self._find_index_by_name(items, name)
        if idx is None:
            raise KeyError(f"Coupe introuvable: {name}")

        items.pop(idx)
        self._save(data)

    # --------------------------------------------------
    # Internals: parsing
    # --------------------------------------------------
    def _coupe_from_obj(self, obj: Any) -> Optional[Coupe]:
        if not isinstance(obj, dict):
            return None
        name = str(obj.get("name", "")).strip()
        if not name:
            return None

        angle = obj.get("angle_deg", None)
        try:
            angle_deg = float(angle) if angle is not None else None
        except Exception:
            angle_deg = None

        ui = obj.get("ui_idx", None)
        ui_idx: int | None
        try:
            ui_idx = int(ui) if ui is not None else None
        except Exception:
            ui_idx = None

        targets_raw = obj.get("targets", []) or []
        targets: List[str] = []
        if isinstance(targets_raw, list):
            for t in targets_raw:
                ts = str(t).strip()
                if ts:
                    targets.append(ts)

        color = obj.get("color", None)
        color = str(color).strip() if isinstance(color, str) and color.strip() else None

        return Coupe(name=name, angle_deg=angle_deg, targets=targets, ui_idx=ui_idx, color=color)

    def _find_index_by_name(self, items: list[Any], name: str) -> Optional[int]:
        for i, o in enumerate(items):
            if isinstance(o, dict) and str(o.get("name", "")).strip() == name:
                return i
        return None

    def _next_ui_idx(self, items: list[Any]) -> int:
        mx = -1
        for o in items:
            if not isinstance(o, dict):
                continue
            v = o.get("ui_idx", None)
            try:
                iv = int(v)
                if iv > mx:
                    mx = iv
            except Exception:
                continue
        return mx + 1 if mx >= 0 else 0

    # --------------------------------------------------
    # Internals: IO / paths
    # --------------------------------------------------
    def _load_with_key(self) -> tuple[Dict[str, Any], str]:
        """
        Retourne (data_dict, key) où key est 'coupes' ou 'zones' selon ce qui existe.
        On préfère 'coupes' si présent, sinon 'zones' si présent, sinon on crée 'coupes'.
        """
        data = self._read_json(self.pg_path)
        if not isinstance(data, dict):
            data = {}

        if "coupes" in data and isinstance(data.get("coupes"), list):
            return data, "coupes"
        if "zones" in data and isinstance(data.get("zones"), list):
            return data, "zones"

        # rien -> créer coupes
        data.setdefault("coupes", [])
        if not isinstance(data["coupes"], list):
            data["coupes"] = []
        return data, "coupes"

    def _read_json(self, path: Path) -> Any:
        # trouve variante existante si besoin
        if not path.exists():
            alt = self._find_existing_variant(path.parent)
            if alt is not None:
                path = alt
            else:
                return {}

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            try:
                return json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception:
                return {}

    def _save(self, data: Dict[str, Any]) -> None:
        self.pg_path.parent.mkdir(parents=True, exist_ok=True)

        # écriture atomique : tmp + replace
        tmp_dir = self.pg_path.parent
        fd, tmp_path = tempfile.mkstemp(prefix=".pg_tmp_", suffix=".json", dir=str(tmp_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(tmp_path, str(self.pg_path))
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

    def _project_root(self) -> Path:
        here = Path(__file__).resolve()
        for p in [here.parent, *here.parents]:
            if (p / "data" / "common_data").exists():
                return p
            if (p / "app.py").exists():
                return p
        return here.parents[2]

    def _default_json_path(self, root: Path) -> Path:
        common = root / "data" / "common_data"
        # Nom principal : sans accents (ton standard)
        p = common / "Parametres_Generaux.json"
        if p.exists():
            return p
        # sinon une variante existante si possible
        alt = self._find_existing_variant(common)
        return alt if alt is not None else p

    def _find_existing_variant(self, common_data_dir: Path) -> Optional[Path]:
        candidates = [
            common_data_dir / "Parametres_Generaux.json",
            common_data_dir / "Paramètres_Généraux.json",
            common_data_dir / "Paramètres Généraux.json",
            common_data_dir / "Parametres Generaux.json",
            common_data_dir / "Paramètres_Generaux.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None
