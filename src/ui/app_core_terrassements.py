# ======================================================
# src/ui/app_core_terrassements.py  (COMPLET)
# ✅ JSON-only storage : nivter.json
# ✅ Table :
#    - 1ère colonne = Date (string DD-MM-YYYY, editable)
#    - Colonnes = COUPES (source unique: CoupesManager / Parametres_Generaux.json)
# ✅ 500 lignes fixes
# ✅ Un seul bouton : Enregistrer
# ✅ ZÉRO "None" affiché : editor reçoit uniquement des strings ("" si vide)
# ✅ render_terrassements() sans arguments (app.py inchangé)
# ======================================================

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd
import streamlit as st

from src.io.coupes_manager import CoupesManager


# ------------------------------------------------------
# Const
# ------------------------------------------------------
N_ROWS = 500
DATE_COL = "Date"
NIVTER_FILENAME = "nivter.json"


# ------------------------------------------------------
# Paths
# ------------------------------------------------------
def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _common_data_dir() -> Path:
    return _project_root() / "data" / "common_data"


def _nivter_path() -> Path:
    return _common_data_dir() / NIVTER_FILENAME


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def _as_str(v: Any) -> str:
    """Jamais None, jamais 'nan'/'none' textuel."""
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in {"nan", "none", "null", "vide", "<na>"}:
        return ""
    return s


def _is_valid_date_str(s: str) -> bool:
    """Validation stricte : DD-MM-YYYY (vide OK)."""
    s = (s or "").strip()
    if not s:
        return True
    try:
        pd.to_datetime(s, format="%d-%m-%Y", errors="raise")
        return True
    except Exception:
        return False


def _clean_date_str(s: str) -> str:
    """
    Normalise si possible en DD-MM-YYYY.
    - vide -> ""
    - si valide déjà -> renvoie normalisé
    - si invalide -> renvoie tel quel (on ne casse pas la saisie)
    """
    s = (s or "").strip()
    if not s:
        return ""
    try:
        dt = pd.to_datetime(s, format="%d-%m-%Y", errors="raise")
        return dt.strftime("%d-%m-%Y")
    except Exception:
        return s


def _clean_number_str(s: str) -> str:
    """
    Nettoyage léger : garde vide, sinon essaie de convertir en float.
    - accepte virgule française "12,3"
    - si non convertible -> retourne tel quel
    """
    s = (s or "").strip()
    if not s:
        return ""
    s2 = s.replace(",", ".")
    try:
        float(s2)
        return s2
    except Exception:
        return s


# ------------------------------------------------------
# Coupes (source unique)
# ------------------------------------------------------
def _load_coupe_names() -> list[str]:
    cm = CoupesManager(project_root=_project_root())
    coupes = cm.list_coupes()
    names = [
        str(getattr(c, "name", "") or "").strip()
        for c in (coupes or [])
        if str(getattr(c, "name", "") or "").strip()
    ]
    # unique, preserve order
    seen = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# ------------------------------------------------------
# JSON IO
# ------------------------------------------------------
def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------
# Data model / normalization
# ------------------------------------------------------
def _default_payload(coupe_names: list[str]) -> dict:
    return {"version": 1, "coupes": coupe_names, "rows": []}


def _normalize_rows(rows_in: Any, coupe_names: list[str]) -> list[dict]:
    """
    Retourne exactement N_ROWS lignes.
    Valeurs = strings, "" si vide.
    """
    base: list[dict] = []
    for _ in range(N_ROWS):
        r = {DATE_COL: ""}
        for c in coupe_names:
            r[c] = ""
        base.append(r)

    if not isinstance(rows_in, list):
        return base

    for i, it in enumerate(rows_in[:N_ROWS]):
        if not isinstance(it, dict):
            continue

        # Date (tolérance clé)
        d = _as_str(it.get(DATE_COL, "")) or _as_str(it.get("date", "")) or _as_str(it.get("Date", ""))
        base[i][DATE_COL] = d

        # Colonnes coupes
        for c in coupe_names:
            base[i][c] = _as_str(it.get(c, ""))

    return base


def _rows_to_df(rows: list[dict], coupe_names: list[str]) -> pd.DataFrame:
    cols = [DATE_COL] + coupe_names
    df = pd.DataFrame(rows, columns=cols)

    # Sécurité : tout en string, jamais None
    for col in cols:
        df[col] = df[col].apply(_as_str)

    # force 500
    if len(df) < N_ROWS:
        pad = pd.DataFrame([{k: "" for k in cols} for _ in range(N_ROWS - len(df))])
        df = pd.concat([df, pad], ignore_index=True)
    elif len(df) > N_ROWS:
        df = df.iloc[:N_ROWS].copy()

    return df


def _df_to_rows(df: pd.DataFrame, coupe_names: list[str]) -> list[dict]:
    cols = [DATE_COL] + coupe_names
    if not isinstance(df, pd.DataFrame):
        return _normalize_rows([], coupe_names)

    # assure colonnes
    for c in cols:
        if c not in df.columns:
            df[c] = ""

    df = df[cols].copy()

    out: list[dict] = []
    for _, r in df.iterrows():
        item: dict = {}

        d = _as_str(r.get(DATE_COL, ""))
        item[DATE_COL] = d

        for c in coupe_names:
            item[c] = _as_str(r.get(c, ""))

        out.append(item)

    return _normalize_rows(out, coupe_names)


# ======================================================
# Public entry (NO ARGS)
# ======================================================
def render_terrassements() -> None:
    st.subheader("Terrassements")

    coupe_names = _load_coupe_names()
    if not coupe_names:
        st.error("Aucune coupe trouvée (CoupesManager / Parametres_Generaux.json).")
        return

    path = _nivter_path()
    payload = _read_json(path)
    if not isinstance(payload, dict) or not payload:
        payload = _default_payload(coupe_names)

    payload.setdefault("version", 1)
    payload["coupes"] = coupe_names
    payload["rows"] = _normalize_rows(payload.get("rows", []), coupe_names)

    # cache-busting editor (comme ton module Parois)
    rev_key = "nivter_editor_rev"
    if rev_key not in st.session_state:
        st.session_state[rev_key] = 0
    editor_key = f"nivter_editor_{st.session_state[rev_key]}"

    df = _rows_to_df(payload["rows"], coupe_names)

    # Column config: header stays "Date", format explained in help
    date_col_cfg = st.column_config.TextColumn(
        label="Date",
        help="Format attendu : JJ-MM-AAAA (ex: 17-02-2026)",
    )

    edited = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_order=[DATE_COL] + coupe_names,
        column_config={
            DATE_COL: date_col_cfg,
            **{c: st.column_config.TextColumn(label=c) for c in coupe_names},
        },
        key=editor_key,
        height=720,
    )

    if st.button("Enregistrer", use_container_width=True, key="nivter_save"):
        try:
            rows_out = _df_to_rows(edited, coupe_names)

            # Nettoyage soft avant écriture
            bad_dates = 0
            for r in rows_out:
                r[DATE_COL] = _clean_date_str(_as_str(r.get(DATE_COL, "")))
                if not _is_valid_date_str(r[DATE_COL]):
                    bad_dates += 1
                for c in coupe_names:
                    r[c] = _clean_number_str(_as_str(r.get(c, "")))

            payload["rows"] = rows_out
            _write_json(path, payload)

            st.session_state[rev_key] += 1
            if bad_dates:
                st.warning(f"Enregistré. ⚠️ {bad_dates} date(s) ne respectent pas DD-MM-YYYY.")
            else:
                st.success("Enregistré.")
            st.rerun()

        except Exception as e:
            st.error("Échec Enregistrer.")
            st.exception(e)
