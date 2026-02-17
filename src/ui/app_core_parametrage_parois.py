# ======================================================
# src/ui/app_core_parametrage_parois.py  (COMPLET)
# ✅ JSON-only source of truth : Parois.json
# ✅ Tableau inversé :
#    1 ligne par PARAMÈTRE
#    Colonnes = COUPES
# ======================================================

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd
import streamlit as st


# ------------------------------------------------------
# JSON files
# ------------------------------------------------------
_PAR_FILENAME_PRIMARY = "Parois.json"
_PAR_FILENAME_FALLBACK = "parois.json"


def _find_parois_json(common_data_dir: Path) -> Path:
    p1 = common_data_dir / _PAR_FILENAME_PRIMARY
    if p1.exists():
        return p1
    return common_data_dir / _PAR_FILENAME_FALLBACK


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def _as_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s


# ------------------------------------------------------
# Data model
# ------------------------------------------------------
def _default_paroi_row(coupe_name: str) -> dict:
    return {
        "coupe": coupe_name,
        "cote_arase_sup": "",
        "cote_arase_inf": "",
        "cote_fond_fouille_def": "",
        "largeur_m_opt": "",
    }


def _normalize_parois_rows(rows_in: Any, coupe_names: list[str]) -> list[dict]:
    out_by_name: dict[str, dict] = {n: _default_paroi_row(n) for n in coupe_names if n}

    if isinstance(rows_in, list):
        for it in rows_in:
            if not isinstance(it, dict):
                continue
            name = str(it.get("coupe", "") or "").strip()
            if not name or name not in out_by_name:
                continue
            out_by_name[name] = {
                "coupe": name,
                "cote_arase_sup": _as_str(it.get("cote_arase_sup", "")),
                "cote_arase_inf": _as_str(it.get("cote_arase_inf", "")),
                "cote_fond_fouille_def": _as_str(it.get("cote_fond_fouille_def", "")),
                "largeur_m_opt": _as_str(it.get("largeur_m_opt", "")),
            }

    return [out_by_name[n] for n in coupe_names if n]


def _read_parois(common_data_dir: Path, coupe_names: list[str]) -> tuple[Path, dict]:
    path = _find_parois_json(common_data_dir)

    if not path.exists():
        payload = {"version": 1, "rows": _normalize_parois_rows([], coupe_names)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path, payload

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}

    payload.setdefault("version", 1)
    rows_in = payload.get("rows", [])
    payload["rows"] = _normalize_parois_rows(rows_in, coupe_names)

    return path, payload


def _write_parois(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ======================================================
# TRANSFORMATION TABLEAU (INVERSION)
# ======================================================
PARAM_LABELS = {
    "cote_arase_sup": "Cote Arase Sup.",
    "cote_arase_inf": "Cote Arase Inf.",
    "cote_fond_fouille_def": "Cote Fond de Fouille def.",
    "largeur_m_opt": "Largeur (m) opt.",
}


def _parois_rows_to_df(rows: list[dict], coupe_names: list[str]) -> pd.DataFrame:
    rows_norm = _normalize_parois_rows(rows, coupe_names)

    data = []
    for key, label in PARAM_LABELS.items():
        row = {"Paramètre": label}
        for r in rows_norm:
            row[r["coupe"]] = _as_str(r.get(key, ""))
        data.append(row)

    return pd.DataFrame(data)


def _df_to_parois_rows(df: pd.DataFrame, coupe_names: list[str]) -> list[dict]:
    rows_out: list[dict] = []

    if not isinstance(df, pd.DataFrame) or "Paramètre" not in df.columns:
        # fallback safe
        return _normalize_parois_rows([], coupe_names)

    for coupe in coupe_names:
        row_dict = _default_paroi_row(coupe)

        for key, label in PARAM_LABELS.items():
            match = df[df["Paramètre"] == label]
            if not match.empty:
                value = match.iloc[0].get(coupe, "")
                row_dict[key] = _as_str(value)

        rows_out.append(row_dict)

    return _normalize_parois_rows(rows_out, coupe_names)


# ======================================================
# Public entry
# ======================================================
def render_parametrage_parois(common_data_dir: str | Path, coupes) -> None:
    """
    coupes: liste d'objets avec attribut .name (utilisé comme clé stable)
    """
    common_data = Path(common_data_dir)

    coupe_names = [
        str(getattr(c, "name", "") or "").strip()
        for c in (coupes or [])
        if str(getattr(c, "name", "") or "").strip()
    ]

    if not coupe_names:
        st.warning("Aucune coupe trouvée.")
        return

    path, payload = _read_parois(common_data, coupe_names)

    # cache-busting editor
    rev_key = "parois_editor_rev"
    if rev_key not in st.session_state:
        st.session_state[rev_key] = 0
    editor_key = f"parois_editor_{st.session_state[rev_key]}"

    df = _parois_rows_to_df(payload.get("rows", []), coupe_names)

    edited = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_order=["Paramètre"] + coupe_names,
        column_config={
            "Paramètre": st.column_config.TextColumn("Paramètre", disabled=True),
        },
        key=editor_key,
    )

    if st.button("Enregistrer", use_container_width=True, key="parois_save"):
        try:
            rows_out = _df_to_parois_rows(edited, coupe_names)
            payload["rows"] = rows_out
            _write_parois(path, payload)

            st.session_state[rev_key] += 1
            st.success("Enregistré.")
            st.rerun()
        except Exception as e:
            st.error("Échec Enregistrer.")
            st.exception(e)
