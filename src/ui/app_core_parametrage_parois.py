# ======================================================
# src/ui/app_core_parametrage_parois.py  (COMPLET)
# ✅ JSON-only source of truth : Parois.json
# ✅ Tableau unique (pas d'onglets) :
#    1 ligne par COUPE, 1ère colonne = Nom coupe
#    Colonnes :
#      - Cote Arase Sup.
#      - Cote Arase Inf.
#      - Cote Fond de Fouille def.
#      - Largeur (m) opt.
# ✅ Même UX que tes autres pages :
#    - data_editor + Enregistrer
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
    # On veut 1 ligne par coupe (dans l'ordre coupe_names)
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


# ------------------------------------------------------
# UI mapping
# ------------------------------------------------------
def _parois_rows_to_df(rows: list[dict], coupe_names: list[str]) -> pd.DataFrame:
    rows_norm = _normalize_parois_rows(rows, coupe_names)
    out = []
    for r in rows_norm:
        out.append(
            {
                "Coupe": _as_str(r.get("coupe", "")),
                "Cote Arase Sup.": _as_str(r.get("cote_arase_sup", "")),
                "Cote Arase Inf.": _as_str(r.get("cote_arase_inf", "")),
                "Cote Fond de Fouille def.": _as_str(r.get("cote_fond_fouille_def", "")),
                "Largeur (m) opt.": _as_str(r.get("largeur_m_opt", "")),
            }
        )

    df = pd.DataFrame(out)
    return df.reindex(
        columns=[
            "Coupe",
            "Cote Arase Sup.",
            "Cote Arase Inf.",
            "Cote Fond de Fouille def.",
            "Largeur (m) opt.",
        ]
    )


def _df_to_parois_rows(df: pd.DataFrame, coupe_names: list[str]) -> list[dict]:
    rows: list[dict] = []

    # Index par coupe pour éviter que l'ordre/filtrage casse tout
    df_by_name: dict[str, int] = {}
    if isinstance(df, pd.DataFrame) and "Coupe" in df.columns:
        for i in range(len(df)):
            name = "" if pd.isna(df.loc[i, "Coupe"]) else str(df.loc[i, "Coupe"]).strip()
            if name:
                df_by_name[name] = i

    for name in (coupe_names or []):
        if not name:
            continue
        i = df_by_name.get(name, None)
        if i is None:
            rows.append(_default_paroi_row(name))
            continue

        cas = "" if pd.isna(df.loc[i, "Cote Arase Sup."]) else str(df.loc[i, "Cote Arase Sup."]).strip()
        cai = "" if pd.isna(df.loc[i, "Cote Arase Inf."]) else str(df.loc[i, "Cote Arase Inf."]).strip()
        cff = "" if pd.isna(df.loc[i, "Cote Fond de Fouille def."]) else str(df.loc[i, "Cote Fond de Fouille def."]).strip()
        larg = "" if pd.isna(df.loc[i, "Largeur (m) opt."]) else str(df.loc[i, "Largeur (m) opt."]).strip()

        rows.append(
            {
                "coupe": name,
                "cote_arase_sup": cas,
                "cote_arase_inf": cai,
                "cote_fond_fouille_def": cff,
                "largeur_m_opt": larg,
            }
        )

    return _normalize_parois_rows(rows, coupe_names)


# ======================================================
# Public entry
# ======================================================
def render_parametrage_parois(common_data_dir: str | Path, coupes) -> None:
    """
    coupes: liste d'objets avec attribut .name (utilisé comme clé stable)
    """
    common_data = Path(common_data_dir)

    coupe_names: list[str] = []
    for c in (coupes or []):
        name = str(getattr(c, "name", "") or "").strip()
        if name:
            coupe_names.append(name)

    if not coupe_names:
        st.warning("Aucune coupe trouvée.")
        st.info("(Contenu vide)")
        return

    # JSON source of truth
    _, payload = _read_parois(common_data, coupe_names)

    # cache-busting editor
    rev_key = "parois_editor_rev"
    if rev_key not in st.session_state:
        st.session_state[rev_key] = 0
    editor_key = f"parois_editor_{st.session_state[rev_key]}"

    # Table
    df = _parois_rows_to_df(payload.get("rows", []), coupe_names)

    edited = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_order=[
            "Coupe",
            "Cote Arase Sup.",
            "Cote Arase Inf.",
            "Cote Fond de Fouille def.",
            "Largeur (m) opt.",
        ],
        column_config={
            "Coupe": st.column_config.TextColumn("Coupe", disabled=True),
            "Cote Arase Sup.": st.column_config.TextColumn("Cote Arase Sup."),
            "Cote Arase Inf.": st.column_config.TextColumn("Cote Arase Inf."),
            "Cote Fond de Fouille def.": st.column_config.TextColumn("Cote Fond de Fouille def."),
            "Largeur (m) opt.": st.column_config.TextColumn("Largeur (m) opt."),
        },
        key=editor_key,
    )

    if st.button("Enregistrer", use_container_width=True, key="parois_save"):
        try:
            rows_out = _df_to_parois_rows(edited, coupe_names)
            path2, payload2 = _read_parois(common_data, coupe_names)
            payload2["rows"] = rows_out
            _write_parois(path2, payload2)

            st.session_state[rev_key] += 1
            st.success("Enregistré.")
            st.rerun()
        except Exception as e:
            st.error("Échec Enregistrer.")
            st.exception(e)
