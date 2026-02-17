# ======================================================
# src/ui/app_core_butons.py  (COMPLET)
# - Même logique que tirants (JSON source of truth + onglets + data_editor + IA)
# - Onglets = "Buton 1", "Buton 2", etc (venant de `butons`)
# - Table: Lit | Cote | Longueur (m) | Inclinaison (°) opt. | Azimut (°) opt. | Distance milieu paroi (m) opt.
# ======================================================

from __future__ import annotations

from pathlib import Path
import base64
import json
import re
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from openai import OpenAI


# ------------------------------------------------------
# Styles option_menu horizontal
# ------------------------------------------------------
_OPT_MENU_STYLES = {
    "container": {"padding": "0!important"},
    "icon": {"display": "none"},
    "nav-link": {
        "font-size": "0.90rem",
        "padding": "8px 10px",
        "margin": "0 4px 0 0",
        "border-radius": "0.6rem",
        "white-space": "nowrap",
        "transition": "background-color 120ms ease",
    },
    "nav-link-hover": {"background-color": "rgba(0,0,0,0.06)"},
    "nav-link-selected": {
        "background-color": "rgba(0,0,0,0.10)",
        "font-weight": "700",
        "border-radius": "0.6rem",
    },
}

_ZONE_EMOJI = ["🔵", "🟢", "🟠", "🟣", "🔴", "🟦", "🟡", "🩷"]
_COLOR_TO_EMOJI = {
    "#146EFF": "🔵",
    "#22C55E": "🟢",
    "#F97316": "🟠",
    "#A855F7": "🟣",
    "#EF4444": "🔴",
    "#06B6D4": "🟦",
    "#EAB308": "🟡",
    "#EC4899": "🩷",
}


# ------------------------------------------------------
# Fichiers / contraintes upload
# ------------------------------------------------------
_BUT_FILENAME_PRIMARY = "Butons.json"
_BUT_FILENAME_FALLBACK = "butons.json"

_MAX_UPLOAD_BYTES = 3 * 1024 * 1024  # 3 Mo
_ALLOWED_MIME = {"application/pdf", "image/png", "image/jpeg"}


# ------------------------------------------------------
# Lits
# ------------------------------------------------------
_N_LITS = 15
_LIT_NUM_RE = re.compile(r"(\d+)")


def _zone_emoji(i: int) -> str:
    return _ZONE_EMOJI[i % len(_ZONE_EMOJI)]


def _emoji_from_color(col: str, fallback_i: int) -> str:
    col = (col or "").strip()
    return _COLOR_TO_EMOJI.get(col, _zone_emoji(fallback_i))


# ======================================================
# OpenAI response text (robuste)
# ======================================================
def _resp_text(resp: Any) -> str:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t

    out = getattr(resp, "output", None)
    if isinstance(out, list):
        parts: list[str] = []
        for item in out:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    txt = getattr(c, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt)
                    elif isinstance(c, dict) and isinstance(c.get("text"), str):
                        parts.append(c["text"])
        if parts:
            return "\n".join(parts)

    return ""


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj

    raise ValueError("Réponse IA: JSON invalide / introuvable.")


# ======================================================
# Butons.json
# ======================================================
def _find_butons_json(common_data_dir: Path) -> Path:
    p1 = common_data_dir / _BUT_FILENAME_PRIMARY
    if p1.exists():
        return p1
    return common_data_dir / _BUT_FILENAME_FALLBACK


def _default_buton_rows() -> list[dict]:
    return [
        {
            "lit": f"Lit {i+1}",
            "cote": "",
            "longueur_m": "",
            "inclinaison_deg_opt": "",
            "azimut_deg_opt": "",
            "dist_milieu_paroi_m_opt": "",
        }
        for i in range(_N_LITS)
    ]


def _canon_lit(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    m = _LIT_NUM_RE.search(s)
    if not m:
        return ""
    try:
        n = int(m.group(1))
    except Exception:
        return ""
    if 1 <= n <= _N_LITS:
        return f"Lit {n}"
    return ""


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s


def _normalize_buton_rows(rows_in: Any) -> list[dict]:
    out = _default_buton_rows()

    if not isinstance(rows_in, list) or not rows_in:
        return out

    by_lit: dict[str, dict] = {}
    for it in rows_in:
        if isinstance(it, dict):
            lit = _canon_lit(it.get("lit"))
            if lit:
                by_lit[lit] = it

    use_order = (len(by_lit) == 0)

    for i in range(_N_LITS):
        lit = f"Lit {i+1}"
        src = None

        if not use_order:
            src = by_lit.get(lit)
        else:
            if i < len(rows_in) and isinstance(rows_in[i], dict):
                src = rows_in[i]

        if not isinstance(src, dict):
            continue

        out[i] = {
            "lit": lit,
            "cote": _as_str(src.get("cote", "")),
            "longueur_m": _as_str(src.get("longueur_m", "")),
            "inclinaison_deg_opt": _as_str(src.get("inclinaison_deg_opt", "")),
            "azimut_deg_opt": _as_str(src.get("azimut_deg_opt", "")),
            "dist_milieu_paroi_m_opt": _as_str(src.get("dist_milieu_paroi_m_opt", "")),
        }

    return out


def _read_butons(common_data_dir: Path, buton_names: list[str]) -> tuple[Path, dict]:
    path = _find_butons_json(common_data_dir)

    if not path.exists():
        payload = {"version": 1, "butons": {n: {"rows": _default_buton_rows()} for n in buton_names if n}}
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
    if "butons" not in payload or not isinstance(payload.get("butons"), dict):
        payload["butons"] = {}

    for n in buton_names:
        if not n:
            continue
        if n not in payload["butons"]:
            payload["butons"][n] = {"rows": _default_buton_rows()}
        else:
            payload["butons"][n]["rows"] = _normalize_buton_rows(payload["butons"][n].get("rows"))

    return path, payload


def _write_butons(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ======================================================
# IA : remplir butons pour un buton
# ======================================================
def _ai_complete_butons_for_buton(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    buton_name: str,
    current_rows: list[dict],
    model: str = "gpt-4.1",
) -> list[dict]:
    api_key = st.secrets.get("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquant dans .streamlit/secrets.toml")

    client = OpenAI(api_key=api_key)

    instruction = f"""
Tu reçois un document (PDF ou image) lié à un chantier et potentiellement aux BUTONS : {buton_name}.
Ce document peut être un PLAN, une COUPE, un détail, une note, un tableau, ou un mélange.

Objectif:
- Extraire TOUT ce qui est possible et COHÉRENT sur les BUTONS, et remplir Lit 1..Lit {_N_LITS}.
- Si un tableau existe : utilise-le en priorité.
- Sinon : lis le plan/coupe (annotations, repères, flèches, légendes, chaînes de cote).

Champs attendus (par lit):
- cote : altitude / niveau (ex: +132.50)
- longueur_m : longueur (m)
- inclinaison_deg_opt : inclinaison (°) opt.
- azimut_deg_opt : azimut (°) opt.
- dist_milieu_paroi_m_opt : distance milieu paroi (m) opt.

Règles UNITÉS:
- Longueurs/distance peuvent être en m, cm, mm.
  - 15000 mm => 15.0 m
  - 1500 cm => 15.0 m
  - 15 m => 15.0 m

Cohérence:
- Si tu n'es pas sûr => "" (vide).
- Ne confonds pas avec des cotes planimétriques ou dimensions non liées.

Sortie:
- Retourne EXACTEMENT Lit 1..Lit {_N_LITS}.
- Réponds UNIQUEMENT avec un JSON valide au format STRICT:

{{
  "rows": [
    {{"lit":"Lit 1","cote":"","longueur_m":"","inclinaison_deg_opt":"","azimut_deg_opt":"","dist_milieu_paroi_m_opt":""}},
    ...
    {{"lit":"Lit {_N_LITS}","cote":"","longueur_m":"","inclinaison_deg_opt":"","azimut_deg_opt":"","dist_milieu_paroi_m_opt":""}}
  ]
}}

Valeurs actuelles (fallback si document partiel) :
{json.dumps(current_rows, ensure_ascii=False, indent=2)}
""".strip()

    if mime_type == "application/pdf":
        uploaded = client.files.create(
            file=(filename or "document.pdf", file_bytes, "application/pdf"),
            purpose="user_data",
        )
        content = [
            {"type": "input_file", "file_id": uploaded.id},
            {"type": "input_text", "text": instruction},
        ]
    else:
        b64 = base64.b64encode(file_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64}"
        content = [
            {"type": "input_image", "image_url": data_url},
            {"type": "input_text", "text": instruction},
        ]

    resp = client.responses.create(model=model, input=[{"role": "user", "content": content}])
    out_text = _resp_text(resp)
    obj = _extract_json_object(out_text)
    return _normalize_buton_rows(obj.get("rows"))


# ======================================================
# UI mapping
# ======================================================
def _buton_rows_to_df(rows: list[dict]) -> pd.DataFrame:
    rows_norm = _normalize_buton_rows(rows)
    out = []
    for i in range(_N_LITS):
        lit = f"Lit {i+1}"
        src = rows_norm[i] if i < len(rows_norm) else {}
        out.append(
            {
                "Lit": lit,
                "Cote": "" if src.get("cote") is None else str(src.get("cote", "")),
                "Longueur (m)": "" if src.get("longueur_m") is None else str(src.get("longueur_m", "")),
                "Inclinaison (°) opt.": "" if src.get("inclinaison_deg_opt") is None else str(src.get("inclinaison_deg_opt", "")),
                "Azimut (°) opt.": "" if src.get("azimut_deg_opt") is None else str(src.get("azimut_deg_opt", "")),
                "Distance milieu paroi (m) opt.": ""
                if src.get("dist_milieu_paroi_m_opt") is None
                else str(src.get("dist_milieu_paroi_m_opt", "")),
            }
        )

    df = pd.DataFrame(out)

    # Force existence + ordre (sinon Streamlit peut “perdre” une colonne)
    return df.reindex(
        columns=[
            "Lit",
            "Cote",
            "Longueur (m)",
            "Inclinaison (°) opt.",
            "Azimut (°) opt.",
            "Distance milieu paroi (m) opt.",
        ]
    )


def _df_to_buton_rows(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for i in range(_N_LITS):
        lit = f"Lit {i+1}"
        if i < len(df):
            cote = "" if pd.isna(df.loc[i, "Cote"]) else str(df.loc[i, "Cote"]).strip()
            lg = "" if pd.isna(df.loc[i, "Longueur (m)"]) else str(df.loc[i, "Longueur (m)"]).strip()
            inc = "" if pd.isna(df.loc[i, "Inclinaison (°) opt."]) else str(df.loc[i, "Inclinaison (°) opt."]).strip()
            azi = "" if pd.isna(df.loc[i, "Azimut (°) opt."]) else str(df.loc[i, "Azimut (°) opt."]).strip()
            dist = (
                ""
                if pd.isna(df.loc[i, "Distance milieu paroi (m) opt."])
                else str(df.loc[i, "Distance milieu paroi (m) opt."]).strip()
            )
        else:
            cote = lg = inc = azi = dist = ""

        rows.append(
            {
                "lit": lit,
                "cote": cote,
                "longueur_m": lg,
                "inclinaison_deg_opt": inc,
                "azimut_deg_opt": azi,
                "dist_milieu_paroi_m_opt": dist,
            }
        )
    return _normalize_buton_rows(rows)


# ======================================================
# Public entry
# ======================================================
def render_parametrage_butons(common_data_dir: str | Path, butons) -> None:
    """
    butons: liste d'objets avec attributs .name et éventuellement .color
    Ex: buton.name = "Buton 1", "Buton 2", etc.
    """
    common_data = Path(common_data_dir)

    # Sous-onglets butons
    display_to_name: dict[str, str] = {}
    display_options: list[str] = []
    for i, b in enumerate(butons or []):
        name = str(getattr(b, "name", "") or "").strip()
        if not name:
            continue
        col = str(getattr(b, "color", "") or "").strip()
        disp = f"{_emoji_from_color(col, i)} {name}"
        display_to_name[disp] = name
        display_options.append(disp)

    if not display_options:
        st.warning("Aucun buton trouvé.")
        st.info("(Contenu vide)")
        return

    buton_names = list(display_to_name.values())

    ss_key = "param_butons_selected_buton"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = display_to_name[display_options[0]]

    default_name = str(st.session_state[ss_key] or "")
    default_disp = next((d for d, n in display_to_name.items() if n == default_name), display_options[0])

    selected_disp = option_menu(
        menu_title=None,
        options=display_options,
        icons=[""] * len(display_options),
        orientation="horizontal",
        key="param_butons_tabs",
        default_index=display_options.index(default_disp) if default_disp in display_options else 0,
        styles=_OPT_MENU_STYLES,
    )

    buton_name = display_to_name.get(selected_disp, display_to_name[display_options[0]])
    st.session_state[ss_key] = buton_name

    # JSON source of truth
    _, but_payload = _read_butons(common_data, buton_names)

    # cache-busting data_editor
    rev_key = f"but_editor_rev_{buton_name}"
    if rev_key not in st.session_state:
        st.session_state[rev_key] = 0

    # Upload + IA au-dessus
    st.caption("PDF / PNG / JPEG — 3 Mo max")
    c1, c2 = st.columns([2, 1], vertical_alignment="center")

    with c1:
        uploaded = st.file_uploader(
            "Document",
            type=["pdf", "png", "jpg", "jpeg"],
            key=f"but_uploader_{buton_name}",
            label_visibility="collapsed",
        )

    file_ok = True
    mime_type = ""
    if uploaded is not None:
        mime_type = (getattr(uploaded, "type", None) or "").strip().lower()
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"

        size = getattr(uploaded, "size", None)
        if size is None:
            try:
                size = len(uploaded.getvalue())
            except Exception:
                size = None

        if size is not None and size > _MAX_UPLOAD_BYTES:
            file_ok = False
            st.error("Fichier trop gros : limite 3 Mo.")

        if mime_type and (mime_type not in _ALLOWED_MIME):
            file_ok = False
            st.error("Type non supporté. Autorisés: PDF, PNG, JPEG.")

    editor_key = f"but_editor_{buton_name}_{st.session_state[rev_key]}"

    with c2:
        if st.button(
            "Remplissage Automatique IA",
            use_container_width=True,
            disabled=(uploaded is None) or (not file_ok),
            key=f"but_ai_btn_{buton_name}",
        ):
            if uploaded is None or not file_ok:
                st.stop()

            file_bytes = uploaded.getvalue()
            if len(file_bytes) > _MAX_UPLOAD_BYTES:
                st.error("Fichier trop gros : limite 3 Mo.")
                st.stop()

            but_path2, but_payload2 = _read_butons(common_data, buton_names)
            current_rows = but_payload2["butons"][buton_name]["rows"]

            try:
                with st.spinner("Analyse…"):
                    new_rows = _ai_complete_butons_for_buton(
                        file_bytes=file_bytes,
                        filename=uploaded.name or "document",
                        mime_type=mime_type or (uploaded.type or ""),
                        buton_name=buton_name,
                        current_rows=current_rows,
                        model="gpt-4.1",
                    )

                but_payload2["butons"][buton_name]["rows"] = new_rows
                _write_butons(but_path2, but_payload2)

                st.session_state[rev_key] += 1
                st.success("Fait.")
                st.rerun()
            except Exception as e:
                st.error("Échec IA.")
                st.exception(e)

    st.divider()

    # Tableur
    current_rows = but_payload["butons"][buton_name]["rows"]
    df = _buton_rows_to_df(current_rows)

    edited = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_order=[
            "Lit",
            "Cote",
            "Longueur (m)",
            "Inclinaison (°) opt.",
            "Azimut (°) opt.",
            "Distance milieu paroi (m) opt.",
        ],
        column_config={
            "Lit": st.column_config.TextColumn("Lit", disabled=True),
            "Cote": st.column_config.TextColumn("Cote"),
            "Longueur (m)": st.column_config.TextColumn("Longueur (m)"),
            "Inclinaison (°) opt.": st.column_config.TextColumn("Inclinaison (°) opt."),
            "Azimut (°) opt.": st.column_config.TextColumn("Azimut (°) opt."),
            "Distance milieu paroi (m) opt.": st.column_config.TextColumn("Distance milieu paroi (m) opt."),
        },
        key=editor_key,
    )

    if st.button("Enregistrer", use_container_width=True, key=f"but_save_{buton_name}"):
        try:
            rows_out = _df_to_buton_rows(edited)

            but_path3, but_payload3 = _read_butons(common_data, buton_names)
            but_payload3["butons"][buton_name]["rows"] = rows_out
            _write_butons(but_path3, but_payload3)

            st.session_state[rev_key] += 1
            st.success("Enregistré.")
            st.rerun()
        except Exception as e:
            st.error("Échec Enregistrer.")
            st.exception(e)
