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

_TIR_FILENAME_PRIMARY = "Tirants.json"
_TIR_FILENAME_FALLBACK = "tirants.json"

_MAX_UPLOAD_BYTES = 3 * 1024 * 1024  # 3 Mo
_ALLOWED_MIME = {"application/pdf", "image/png", "image/jpeg"}

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
# Tirants.json
# ======================================================
def _find_tirants_json(common_data_dir: Path) -> Path:
    p1 = common_data_dir / _TIR_FILENAME_PRIMARY
    if p1.exists():
        return p1
    return common_data_dir / _TIR_FILENAME_FALLBACK


def _default_tir_rows() -> list[dict]:
    return [
        {
            "lit": f"Lit {i+1}",
            "cote": "",
            "l_libre_m": "",
            "l_ancree_m": "",
            "inclinaison_deg": "",
            "azimut_deg": "",
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


def _normalize_tir_rows(rows_in: Any) -> list[dict]:
    out = _default_tir_rows()

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
            "l_libre_m": _as_str(src.get("l_libre_m", "")),
            "l_ancree_m": _as_str(src.get("l_ancree_m", "")),
            "inclinaison_deg": _as_str(src.get("inclinaison_deg", "")),
            "azimut_deg": _as_str(src.get("azimut_deg", "")),
        }

    return out


def _read_tirants(common_data_dir: Path, coupe_names: list[str]) -> tuple[Path, dict]:
    path = _find_tirants_json(common_data_dir)

    if not path.exists():
        payload = {"version": 1, "coupes": {n: {"rows": _default_tir_rows()} for n in coupe_names if n}}
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
    if "coupes" not in payload or not isinstance(payload.get("coupes"), dict):
        payload["coupes"] = {}

    for n in coupe_names:
        if not n:
            continue
        if n not in payload["coupes"]:
            payload["coupes"][n] = {"rows": _default_tir_rows()}
        else:
            payload["coupes"][n]["rows"] = _normalize_tir_rows(payload["coupes"][n].get("rows"))

    return path, payload


def _write_tirants(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ======================================================
# IA : remplir tirants pour une coupe
# ======================================================
def _ai_complete_tirants_for_coupe(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    coupe_name: str,
    current_rows: list[dict],
    model: str = "gpt-4.1",
) -> list[dict]:
    api_key = st.secrets.get("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquant dans .streamlit/secrets.toml")

    client = OpenAI(api_key=api_key)

    # ✅ PROMPT renforcé (plan/tableau/texte + cohérence + relecture si aberrant)
    instruction = f"""
Tu reçois un document (PDF ou image) lié à un chantier et potentiellement à la coupe : {coupe_name}.
Ce document peut être un PLAN, une COUPE, un détail, une note, un tableau, ou un mélange.

Objectif:
- Extraire TOUT ce qui est possible et COHÉRENT sur les TIRANTS, et remplir Lit 1..Lit {_N_LITS}.
- Si un tableau existe : utilise-le en priorité.
- Si aucun tableau n’existe : lis le plan lui-même (annotations près des tirants, repères T1/T2…, flèches, textes, légendes, chaînes de cote).
- Si une info est indiquée "voir vue en plan" / "voir coupe" : essaie de la retrouver ailleurs dans le document.

Champs attendus (par lit):
- cote : altitude / niveau (ex: +132.50)
- l_libre_m : longueur libre (m)
- l_ancree_m : longueur ancrée (m)
- inclinaison_deg : inclinaison (°)
- azimut_deg : azimut (°)

Règles UNITÉS (important):
- Longueurs peuvent être en m, cm, mm.
  - Si tu vois 15000 mm => 15.0 m
  - Si tu vois 1500 cm => 15.0 m
  - Si tu vois 15 m => 15.0 m
- Ne confonds jamais avec des cotes planimétriques, dimensions de paroi, échelles graphiques, repères (ex: 90.00 peut être une cote/chaîne de cote, pas une longueur de tirant).

Règles COHÉRENCE (anti-erreurs d'ordre de grandeur):
- Un tirant de chantier a typiquement des longueurs de l’ordre de quelques mètres à quelques dizaines de mètres.
- Si tu extrais une longueur > 40 m (libre ou ancrée), considère que c’est PROBABLEMENT une mauvaise lecture.
  => Dans ce cas, REVIENS sur le document et cherche une valeur alternative plus plausible (souvent 12–25 m), ou laisse "" si tu n’as rien de fiable.
- Si l_libre_m + l_ancree_m est aberrant (ex: ~90 m) alors que le contexte montre des tirants courts, c’est un signal d’erreur: relecture obligatoire.

Stratégie de lecture:
1) Cherche un bloc "Tableau tirants", "tirants provisoires", "ancrages", ou similaire.
2) Sinon, repère les tirants sur le plan/coupe (symboles/traits obliques) et lis les annotations proches:
   - L libre, L ancrée, longueur totale, inclinaison, azimut, altitude
3) Si plusieurs valeurs possibles existent, choisis celle:
   - explicitement associée aux tirants (mots-clés: tirant, ancrage, L libre, L ancrée, inclinaison, azimut)
   - la plus cohérente (pas d’ordre de grandeur absurde)

Sortie:
- Retourne EXACTEMENT Lit 1..Lit {_N_LITS}.
- Si une valeur n’est pas trouvable ou pas fiable => "".
- Ne crée pas d'autres lits.
- Réponds UNIQUEMENT avec un JSON valide au format STRICT:

{{
  "rows": [
    {{"lit":"Lit 1","cote":"","l_libre_m":"","l_ancree_m":"","inclinaison_deg":"","azimut_deg":""}},
    ...
    {{"lit":"Lit {_N_LITS}","cote":"","l_libre_m":"","l_ancree_m":"","inclinaison_deg":"","azimut_deg":""}}
  ]
}}

Valeurs actuelles (à utiliser comme fallback si le document est partiel) :
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
    return _normalize_tir_rows(obj.get("rows"))


# ======================================================
# UI mapping
# ======================================================
def _tir_rows_to_df(rows: list[dict]) -> pd.DataFrame:
    rows_norm = _normalize_tir_rows(rows)
    out = []
    for i in range(_N_LITS):
        lit = f"Lit {i+1}"
        src = rows_norm[i] if i < len(rows_norm) else {}
        out.append(
            {
                "Lit": lit,
                "Cote": "" if src.get("cote") is None else str(src.get("cote", "")),
                "L libre (m)": "" if src.get("l_libre_m") is None else str(src.get("l_libre_m", "")),
                "L ancrée (m)": "" if src.get("l_ancree_m") is None else str(src.get("l_ancree_m", "")),
                "Inclinaison (°)": "" if src.get("inclinaison_deg") is None else str(src.get("inclinaison_deg", "")),
                "Azimut (°)": "" if src.get("azimut_deg") is None else str(src.get("azimut_deg", "")),
            }
        )
    return pd.DataFrame(out, columns=["Lit", "Cote", "L libre (m)", "L ancrée (m)", "Inclinaison (°)", "Azimut (°)"])


def _df_to_tir_rows(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for i in range(_N_LITS):
        lit = f"Lit {i+1}"
        if i < len(df):
            cote = "" if pd.isna(df.loc[i, "Cote"]) else str(df.loc[i, "Cote"]).strip()
            ll = "" if pd.isna(df.loc[i, "L libre (m)"]) else str(df.loc[i, "L libre (m)"]).strip()
            la = "" if pd.isna(df.loc[i, "L ancrée (m)"]) else str(df.loc[i, "L ancrée (m)"]).strip()
            inc = "" if pd.isna(df.loc[i, "Inclinaison (°)"]) else str(df.loc[i, "Inclinaison (°)"]).strip()
            azi = "" if pd.isna(df.loc[i, "Azimut (°)"]) else str(df.loc[i, "Azimut (°)"]).strip()
        else:
            cote = ll = la = inc = azi = ""

        rows.append(
            {
                "lit": lit,
                "cote": cote,
                "l_libre_m": ll,
                "l_ancree_m": la,
                "inclinaison_deg": inc,
                "azimut_deg": azi,
            }
        )
    return _normalize_tir_rows(rows)


# ======================================================
# Public entry
# ======================================================
def render_parametrage_tirants(common_data_dir: str | Path, coupes) -> None:
    common_data = Path(common_data_dir)

    # Sous-onglets coupes
    display_to_name: dict[str, str] = {}
    display_options: list[str] = []
    for i, c in enumerate(coupes or []):
        name = str(getattr(c, "name", "") or "").strip()
        if not name:
            continue
        col = str(getattr(c, "color", "") or "").strip()
        disp = f"{_emoji_from_color(col, i)} {name}"
        display_to_name[disp] = name
        display_options.append(disp)

    if not display_options:
        st.warning("Aucune coupe trouvée.")
        st.info("(Contenu vide)")
        return

    coupe_names = list(display_to_name.values())

    ss_key = "param_tirants_selected_coupe"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = display_to_name[display_options[0]]

    default_name = str(st.session_state[ss_key] or "")
    default_disp = next((d for d, n in display_to_name.items() if n == default_name), display_options[0])

    selected_disp = option_menu(
        menu_title=None,
        options=display_options,
        icons=[""] * len(display_options),
        orientation="horizontal",
        key="param_tirants_coupes_tabs",
        default_index=display_options.index(default_disp) if default_disp in display_options else 0,
        styles=_OPT_MENU_STYLES,
    )

    coupe_name = display_to_name.get(selected_disp, display_to_name[display_options[0]])
    st.session_state[ss_key] = coupe_name

    # JSON source of truth
    _, tir_payload = _read_tirants(common_data, coupe_names)

    # cache-busting data_editor
    rev_key = f"tir_editor_rev_{coupe_name}"
    if rev_key not in st.session_state:
        st.session_state[rev_key] = 0

    # Upload + IA au-dessus
    st.caption("PDF / PNG / JPEG — 3 Mo max")
    c1, c2 = st.columns([2, 1], vertical_alignment="center")

    with c1:
        uploaded = st.file_uploader(
            "Document",
            type=["pdf", "png", "jpg", "jpeg"],
            key=f"tir_uploader_{coupe_name}",
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

    # ✅ cache-busting fiable
    editor_key = f"tir_editor_{coupe_name}_{st.session_state[rev_key]}"

    with c2:
        if st.button(
            "Remplissage Automatique IA",
            use_container_width=True,
            disabled=(uploaded is None) or (not file_ok),
            key=f"tir_ai_btn_{coupe_name}",
        ):
            if uploaded is None or not file_ok:
                st.stop()

            file_bytes = uploaded.getvalue()
            if len(file_bytes) > _MAX_UPLOAD_BYTES:
                st.error("Fichier trop gros : limite 3 Mo.")
                st.stop()

            tir_path2, tir_payload2 = _read_tirants(common_data, coupe_names)
            current_rows = tir_payload2["coupes"][coupe_name]["rows"]

            try:
                with st.spinner("Analyse…"):
                    new_rows = _ai_complete_tirants_for_coupe(
                        file_bytes=file_bytes,
                        filename=uploaded.name or "document",
                        mime_type=mime_type or (uploaded.type or ""),
                        coupe_name=coupe_name,
                        current_rows=current_rows,
                        model="gpt-4.1",
                    )

                tir_payload2["coupes"][coupe_name]["rows"] = new_rows
                _write_tirants(tir_path2, tir_payload2)

                st.session_state[rev_key] += 1
                st.success("Fait.")
                st.rerun()
            except Exception as e:
                st.error("Échec IA.")
                st.exception(e)

    st.divider()

    # Tableur
    current_rows = tir_payload["coupes"][coupe_name]["rows"]
    df = _tir_rows_to_df(current_rows)

    edited = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Lit": st.column_config.TextColumn("Lit", disabled=True),
            "Cote": st.column_config.TextColumn("Cote"),
            "L libre (m)": st.column_config.TextColumn("L libre (m)"),
            "L ancrée (m)": st.column_config.TextColumn("L ancrée (m)"),
            "Inclinaison (°)": st.column_config.TextColumn("Inclinaison (°)"),
            "Azimut (°)": st.column_config.TextColumn("Azimut (°)"),
        },
        key=editor_key,
    )

    if st.button("Enregistrer", use_container_width=True, key=f"tir_save_{coupe_name}"):
        try:
            rows_out = _df_to_tir_rows(edited)

            tir_path3, tir_payload3 = _read_tirants(common_data, coupe_names)
            tir_payload3["coupes"][coupe_name]["rows"] = rows_out
            _write_tirants(tir_path3, tir_payload3)

            st.session_state[rev_key] += 1
            st.success("Enregistré.")
            st.rerun()
        except Exception as e:
            st.error("Échec Enregistrer.")
            st.exception(e)
