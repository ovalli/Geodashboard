from __future__ import annotations

from pathlib import Path
import base64
import json

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from openai import OpenAI


# UI menu styles
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

_LITHO_FILENAME_PRIMARY = "Lithologie.json"
_LITHO_FILENAME_FALLBACK = "lithologie.json"

_GEO_FILENAME_PRIMARY = "Geologie.json"
_GEO_FILENAME_FALLBACK = "geologie.json"

_MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 3 Mo
_ALLOWED_MIME = {"application/pdf", "image/png", "image/jpeg"}


def _zone_emoji(i: int) -> str:
    return _ZONE_EMOJI[i % len(_ZONE_EMOJI)]


def _emoji_from_color(col: str, fallback_i: int) -> str:
    col = (col or "").strip()
    return _COLOR_TO_EMOJI.get(col, _zone_emoji(fallback_i))


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


# ---------------- Lithologie: lire les 10 noms
def _find_lithologie_json(common_data_dir: Path) -> Path:
    p1 = common_data_dir / _LITHO_FILENAME_PRIMARY
    if p1.exists():
        return p1
    return common_data_dir / _LITHO_FILENAME_FALLBACK


def _read_lithologie_layer_names(common_data_dir: Path) -> list[str]:
    """
    Retourne 10 noms (colonne 'nom' de Lithologie.json). Fallback => 'SOL i'
    """
    path = _find_lithologie_json(common_data_dir)
    if not path.exists():
        return [f"SOL {i+1}" for i in range(10)]

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return [f"SOL {i+1}" for i in range(10)]
    except Exception:
        return [f"SOL {i+1}" for i in range(10)]

    rows = payload.get("rows")
    if not isinstance(rows, list):
        rows = []

    out: list[str] = []
    for i in range(10):
        src = rows[i] if i < len(rows) and isinstance(rows[i], dict) else {}
        nom = "" if src.get("nom") is None else str(src.get("nom", "")).strip()
        out.append(nom if nom else f"SOL {i+1}")
    return out


# ---------------- Geologie JSON
def _find_geologie_json(common_data_dir: Path) -> Path:
    p1 = common_data_dir / _GEO_FILENAME_PRIMARY
    if p1.exists():
        return p1
    return common_data_dir / _GEO_FILENAME_FALLBACK


def _default_geo_rows() -> list[dict]:
    return [
        {"sol": f"SOL {i+1}", "arase_sup": "", "arase_inf": "", "pendage_sup": ""}
        for i in range(10)
    ]


def _normalize_geo_rows(rows) -> list[dict]:
    by_sol: dict[str, dict] = {}
    if isinstance(rows, list):
        for it in rows:
            if isinstance(it, dict):
                sol = str(it.get("sol", "")).strip()
                if sol:
                    by_sol[sol] = it

    out: list[dict] = []
    for i in range(10):
        sol = f"SOL {i+1}"
        src = by_sol.get(sol, {}) or {}
        out.append(
            {
                "sol": sol,
                "arase_sup": "" if src.get("arase_sup") is None else str(src.get("arase_sup", "")).strip(),
                "arase_inf": "" if src.get("arase_inf") is None else str(src.get("arase_inf", "")).strip(),
                "pendage_sup": "" if src.get("pendage_sup") is None else str(src.get("pendage_sup", "")).strip(),
            }
        )
    return out


def _read_geologie(common_data_dir: Path, coupe_names: list[str]) -> tuple[Path, dict]:
    path = _find_geologie_json(common_data_dir)

    if not path.exists():
        payload = {"version": 1, "coupes": {n: {"rows": _default_geo_rows()} for n in coupe_names if n}}
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
            payload["coupes"][n] = {"rows": _default_geo_rows()}
        else:
            payload["coupes"][n]["rows"] = _normalize_geo_rows(payload["coupes"][n].get("rows"))

    return path, payload


def _write_geologie(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------- IA
def _ai_complete_geologie_for_coupe(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    coupe_name: str,
    litho_names: list[str],
    current_rows: list[dict],
    model: str = "gpt-4o-mini",
) -> list[dict]:
    api_key = st.secrets.get("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquant dans .streamlit/secrets.toml")

    client = OpenAI(api_key=api_key)

    layers = [{"sol": f"SOL {i+1}", "nom": litho_names[i] if i < len(litho_names) else f"SOL {i+1}"} for i in range(10)]

    instruction = f"""
Tu reçois:
- un document (PDF ou image)
- la coupe: {coupe_name}
- les couches SOL 1..SOL 10 (avec leur nom)
- les valeurs actuelles (peuvent être vides)

Objectif:
- Remplir pour SOL 1..SOL 10 :
  - arase_sup
  - arase_inf
  - pendage_sup (optionnel, peut rester vide)
- Si tu ne sais pas, laisse vide.
- Réponds UNIQUEMENT avec un JSON valide au format:

{{
  "rows": [
    {{"sol":"SOL 1","arase_sup":"","arase_inf":"","pendage_sup":""}},
    ...
    {{"sol":"SOL 10","arase_sup":"","arase_inf":"","pendage_sup":""}}
  ]
}}

Couches (noms depuis Lithologie):
{json.dumps(layers, ensure_ascii=False, indent=2)}

Valeurs actuelles:
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
    out_text = getattr(resp, "output_text", None) or ""
    obj = _extract_json_object(out_text)
    return _normalize_geo_rows(obj.get("rows"))


# ---------------- UI mapping
def _geo_rows_to_df(litho_names: list[str], geo_rows: list[dict]) -> pd.DataFrame:
    by_sol = {r.get("sol"): r for r in (geo_rows or []) if isinstance(r, dict)}
    out = []
    for i in range(10):
        sol = f"SOL {i+1}"
        name = litho_names[i] if i < len(litho_names) else sol
        src = by_sol.get(sol, {}) or {}
        out.append(
            {
                "Couche": name,
                "Arase sup.": "" if src.get("arase_sup") is None else str(src.get("arase_sup", "")),
                "Arase inf.": "" if src.get("arase_inf") is None else str(src.get("arase_inf", "")),
                "Pendage sup (opt.)": "" if src.get("pendage_sup") is None else str(src.get("pendage_sup", "")),
            }
        )
    return pd.DataFrame(out, columns=["Couche", "Arase sup.", "Arase inf.", "Pendage sup (opt.)"])


def _df_to_geo_rows(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for i in range(10):
        sol = f"SOL {i+1}"
        if i < len(df):
            ar_sup = "" if pd.isna(df.loc[i, "Arase sup."]) else str(df.loc[i, "Arase sup."]).strip()
            ar_inf = "" if pd.isna(df.loc[i, "Arase inf."]) else str(df.loc[i, "Arase inf."]).strip()
            pen = "" if pd.isna(df.loc[i, "Pendage sup (opt.)"]) else str(df.loc[i, "Pendage sup (opt.)"]).strip()
        else:
            ar_sup = ar_inf = pen = ""
        rows.append({"sol": sol, "arase_sup": ar_sup, "arase_inf": ar_inf, "pendage_sup": pen})
    return _normalize_geo_rows(rows)


# ======================================================
# Public entry
# ======================================================
def render_parametrage_geologie(common_data_dir: str | Path, coupes) -> None:
    common_data = Path(common_data_dir)

    # Tabs coupes
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

    ss_key = "param_geologie_selected_coupe"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = display_to_name[display_options[0]]

    default_name = str(st.session_state[ss_key] or "")
    default_disp = next((d for d, n in display_to_name.items() if n == default_name), display_options[0])

    selected_disp = option_menu(
        menu_title=None,
        options=display_options,
        icons=[""] * len(display_options),
        orientation="horizontal",
        key="param_geologie_coupes_tabs",
        default_index=display_options.index(default_disp) if default_disp in display_options else 0,
        styles=_OPT_MENU_STYLES,
    )

    coupe_name = display_to_name.get(selected_disp, display_to_name[display_options[0]])
    st.session_state[ss_key] = coupe_name

    litho_names = _read_lithologie_layer_names(common_data)

    # Load JSON source of truth
    geo_path, geo_payload = _read_geologie(common_data, coupe_names)

    # ---- Upload + IA AU DESSUS
    st.caption("PDF / PNG / JPEG — 3 Mo max")
    c1, c2 = st.columns([2, 1], vertical_alignment="center")

    with c1:
        uploaded = st.file_uploader(
            "Document",
            type=["pdf", "png", "jpg", "jpeg"],
            key=f"geo_uploader_{coupe_name}",
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

        if mime_type not in _ALLOWED_MIME:
            file_ok = False
            st.error("Type non supporté. Autorisés: PDF, PNG, JPEG.")

    # ✅ editor revision (pour casser le cache du data_editor après IA / save)
    rev_key = f"geo_editor_rev_{coupe_name}"
    if rev_key not in st.session_state:
        st.session_state[rev_key] = 0

    with c2:
        if st.button(
            "Remplissage Automatique IA",
            use_container_width=True,
            disabled=(uploaded is None) or (not file_ok),
            key=f"geo_ai_btn_{coupe_name}",
        ):
            if uploaded is None or not file_ok:
                st.stop()

            file_bytes = uploaded.getvalue()
            if len(file_bytes) > _MAX_UPLOAD_BYTES:
                st.error("Fichier trop gros : limite 3 Mo.")
                st.stop()

            # reload JSON source of truth (important)
            geo_path2, geo_payload2 = _read_geologie(common_data, coupe_names)
            current_rows = geo_payload2["coupes"][coupe_name]["rows"]

            with st.spinner("Analyse…"):
                new_rows = _ai_complete_geologie_for_coupe(
                    file_bytes=file_bytes,
                    filename=uploaded.name or "document",
                    mime_type=mime_type or (uploaded.type or ""),
                    coupe_name=coupe_name,
                    litho_names=litho_names,
                    current_rows=current_rows,
                    model="gpt-4o-mini",
                )

            geo_payload2["coupes"][coupe_name]["rows"] = new_rows
            _write_geologie(geo_path2, geo_payload2)

            # ✅ casse le cache du data_editor (sinon tu revois l’ancien contenu)
            st.session_state[rev_key] += 1
            st.success("Fait.")
            st.rerun()

    st.divider()

    # ---- TABLEUR (lignes=couches, colonnes=3 champs)
    current_rows = geo_payload["coupes"][coupe_name]["rows"]
    df = _geo_rows_to_df(litho_names, current_rows)

    edited = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Couche": st.column_config.TextColumn("Couche", disabled=True),
            "Arase sup.": st.column_config.TextColumn("Arase sup."),
            "Arase inf.": st.column_config.TextColumn("Arase inf."),
            "Pendage sup (opt.)": st.column_config.TextColumn("Pendage sup (opt.)"),
        },
        key=f"geo_editor_{coupe_name}_{st.session_state[rev_key]}",
    )

    if st.button("Enregistrer", use_container_width=True, key=f"geo_save_{coupe_name}"):
        try:
            rows_out = _df_to_geo_rows(edited)

            geo_path3, geo_payload3 = _read_geologie(common_data, coupe_names)
            geo_payload3["coupes"][coupe_name]["rows"] = rows_out
            _write_geologie(geo_path3, geo_payload3)

            # ✅ casse le cache du data_editor
            st.session_state[rev_key] += 1
            st.success("Enregistré.")
            st.rerun()
        except Exception as e:
            st.error("Échec Enregistrer.")
            st.exception(e)
