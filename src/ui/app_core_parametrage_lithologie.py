from __future__ import annotations

from pathlib import Path
import base64
import json
import re

import streamlit as st
from openai import OpenAI

from src.ui.lithologie_component import lithologie_component


_LITHO_FILENAME_PRIMARY = "Lithologie.json"
_LITHO_FILENAME_FALLBACK = "lithologie.json"
_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")

# ✅ Limite upload à 3 Mo (PDF/PNG/JPEG)
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 Mo


# ======================================================
# Lithologie JSON
# ======================================================
def _find_lithologie_json(common_data_dir: Path) -> Path:
    p1 = common_data_dir / _LITHO_FILENAME_PRIMARY
    if p1.exists():
        return p1
    return common_data_dir / _LITHO_FILENAME_FALLBACK


def _default_lithologie_payload() -> dict:
    palette = [
        "#146EFF", "#1D4ED8", "#06B6D4", "#0891B2",
        "#22C55E", "#16A34A", "#84CC16", "#EAB308",
        "#F97316", "#EA580C", "#EF4444", "#DC2626",
        "#EC4899", "#DB2777", "#A855F7", "#7C3AED",
        "#64748B", "#334155", "#8B5E34", "#111827",
    ]
    base = ["#146EFF", "#22C55E", "#F97316", "#A855F7", "#EF4444", "#06B6D4", "#EAB308", "#EC4899", "#64748B", "#334155"]
    rows = [{"sol": f"SOL {i+1}", "nom": "", "couleur": base[i]} for i in range(10)]
    return {"version": 4, "palette": palette, "rows": rows}


def _read_lithologie(common_data_dir: Path) -> tuple[Path, dict]:
    path = _find_lithologie_json(common_data_dir)

    if not path.exists():
        payload = _default_lithologie_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path, payload

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Lithologie.json invalide")
    except Exception:
        payload = _default_lithologie_payload()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path, payload

    palette = payload.get("palette")
    if not isinstance(palette, list) or not palette:
        palette = _default_lithologie_payload()["palette"]

    palette = [
        str(x).strip().upper()
        for x in palette
        if isinstance(x, str) and _HEX_RE.match(str(x).strip())
    ]
    if not palette:
        palette = _default_lithologie_payload()["palette"]

    rows_raw = payload.get("rows")
    if not isinstance(rows_raw, list):
        rows_raw = []

    rows: list[dict] = []
    for i in range(10):
        sol = f"SOL {i+1}"
        src = rows_raw[i] if i < len(rows_raw) and isinstance(rows_raw[i], dict) else {}
        nom = src.get("nom", "")
        col = src.get("couleur", palette[i % len(palette)])
        nom = "" if nom is None else str(nom)
        col = str(col).strip().upper() if col is not None else palette[0]
        if not _HEX_RE.match(col):
            col = palette[0]
        rows.append({"sol": sol, "nom": nom, "couleur": col})

    return path, {"version": 4, "palette": palette, "rows": rows}


def _write_lithologie(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
        candidate = text[start : end + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Réponse IA: JSON invalide / introuvable.")


def _normalize_lithologie_payload(obj: dict, fallback_palette: list[str]) -> dict:
    palette = obj.get("palette", fallback_palette)
    if not isinstance(palette, list) or not palette:
        palette = fallback_palette

    palette = [
        str(x).strip().upper()
        for x in palette
        if isinstance(x, str) and _HEX_RE.match(str(x).strip())
    ]
    if not palette:
        palette = fallback_palette

    rows_in = obj.get("rows")
    if not isinstance(rows_in, list):
        raise ValueError("JSON IA invalide: 'rows' manquant ou mauvais format.")

    rows: list[dict] = []
    for i in range(10):
        sol = f"SOL {i+1}"
        src = rows_in[i] if i < len(rows_in) and isinstance(rows_in[i], dict) else {}

        # ✅ cases peuvent rester vides
        nom = "" if src.get("nom") is None else str(src.get("nom", "")).strip()

        col = src.get("couleur")
        if col is None or str(col).strip() == "":
            col = fallback_palette[i % len(fallback_palette)] if fallback_palette else palette[0]
        col = str(col).strip().upper()
        if not _HEX_RE.match(col):
            col = palette[0]

        rows.append({"sol": sol, "nom": nom, "couleur": col})

    return {"version": 4, "palette": palette, "rows": rows}


def _ai_complete_lithologie_from_document(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    current_payload: dict,
    model: str = "gpt-4o-mini",
) -> dict:
    api_key = st.secrets.get("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquant dans .streamlit/secrets.toml")

    client = OpenAI(api_key=api_key)

    instruction = f"""
Tu reçois:
- un document (PDF OU image)
- un JSON Lithologie.json actuel (ci-dessous)

Objectif:
- Compléter le champ "nom" pour SOL 1..SOL 10 à partir du document.
- Certaines cases peuvent rester vides si l'information n'existe pas ou n'est pas fiable.
- Ne crée pas de SOL supplémentaires: retourne exactement SOL 1..SOL 10.
- Conserve "couleur" au format #RRGGBB. Si tu ne sais pas, laisse la couleur existante.
- Réponds UNIQUEMENT avec un JSON valide (aucun texte autour).

JSON actuel:
{json.dumps(current_payload, ensure_ascii=False, indent=2)}
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

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
    )

    out_text = getattr(resp, "output_text", None) or ""
    obj = _extract_json_object(out_text)

    fallback_palette = current_payload.get("palette", _default_lithologie_payload()["palette"])
    if not isinstance(fallback_palette, list) or not fallback_palette:
        fallback_palette = _default_lithologie_payload()["palette"]

    return _normalize_lithologie_payload(obj, fallback_palette=fallback_palette)


# ======================================================
# Public entry (tab Lithologie)
# ======================================================
def render_parametrage_lithologie(common_data_dir: str | Path) -> None:
    common_data = Path(common_data_dir)

    path, payload = _read_lithologie(common_data)

    if "litho_rev" not in st.session_state:
        st.session_state["litho_rev"] = 0

    # ==================================================
    # ✅ AU DESSUS DU TABLEUR :
    #    - drag&drop (uploader)
    #    - bouton "Remplissage Automatique IA"
    #    - texte "3 Mo max"
    # ==================================================
    st.caption("PDF / PNG / JPEG — 3 Mo max")

    top_left, top_right = st.columns([2, 1], vertical_alignment="center")

    with top_left:
        uploaded_file = st.file_uploader(
            "Document",
            type=["pdf", "png", "jpg", "jpeg"],
            key="litho_doc_uploader",
            label_visibility="collapsed",
        )

    file_ok = True
    mime_type = ""

    if uploaded_file is not None:
        size = getattr(uploaded_file, "size", None)
        if size is None:
            try:
                size = len(uploaded_file.getvalue())
            except Exception:
                size = None

        if size is not None and size > _MAX_UPLOAD_BYTES:
            file_ok = False
            st.error("Fichier trop gros : limite 3 Mo.")

        mime_type = (getattr(uploaded_file, "type", None) or "").strip().lower()
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"

        allowed = {"application/pdf", "image/png", "image/jpeg"}
        if mime_type not in allowed:
            file_ok = False
            st.error("Type non supporté. Autorisés: PDF, PNG, JPEG.")

    with top_right:
        if st.button(
            "Remplissage Automatique IA",
            use_container_width=True,
            disabled=(uploaded_file is None) or (not file_ok),
        ):
            if uploaded_file is None:
                st.warning("Upload un fichier.")
                st.stop()
            if not file_ok:
                st.stop()

            try:
                file_bytes = uploaded_file.getvalue()
                if len(file_bytes) > _MAX_UPLOAD_BYTES:
                    st.error("Fichier trop gros : limite 3 Mo.")
                    st.stop()

                _, current_payload = _read_lithologie(common_data)

                with st.spinner("Analyse…"):
                    new_payload = _ai_complete_lithologie_from_document(
                        file_bytes=file_bytes,
                        filename=uploaded_file.name or "document",
                        mime_type=mime_type or (uploaded_file.type or ""),
                        current_payload=current_payload,
                        model="gpt-4o-mini",
                    )

                _write_lithologie(path, new_payload)

                st.session_state["litho_rows_live"] = new_payload["rows"]
                st.session_state["litho_rev"] += 1
                st.success("Fait.")
                st.rerun()
            except Exception as e:
                st.error("Échec IA.")
                st.exception(e)

    st.divider()

    # ==================================================
    # ✅ TABLEUR EN DESSOUS
    # ==================================================
    data = {"palette": payload["palette"], "rows": payload["rows"]}

    event = lithologie_component(
        data=data,
        key=f"lithologie_component_{st.session_state['litho_rev']}",
    )

    if event and isinstance(event, dict) and event.get("type") == "change":
        rows = event.get("rows")
        if isinstance(rows, list):
            palette = payload["palette"]
            norm = []
            for i in range(10):
                sol = f"SOL {i+1}"
                src = rows[i] if i < len(rows) and isinstance(rows[i], dict) else {}
                nom = "" if src.get("nom") is None else str(src.get("nom", ""))
                col = src.get("couleur", palette[i % len(palette)])
                col = str(col).strip().upper() if col is not None else palette[0]
                if not _HEX_RE.match(col):
                    col = palette[0]
                norm.append({"sol": sol, "nom": nom, "couleur": col})
            st.session_state["litho_rows_live"] = norm

    # ==================================================
    # ✅ Enregistrer sous le tableur
    # ==================================================
    if st.button("Enregistrer", use_container_width=True):
        rows_to_save = st.session_state.get("litho_rows_live", payload["rows"])
        payload_out = {"version": 4, "palette": payload["palette"], "rows": rows_to_save}
        _write_lithologie(path, payload_out)

        st.session_state["litho_rows_live"] = rows_to_save
        st.session_state["litho_rev"] += 1
        st.success("Enregistré.")
        st.rerun()
