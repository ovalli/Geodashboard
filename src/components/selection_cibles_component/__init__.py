from __future__ import annotations
from pathlib import Path
import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

_selection_cibles = components.declare_component(
    "selection_cibles_component",
    path=str(_FRONTEND_DIR),
)

def selection_cibles_component(srcdoc: str, height: int = 720, key: str | None = None):
    return _selection_cibles(srcdoc=srcdoc, height=height, key=key, default=None)
