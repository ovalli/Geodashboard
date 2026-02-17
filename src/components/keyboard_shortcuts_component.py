from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT = None

# anti-cache dur
_BASE = Path(tempfile.gettempdir()) / "geodashboard_keyboard_shortcuts_component_v1"


def keyboard_shortcuts_component(enabled: bool, context: str, key: str):
    """
    context:
      - "rename": Esc => cancel
      - "delete": Enter => confirm, Esc => cancel
    returns: dict | None  e.g. {"action":"cancel"} / {"action":"confirm"} / None
    """
    _ensure_component()
    return _COMPONENT(enabled=enabled, context=context, default=None, key=key)  # type: ignore[name-defined]


def _ensure_component() -> None:
    global _COMPONENT
    _BASE.mkdir(parents=True, exist_ok=True)

    idx = _BASE / "index.html"
    idx.write_text(_index_html(), encoding="utf-8")

    _COMPONENT = components.declare_component("keyboard_shortcuts_component", path=str(_BASE))


def _index_html() -> str:
    # ⚠️ Ce mini composant utilise l'API Streamlit Components via window.Streamlit
    # Il écoute keydown global et renvoie une valeur au backend.
    return r"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      /* composant invisible */
      html, body { margin:0; padding:0; width:0; height:0; overflow:hidden; }
    </style>
  </head>
  <body>
    <script>
      const Streamlit = window.Streamlit;

      let current = { enabled:false, context:"" };
      let lastSent = 0;

      function send(action) {
        const now = Date.now();
        if (now - lastSent < 150) return; // throttle
        lastSent = now;

        Streamlit.setComponentValue({ action });
        // reset rapide pour permettre un second event identique
        setTimeout(() => Streamlit.setComponentValue(null), 30);
      }

      function onKeyDown(e) {
        if (!current.enabled) return;
        if (e.defaultPrevented) return;
        if (e.altKey || e.ctrlKey || e.metaKey) return;

        const k = e.key;

        if (current.context === "rename") {
          if (k === "Escape") {
            e.preventDefault();
            send("cancel");
          }
          return;
        }

        if (current.context === "delete") {
          if (k === "Escape") {
            e.preventDefault();
            send("cancel");
            return;
          }
          if (k === "Enter") {
            // Enter confirme
            e.preventDefault();
            send("confirm");
            return;
          }
        }
      }

      function onRender(event) {
        const args = event.detail.args || {};
        current.enabled = !!args.enabled;
        current.context = (args.context || "");
        Streamlit.setFrameHeight(0);
      }

      window.addEventListener("keydown", onKeyDown, true);
      Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
      Streamlit.setComponentReady();
      Streamlit.setFrameHeight(0);
      Streamlit.setComponentValue(null);
    </script>
  </body>
</html>
"""
