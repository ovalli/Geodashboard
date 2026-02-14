import html
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_iframe_css() -> str:
    """
    Single source of truth for hover styling.

    The Streamlit page loads styles/main.css globally, but the schema is rendered
    inside an iframe (components.html), so we also need to inline the same CSS
    in the iframe document.
    """
    try:
        root = Path(__file__).resolve().parents[2]  # project root
        css_path = root / "styles" / "main.css"
        if css_path.exists():
            return css_path.read_text(encoding="utf-8")
    except Exception:
        pass
    return ""


def _normalize_shapes(shapes: Any) -> List[Any]:
    if shapes is None:
        return []
    if isinstance(shapes, (list, tuple)):
        return list(shapes)
    if isinstance(shapes, dict):
        if "shapes" in shapes and isinstance(shapes["shapes"], (list, tuple)):
            return list(shapes["shapes"])
        return list(shapes.values())
    return []


def _as_dict(sh: Any) -> Dict[str, Any]:
    if sh is None:
        return {}
    if isinstance(sh, dict):
        return sh
    if hasattr(sh, "__dict__"):
        try:
            return dict(sh.__dict__)
        except Exception:
            return {}
    return {}


def _parse_border(border: str) -> Tuple[float, str]:
    if not border:
        return 0.0, ""
    s = str(border).strip().lower()
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*px\s+solid\s+(.+?)\s*$", s)
    if not m:
        return 0.0, ""
    try:
        w = float(m.group(1))
    except Exception:
        w = 0.0
    col = m.group(2).strip()
    return w, col


def _looks_like_red(fill: str) -> bool:
    if not fill:
        return False
    s = str(fill).replace(" ", "").lower()
    return ("rgba(255,0,0" in s) or ("rgb(255,0,0" in s)


def _needs_hit_fill(fill: str) -> bool:
    if fill is None:
        return True
    s = str(fill).strip().lower().replace(" ", "")
    if s in ("", "none", "transparent"):
        return True
    if s.startswith("rgba(") and s.endswith(")"):
        inside = s[5:-1]
        parts = inside.split(",")
        if len(parts) == 4:
            try:
                a = float(parts[3])
                if a <= 0.0:
                    return True
            except Exception:
                pass
    return False


def _force_opaque_color(c: str) -> str:
    """
    Force une couleur opaque pour les polylines.
    - rgba(r,g,b,a) -> rgb(r,g,b)
    - #RRGGBBAA -> #RRGGBB
    """
    if not c:
        return c
    s = str(c).strip()
    s_l = s.lower().replace(" ", "")

    if s_l.startswith("rgba(") and s_l.endswith(")"):
        inside = s_l[5:-1]
        parts = inside.split(",")
        if len(parts) >= 3:
            try:
                r = int(float(parts[0]))
                g = int(float(parts[1]))
                b = int(float(parts[2]))
                return f"rgb({r},{g},{b})"
            except Exception:
                return s

    if s_l.startswith("#") and len(s_l) == 9:
        return s[:7]

    return s


def _extract_cible_id_from_name(name_raw: str) -> str:
    """
    Tente d'extraire l'identifiant de cible depuis le Name.
    - "CIBLE A1" -> "A1"
    - "vecteur A1 ..." / "vecteur relatif A1 ..." -> "A1"
    """
    if not name_raw:
        return ""
    s = str(name_raw).strip()

    m = re.match(r"^\s*cible\s+(.+?)\s*$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.match(r"^\s*vecteur(?:\s+relatif)?\s+([^\s]+)\b", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return ""


def compute_canvas_bounds(
    shapes: Any,
    margin: int = 20,
    min_width: int = 600,
    min_height: int = 300,
) -> Tuple[float, float, int, int]:
    shapes_list = _normalize_shapes(shapes)

    xs: List[float] = []
    ys: List[float] = []

    for sh in shapes_list:
        d = _as_dict(sh)
        if not d:
            continue

        left = float(d.get("Left") or 0.0)
        top = float(d.get("Top") or 0.0)
        w = float(d.get("Width") or 0.0)
        h = float(d.get("Height") or 0.0)

        x1, x2 = (left, left + w) if w >= 0 else (left + w, left)
        y1, y2 = (top, top + h) if h >= 0 else (top + h, top)

        xs.extend([x1, x2])
        ys.extend([y1, y2])

        pts = d.get("points")
        if isinstance(pts, list):
            for p in pts:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    xs.append(left + float(p[0] or 0.0))
                    ys.append(top + float(p[1] or 0.0))

    if not xs or not ys:
        return float(margin), float(margin), int(min_width), int(min_height)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = int(max(min_width, (max_x - min_x) + 2 * margin))
    height = int(max(min_height, (max_y - min_y) + 2 * margin))

    offset_x = float(margin) - float(min_x)
    offset_y = float(margin) - float(min_y)

    return offset_x, offset_y, width, height


def render_shapes_html(
    shapes: Any,
    width: int,
    height: int,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> str:
    shapes_list = _normalize_shapes(shapes)

    def z_of(d: Dict[str, Any]) -> float:
        try:
            return float(d.get("zIndex") or 0.0)
        except Exception:
            return 0.0

    ds = [_as_dict(s) for s in shapes_list]
    ds = [d for d in ds if d]

    # ------------------------------------------------------
    # PREPASS: centres des CIBLES pour supprimer le vieux texte centré
    # ------------------------------------------------------
    cible_centers: List[Tuple[float, float, float]] = []

    for d in ds:
        name_raw = str(d.get("Name") or d.get("name") or "")
        name_clean = name_raw.strip()
        name_norm = name_clean.upper()

        if not name_norm.startswith("CIBLE "):
            continue

        left = float(d.get("Left") or 0.0) + float(offset_x or 0.0)
        top = float(d.get("Top") or 0.0) + float(offset_y or 0.0)
        w = float(d.get("Width") or 0.0)
        h = float(d.get("Height") or 0.0)

        x = left
        y = top
        ww = w
        hh = h
        if ww < 0:
            x = x + ww
            ww = -ww
        if hh < 0:
            y = y + hh
            hh = -hh

        if ww > 0 and hh > 0:
            r = min(ww, hh) / 2.0
            cx = x + ww / 2.0
            cy = y + hh / 2.0
            cible_centers.append((cx, cy, r))

    def _text_is_on_a_cible(cx: float, cy: float) -> bool:
        for (x0, y0, r0) in cible_centers:
            dx = cx - x0
            dy = cy - y0
            if (dx * dx + dy * dy) <= (0.60 * r0) ** 2:
                return True
        return False

    # ------------------------------------------------------
    # LAYER ORDER: sols < fouille < reste < paroi < cibles < polylines
    # ------------------------------------------------------
    def _layer_of(d: Dict[str, Any]) -> int:
        kind = str(d.get("kind") or "rect").strip().lower()
        name_raw = str(d.get("Name") or d.get("name") or "")
        name_norm = name_raw.strip().upper()

        is_cible = name_norm.startswith("CIBLE ")
        is_paroi = (name_norm == "PAROI")
        is_fouille = (name_norm == "FOUILLE")

        if is_fouille:
            return 10
        if is_paroi:
            return 80
        if is_cible:
            return 90
        if kind == "polyline":
            return 95
        return 50

    ds.sort(key=lambda d: (z_of(d), _layer_of(d)))

    parts: List[str] = []

    parts.append("""
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L10,3.5 L0,7 z" fill="context-stroke"></path>
  </marker>
</defs>
""".strip())

    for d in ds:
        kind = str(d.get("kind") or "rect").strip().lower()

        name_raw = str(d.get("Name") or d.get("name") or "")
        name_clean = name_raw.strip()
        name_norm = name_clean.upper()

        left = float(d.get("Left") or 0.0) + float(offset_x or 0.0)
        top = float(d.get("Top") or 0.0) + float(offset_y or 0.0)
        w = float(d.get("Width") or 0.0)
        h = float(d.get("Height") or 0.0)

        x = left
        y = top
        ww = w
        hh = h
        if ww < 0:
            x = x + ww
            ww = -ww
        if hh < 0:
            y = y + hh
            hh = -hh

        fill_src_present = ("fill" in d)
        fill = d.get("fill") if fill_src_present else None

        stroke = d.get("stroke") if ("stroke" in d) else None
        stroke_w = d.get("stroke_width") if ("stroke_width" in d) else None

        if (not stroke) and d.get("border"):
            bw, bc = _parse_border(d.get("border"))
            if stroke_w is None and bw > 0:
                stroke_w = bw
            if bc:
                stroke = bc

        if stroke_w is None:
            stroke_w = 1.2
        try:
            stroke_w = float(stroke_w)
        except Exception:
            stroke_w = 1.2

        is_rect = (kind == "rect")
        is_special = name_norm in ("PAROI", "FOUILLE")
        is_fouille = (name_norm == "FOUILLE")
        is_cible = name_norm.startswith("CIBLE ")
        is_bulbe = name_norm.startswith("BULBE_")
        is_tirant = name_norm.startswith("TIRANT_")
        is_plancher = name_norm.startswith("PLANCHER_")

        # Sol = rect non special, non cible, non bulbe/tirant/plancher
        looks_like_soil = (
            is_rect
            and (not is_special)
            and (not is_cible)
            and (not is_bulbe)
            and (not is_tirant)
            and (not is_plancher)
        )

        # fallback fill minimal si jamais aucun fill
        if looks_like_soil and fill is None:
            fill = "rgba(120,120,120,0.10)"

        # Sols : jamais de contour
        if looks_like_soil:
            stroke = "none"
            stroke_w = 0.0

        if fill is None:
            fill = "rgba(0,0,0,0.04)"

        # Fouille : contour blanc
        if is_fouille:
            stroke = "rgba(255,255,255,1.0)"
            stroke_w = max(float(stroke_w or 1.2), 1.2)

        if not stroke:
            stroke = "rgba(0,0,0,0.75)"

        # transparence (sauf polylines)
        opacity = None
        if kind != "polyline" and d.get("transparence") is not None:
            try:
                tr = float(d.get("transparence"))
                opacity = max(0.0, min(1.0, 1.0 - tr))
            except Exception:
                opacity = None

        href = d.get("href") or d.get("url") or ""
        cursor = "cursor:pointer;" if href else ""
        data_href = f' data-href="{html.escape(str(href), quote=True)}"' if href else ""

        # ---------- GROUP HOVER ID (cible) ----------
        cible_id = ""
        if is_cible:
            cible_id = _extract_cible_id_from_name(name_clean)
        elif kind == "polyline":
            cible_id = _extract_cible_id_from_name(name_clean)

        data_cible = f' data-cible="{html.escape(cible_id, quote=True)}"' if cible_id else ""

        shape_class = "gd-shape gd-soil" if looks_like_soil else "gd-shape"

        style = f"vector-effect:non-scaling-stroke;{cursor}--sw:{stroke_w};stroke-width:var(--sw);"
        if opacity is not None:
            style += f"opacity:{opacity};"

        fill_for_svg = fill
        if _needs_hit_fill(fill_for_svg):
            fill_for_svg = "rgba(0,0,0,0.001)"

        # CIBLES : cercle forcé + label à gauche
        force_circle = (kind == "circle") or is_cible or _looks_like_red(fill)
        if force_circle and ww > 0 and hh > 0:
            r = min(ww, hh) / 2.0
            cx = x + ww / 2.0
            cy = y + hh / 2.0

            circle_svg = (
                f'<circle class="{shape_class}" cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" '
                f'style="{style}" fill="{fill_for_svg}" stroke="{stroke}"{data_href}{data_cible}></circle>'
            )

            if is_cible:
                label = name_clean
                m = re.match(r"^\s*cible\s+(.*)\s*$", name_clean, flags=re.IGNORECASE)
                if m:
                    label = m.group(1).strip()
                if not label:
                    label = name_clean

                pad = 10.0
                lx = cx - r - pad
                ly = cy

                text_svg = (
                    f'<text class="gd-cible-label" x="{lx:.3f}" y="{ly:.3f}" '
                    f'text-anchor="end" dominant-baseline="middle"{data_cible}>'
                    f'{html.escape(label)}</text>'
                )

                # <g> reçoit data-cible aussi
                parts.append(f'<g class="gd-cible" {data_cible.strip()}>{circle_svg}{text_svg}</g>')
            else:
                parts.append(circle_svg)

            continue

        # polylines (vecteurs): opaques + data-cible
        if kind == "polyline":
            pts = d.get("points")
            if isinstance(pts, list) and len(pts) >= 2:
                abs_pts = []
                ok = True
                for p in pts:
                    if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                        ok = False
                        break
                    abs_pts.append((x + float(p[0] or 0.0), y + float(p[1] or 0.0)))
                if ok and len(abs_pts) >= 2:
                    pts_str = " ".join([f"{px:.3f},{py:.3f}" for px, py in abs_pts])

                    arrow = bool(d.get("arrow"))
                    marker = ' marker-end="url(#arrow)"' if arrow else ""

                    dash = str(d.get("dash") or "").strip()
                    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""

                    cap = str(d.get("linecap") or "").lower().strip()
                    join = str(d.get("linejoin") or "").lower().strip()

                    if cap not in ("butt", "round", "square"):
                        cap = "round" if is_bulbe else "butt"
                    if join not in ("miter", "round", "bevel"):
                        join = "round" if is_bulbe else "miter"

                    stroke_opaque = _force_opaque_color(stroke)

                    parts.append(
                        f'<polyline class="{shape_class}" points="{pts_str}" fill="none" stroke="{stroke_opaque}" '
                        f'stroke-linecap="{cap}" stroke-linejoin="{join}" '
                        f'style="{style}opacity:1.0;"{dash_attr}{marker}{data_href}{data_cible}></polyline>'
                    )
            continue

        # text : supprimer les labels centrés sur les cibles
        if kind == "text":
            txt_raw = str(d.get("text") or "").strip()

            cx = x + ww / 2.0
            cy = y + hh / 2.0

            if _text_is_on_a_cible(cx, cy):
                continue

            fs = d.get("font_size") or 11
            tc = d.get("text_color") or "rgba(0,0,0,0.95)"
            parts.append(
                f'<text class="gd-text" x="{cx:.3f}" y="{cy:.3f}" text-anchor="middle" dominant-baseline="middle" '
                f'style="font-family:Arial, sans-serif;font-size:{float(fs)}px;fill:{tc};user-select:none;">'
                f'{html.escape(str(txt_raw))}</text>'
            )
            continue

        # rect default
        if ww > 0 and hh > 0:
            parts.append(
                f'<rect class="{shape_class}" x="{x:.3f}" y="{y:.3f}" width="{ww:.3f}" height="{hh:.3f}" '
                f'style="{style}" fill="{fill_for_svg}" stroke="{stroke}"{data_href}{data_cible}></rect>'
            )

    svg_inner = "\n".join(parts)

    # Single source of truth: styles/main.css (also injected in Streamlit page)
    iframe_css = _load_iframe_css()
    if not iframe_css.strip():
        # Minimal fallback (keeps the SVG visible even if main.css is missing)
        iframe_css = "html,body{margin:0;padding:0;}svg{display:block;background:white;}"

    css = f"<style>\n{iframe_css}\n</style>"

    js = """
<script>
(function(){
  const svg = document.getElementById('gdsvg');
  if (!svg) return;

  // --- click links ---
  svg.addEventListener('click', (e) => {
    const t = e.target;
    if (!t || !t.getAttribute) return;
    const href = t.getAttribute('data-href');
    if (href) {
      window.open(href, '_self');
      e.preventDefault();
      e.stopPropagation();
    }
  });

  // --- group hover (vecteur + cible + nom) ---
  let activeCible = null;

  function clearHover(){
    if (!activeCible) return;
    const nodes = svg.querySelectorAll('[data-cible="' + CSS.escape(activeCible) + '"]');
    nodes.forEach(n => n.classList.remove('gd-hover'));
    const groups = svg.querySelectorAll('g.gd-cible[data-cible="' + CSS.escape(activeCible) + '"]');
    groups.forEach(g => g.classList.remove('gd-hover'));
    activeCible = null;
  }

  function setHover(cible){
    if (!cible) return;
    if (activeCible === cible) return;
    clearHover();
    activeCible = cible;

    const nodes = svg.querySelectorAll('[data-cible="' + CSS.escape(cible) + '"]');
    nodes.forEach(n => {
      // ✅ ne jamais appliquer gd-hover aux sols
      if (n.classList && n.classList.contains('gd-soil')) return;
      n.classList.add('gd-hover');
    });

    const groups = svg.querySelectorAll('g.gd-cible[data-cible="' + CSS.escape(cible) + '"]');
    groups.forEach(g => g.classList.add('gd-hover'));
  }

  function pickCibleFromTarget(t){
    if (!t) return null;
    if (t.getAttribute && t.getAttribute('data-cible')) return t.getAttribute('data-cible');
    let p = t;
    while (p && p !== svg) {
      if (p.getAttribute && p.getAttribute('data-cible')) return p.getAttribute('data-cible');
      p = p.parentNode;
    }
    return null;
  }

  svg.addEventListener('mousemove', (e) => {
    const cible = pickCibleFromTarget(e.target);
    if (cible) setHover(cible);
    else clearHover();
  });

  svg.addEventListener('mouseleave', () => {
    clearHover();
  });

  // --- pan/zoom ---
  const vb = svg.viewBox.baseVal;
  const BASE = { x: vb.x, y: vb.y, w: vb.width, h: vb.height };

  let panning = false;
  let start = {x:0, y:0};
  let vb0 = {x:0, y:0};

  function svgPoint(evt){
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    const m = svg.getScreenCTM();
    return m ? pt.matrixTransform(m.inverse()) : {x:evt.clientX, y:evt.clientY};
  }

  svg.addEventListener('mousedown', (evt) => {
    if (evt.button !== 0) return;
    if (evt.target && evt.target.getAttribute && evt.target.getAttribute('data-href')) return;
    panning = true;
    const p = svgPoint(evt);
    start = {x:p.x, y:p.y};
    vb0 = {x:vb.x, y:vb.y};
    svg.style.cursor = 'grabbing';
  });

  window.addEventListener('mousemove', (evt) => {
    if (!panning) return;
    const p = svgPoint(evt);
    vb.x = vb0.x - (p.x - start.x);
    vb.y = vb0.y - (p.y - start.y);
  });

  window.addEventListener('mouseup', () => {
    panning = false;
    svg.style.cursor = 'default';
  });

  svg.addEventListener('wheel', (evt) => {
    evt.preventDefault();
    let scale = (evt.deltaY < 0) ? 0.9 : 1.1;

    if (scale > 1.0) {
      const maxOut = Math.min(BASE.w / vb.width, BASE.h / vb.height);
      if (maxOut >= 1.0 && maxOut <= 1.000001) return;
      if (scale > maxOut) {
        scale = maxOut;
        if (scale <= 1.000001) return;
      }
    }

    const p = svgPoint(evt);
    const nx = p.x, ny = p.y;

    vb.x = nx - (nx - vb.x) * scale;
    vb.y = ny - (ny - vb.y) * scale;
    vb.width *= scale;
    vb.height *= scale;

    const eps = 1e-3;
    if (vb.width >= BASE.w - eps && vb.height >= BASE.h - eps) {
      vb.x = BASE.x; vb.y = BASE.y; vb.width = BASE.w; vb.height = BASE.h;
    }
  }, {passive:false});
})();
</script>
""".strip()

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  {css}
</head>
<body>
  <svg id="gdsvg" width="{int(width)}" height="{int(height)}"
       viewBox="0 0 {int(width)} {int(height)}"
       xmlns="http://www.w3.org/2000/svg">
    {svg_inner}
  </svg>
  {js}
</body>
</html>"""
