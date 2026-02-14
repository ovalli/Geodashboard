import math
from typing import Optional

import src.io.excel_reader as liredansexcel
from src.render.schema import utils_schema as dessinfonctions


def _is_blank(v) -> bool:
    return v is None or str(v).strip() == ""


def _safe_float(v, default=0.0) -> float:
    try:
        if v is None:
            return float(default)
        s = str(v).strip()
        if s == "":
            return float(default)
        s = s.replace(" ", "").replace(",", ".")
        return float(s)
    except Exception:
        return float(default)


def _sanitize_num(v, nd=2) -> str:
    x = _safe_float(v, 0.0)
    s = f"{x:.{nd}f}"
    return s.replace("-", "m").replace(".", "p")


def _tirant_name(prefix: str, AltTir, Incl, Libre, Ancree) -> str:
    return (
        f"{prefix}_"
        f"{_sanitize_num(AltTir,2)}_"
        f"{_sanitize_num(Incl,1)}_"
        f"{_sanitize_num(Libre,2)}_"
        f"{_sanitize_num(Ancree,2)}"
    ).upper()


def _make_polyline_shape(
    name: str,
    x1: float, y1: float,
    x2: float, y2: float,
    stroke: str,
    stroke_width: float,
    transparence: float,
    z: int = 70,
    dash: str = "",
    linecap: Optional[str] = None,
    linejoin: Optional[str] = None,
    arrow: Optional[bool] = None,
):
    key = dessinfonctions.NomShape(name)
    sh = dessinfonctions.Shapes.get(key)
    sh = dessinfonctions.CreerSolSiNecessaire(sh, key)

    minx, maxx = (x1, x2) if x1 <= x2 else (x2, x1)
    miny, maxy = (y1, y2) if y1 <= y2 else (y2, y1)

    sh.kind = "polyline"
    sh.Left = minx
    sh.Top = miny
    sh.Width = max(1.0, maxx - minx)
    sh.Height = max(1.0, maxy - miny)

    sh.points = [(x1 - minx, y1 - miny), (x2 - minx, y2 - miny)]
    sh.stroke = stroke
    sh.stroke_width = float(stroke_width)
    sh.transparence = float(transparence)
    sh.dash = dash
    sh.zIndex = z

    # styles SVG (utilisés par render_html.py)
    if linecap is not None:
        sh.linecap = linecap  # "round" | "butt" | "square"
    if linejoin is not None:
        sh.linejoin = linejoin  # "round" | "miter" | "bevel"
    if arrow is not None:
        sh.arrow = bool(arrow)

    return sh


def tirantUnitaire(shapes, AltTir, Incl, Libre, Ancree, Forcing):
    AltTir = _safe_float(AltTir)
    Incl = _safe_float(Incl)
    Libre = _safe_float(Libre)
    Ancree = _safe_float(Ancree)
    forcing = "" if Forcing is None else str(Forcing).strip().lower()

    Ltot = Ancree + Libre
    if Ltot <= 0:
        return shapes

    d = float(dessinfonctions.ZDessin(AltTir))
    c = float(dessinfonctions.GaucheParoi(shapes))

    a = c - Ltot * float(dessinfonctions.facteurh()) * math.cos(math.radians(Incl))
    b = d + Ltot * (float(dessinfonctions.echelle()) / float(dessinfonctions.echellebis())) * math.sin(math.radians(Incl))

    transp = float(dessinfonctions.Transptir(AltTir, forcing))

    couleur_tirant = "rgb(0,0,0)"
    couleur_bulbe = "rgb(160,160,160)"

    name_tirant = _tirant_name("tirant", AltTir, Incl, Libre, Ancree)
    name_bulbe = _tirant_name("bulbe", AltTir, Incl, Libre, Ancree)

    # Tirant (noir fin)
    _make_polyline_shape(
        name=name_tirant,
        x1=a, y1=b, x2=c, y2=d,
        stroke=couleur_tirant,
        stroke_width=2.2,
        transparence=transp,
        z=80,
        linecap="butt",
        linejoin="miter",
        arrow=False
    )

    # Bulbe (gris épais) : oblong => caps arrondis
    r = 0.0 if Ltot == 0 else (Libre / Ltot)
    if r != 0:
        x2b = a + (c - a) * r
        y2b = b + (d - b) * r
        _make_polyline_shape(
            name=name_bulbe,
            x1=a, y1=b, x2=x2b, y2=y2b,
            stroke=couleur_bulbe,
            stroke_width=12.0,
            transparence=transp,
            z=79,
            linecap="round",
            linejoin="round",
            arrow=False
        )

    return shapes


def tirants(shapes, sheet_name: str = "Tirants 1"):
    classeur = dessinfonctions.classeurCoupe()
    if hasattr(dessinfonctions, "sheetExist"):
        if not dessinfonctions.sheetExist(classeur, sheet_name):
            return shapes

    i = 2
    while True:
        alt = liredansexcel.LireExcel(classeur, sheet_name, i, 1)
        if _is_blank(alt):
            break

        incl = liredansexcel.LireExcel(classeur, sheet_name, i, 2)
        libre = liredansexcel.LireExcel(classeur, sheet_name, i, 3)
        ancree = liredansexcel.LireExcel(classeur, sheet_name, i, 4)
        forcing = liredansexcel.LireExcel(classeur, sheet_name, i, 9)

        tirantUnitaire(shapes, alt, incl, libre, ancree, forcing)
        i += 1

    return shapes
