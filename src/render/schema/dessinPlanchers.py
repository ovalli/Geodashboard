from __future__ import annotations

from typing import Any, Dict

import src.io.excel_reader as liredansexcel
from src.render.schema import utils_schema as dessinfonctions


def _is_blank(v) -> bool:
    return v is None or str(v).strip() == ""


def _safe_float(v, default: float = 0.0) -> float:
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


def _sanitize_num(v, nd: int = 2) -> str:
    x = _safe_float(v, 0.0)
    s = f"{x:.{nd}f}"
    return s.replace("-", "m").replace(".", "p")


def _plancher_name(z_plancher) -> str:
    return f"PLANCHER_{_sanitize_num(z_plancher, 2)}".upper()


def plancher_unitaire(shapes: Dict[str, Any], z_plancher) -> Dict[str, Any]:
    """Ajoute un plancher (rectangle) à droite de la paroi, au Z donné."""
    if _is_blank(z_plancher):
        return shapes

    z = _safe_float(z_plancher, 0.0)

    # géométrie (comme VBA)
    e = float(dessinfonctions.DroiteParoi(shapes))
    f = float(dessinfonctions.ZDessin(z))
    g = 22.0 * float(dessinfonctions.LargeurParoi(shapes))
    h = 10.0

    # si paroi absente ou largeur nulle, on ne trace pas
    if g <= 0.0:
        return shapes

    key = dessinfonctions.NomShape(_plancher_name(z))
    sh = dessinfonctions.Shapes.get(key)
    sh = dessinfonctions.CreerSolSiNecessaire(sh, key)

    sh.kind = "rect"
    sh.Left = e
    sh.Top = f
    sh.Width = g
    sh.Height = h

    # style (approx du VBA)
    # VBA: .Line.Transparency = 0.5 (=> stroke semi-transparent)
    sh.stroke = "rgba(0,0,0,0.5)"
    sh.stroke_width = 1.2

    # VBA: fill sombre + pattern 20% (on approx par un fill sombre lisible)
    sh.fill = "rgba(0,0,0,0.22)"

    # VBA: If ZPlancher < DerExcav() - 1 Then .Transparency = 0.5
    if z < float(dessinfonctions.DerExcav()) - 1.0:
        sh.transparence = 0.5  # => opacity 0.5 côté rendu
    else:
        sh.transparence = 0.0

    # au-dessus de la fouille (même logique que tirants)
    sh.zIndex = 80

    shapes[key] = sh
    return shapes


def planchers(shapes: Dict[str, Any], sheet_name: str = "Planchers") -> Dict[str, Any]:
    """Dessine les planchers listés dans la feuille 'Planchers' (col A = Z)."""
    classeur = dessinfonctions.classeurCoupe()
    if hasattr(dessinfonctions, "sheetExist"):
        if not dessinfonctions.sheetExist(classeur, sheet_name):
            return shapes

    i = 2
    while True:
        z = liredansexcel.LireExcel(classeur, sheet_name, i, 1)
        if _is_blank(z):
            break
        plancher_unitaire(shapes, z)
        i += 1

    return shapes
