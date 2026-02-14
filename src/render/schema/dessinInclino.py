import pandas as pd
from src.render.schema import utils_schema as dessinfonctions


def _is_blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return str(v).strip() == ""


def _f(v, default=0.0) -> float:
    try:
        if _is_blank(v):
            return float(default)
        s = str(v).strip().replace(" ", "").replace(",", ".")
        return float(s)
    except Exception:
        return float(default)


def _non_empty_cols_in_row(df: pd.DataFrame, row0: int) -> list[int]:
    """Liste des colonnes (0-based) non vides sur la ligne row0, triées croissant."""
    if row0 < 0 or row0 >= df.shape[0]:
        return []
    cols = []
    for c in range(df.shape[1]):
        if not _is_blank(df.iat[row0, c]):
            cols.append(c)
    return cols


def inclinoUnique(
    Inclino: str,          # nom de feuille (ex: "Inclino")
    nomReel: str,          # juste pour info
    nomShape: str,         # nom de shape
    Recul: float,
    Zincl,                 # 0 => ZHautParoi
    Col: int,              # 0 => dernière colonne non vide / -1 => avant-dernière / >0 => index excel 1-based
    StyleDeCourbe: str,    # "solid" / "dashed"
    epaisseur: float,
    transparence: float,
    facteurSpecial: float,
    shapes: dict,
    couleur: str = "black" # ✅ nouveau
):
    """
    Construit une polyline (points) depuis la feuille Inclino.

    Convention Col :
    - Col == 0  : dernière colonne non vide (sur la ligne 2 / index 1)
    - Col == -1 : avant-dernière colonne non vide (sur la ligne 2 / index 1)
    - Col > 0   : colonne Excel 1-based (comme VBA)
    """
    classeur = dessinfonctions.classeurCoupe()
    df = pd.read_excel(classeur, sheet_name=Inclino, header=None, engine="openpyxl")

    if facteurSpecial in (None, "", 0):
        facteurSpecial = 1

    if Zincl in (None, "", 0):
        Zincl = dessinfonctions.ZHautParoi()

    # ligne 2 Excel => row0=1
    cols = _non_empty_cols_in_row(df, row0=1)
    if not cols:
        return shapes

    if Col is None:
        Col = 0

    if int(Col) == 0:
        col0 = cols[-1]  # dernière
    elif int(Col) == -1:
        if len(cols) < 2:
            # pas d'avant-dernière -> on ne trace rien
            return shapes
        col0 = cols[-2]  # avant-dernière
    else:
        col0 = int(Col) - 1
        if col0 < 0 or col0 >= df.shape[1]:
            return shapes

    ZeroDessin = dessinfonctions.MilieuParoi(shapes) - float(Recul) * (dessinfonctions.facteurh() * 1.0)

    pts = []
    i0 = 1  # premligne = 2 => index 1
    while i0 < df.shape[0]:
        prof = df.iat[i0, 0] if 0 < df.shape[1] else ""
        if _is_blank(prof):
            break

        val = df.iat[i0, col0] if col0 < df.shape[1] else ""
        if not _is_blank(val):
            prof = _f(prof)
            val = _f(val)
            x = val * float(facteurSpecial) * float(dessinfonctions.facteurmm()) + float(ZeroDessin)
            y = float(dessinfonctions.ZDessin(float(Zincl) - prof))
            pts.append((x, y))

        i0 += 1

    if len(pts) < 2:
        return shapes

    # bbox + padding
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    pad = 12
    minx -= pad
    maxx += pad
    miny -= pad
    maxy += pad

    key = dessinfonctions.NomShape(nomShape)
    sh = dessinfonctions.Shapes.get(key)
    sh = dessinfonctions.CreerSolSiNecessaire(sh, key)

    sh.kind = "polyline"
    sh.Left = float(minx)
    sh.Top = float(miny)
    sh.Width = float(max(1.0, maxx - minx))
    sh.Height = float(max(1.0, maxy - miny))
    sh.points = [(float(x - minx), float(y - miny)) for x, y in pts]

    # ✅ couleur
    sh.stroke = str(couleur)

    sh.stroke_width = float(epaisseur)
    sh.transparence = float(transparence)

    style = str(StyleDeCourbe).lower()
    sh.dash = "6 4" if style in ("dashed", "dash", "pointille", "pointillé") else ""

    # un poil au-dessus si pointillé (2e courbe) pour bien la voir
    sh.zIndex = 61 if sh.dash else 60

    return shapes
