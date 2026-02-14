import math
from functools import lru_cache

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


def _orange_css():
    return "rgb(255,165,0)"


def _blue_css():
    return "rgb(0,90,255)"


def _shape_key(name: str) -> str:
    return dessinfonctions.NomShape(name)


@lru_cache(maxsize=64)
def _df_topo(workbook_path: str) -> pd.DataFrame:
    return pd.read_excel(
        workbook_path,
        sheet_name="Topo",
        header=None,
        engine="openpyxl"
    )


def _sheet_exists_topo(workbook_path: str) -> bool:
    try:
        _df_topo(workbook_path)
        return True
    except Exception:
        return False


def _last_contiguous_non_empty_row_in_col(df: pd.DataFrame, col0: int, start_row0: int) -> int:
    last = start_row0
    r = start_row0
    while r < df.shape[0]:
        v = df.iat[r, col0] if col0 < df.shape[1] else ""
        if _is_blank(v):
            break
        last = r
        r += 1
    return last + 1  # 1-based


def FormeCible(nomCible: str):
    key = _shape_key(f"CIBLE {nomCible}")
    return dessinfonctions.Shapes.get(key)


def CIbleUnitaire(nomCible: str, a: float, b: float, Diam: float):
    nomForme = f"CIBLE {nomCible}"
    key = _shape_key(nomForme)
    sh = dessinfonctions.Shapes.get(key)
    sh = dessinfonctions.CreerSolSiNecessaire(sh, key)

    # ✅ CIBLE RONDE
    sh.kind = "circle"
    sh.Left = a - Diam / 2
    sh.Top = b - Diam / 2
    sh.Width = Diam
    sh.Height = Diam

    sh.fill = "rgba(255,0,0,0.85)"
    sh.stroke = "rgba(0,0,0,0.90)"
    sh.stroke_width = 0.9
    sh.border = "0.9px solid rgba(0,0,0,0.90)"  # compat si jamais

    sh.zIndex = 120
    sh.text = ""

    # Texte centré
    tkey = _shape_key(f"TEXTE_CIBLE_{nomCible}")
    t = dessinfonctions.Shapes.get(tkey)
    t = dessinfonctions.CreerSolSiNecessaire(t, tkey)

    t.kind = "text"
    t.text = str(nomCible)
    t.Left = sh.Left
    t.Top = sh.Top + (Diam / 2) - 6
    t.Width = Diam
    t.Height = 12
    t.zIndex = 121

    return sh


def CentreHCible(nomCible: str) -> float:
    f = FormeCible(nomCible)
    if not f:
        return 0.0
    return float(f.Left + f.Width / 2)


def CentreVCible(nomCible: str) -> float:
    f = FormeCible(nomCible)
    if not f:
        return 0.0
    return float(f.Top + f.Height / 2)


def cibles(shapes, workbook_path: str):
    df = _df_topo(workbook_path)

    ang = _f(dessinfonctions.angle(), 0.0)
    ca = math.cos(math.radians(ang))
    sa = math.sin(math.radians(ang))

    xref = dessinfonctions.Xref()
    yref = dessinfonctions.Yref()
    if _is_blank(xref):
        return shapes

    c = 5
    while c < df.shape[1]:
        nomCible = df.iat[0, c] if df.shape[0] > 0 else ""
        if _is_blank(nomCible):
            break

        CoordX = df.iat[1, c] if df.shape[0] > 1 else ""
        CoordY = df.iat[1, c + 1] if (df.shape[0] > 1 and c + 1 < df.shape[1]) else ""
        Z = df.iat[1, c + 2] if (df.shape[0] > 1 and c + 2 < df.shape[1]) else ""

        if (not _is_blank(CoordX)) and (not _is_blank(CoordY)):
            dX = _f(xref) - _f(CoordX)
            dY = _f(yref) - _f(CoordY)

            e = ca * dX + sa * dY

            a = float(dessinfonctions.MilieuParoi(shapes)) - e * float(dessinfonctions.facteurh())
            b = float(dessinfonctions.ZDessin(_f(Z)))

            CIbleUnitaire(str(nomCible), a, b, float(dessinfonctions.DiamCible()))

        c += 6

    return shapes


def _transp_vecteur_df(df: pd.DataFrame, row1: int, col1: int) -> float:
    r = row1 - 1
    c = col1 - 1
    v = df.iat[r, c] if (0 <= r < df.shape[0] and 0 <= c < df.shape[1]) else ""
    if not _is_blank(v):
        return 0.0

    rr = r - 1
    while rr >= 0:
        vv = df.iat[rr, c] if (0 <= c < df.shape[1]) else ""
        if not _is_blank(vv):
            break
        rr -= 1

    if rr < 0:
        return 1.0

    transp = 0.02 * ((r + 1) - (rr + 1))
    return 1.0 if transp >= 1.0 else float(transp)


def _ligne_der_mes_df(df: pd.DataFrame, row1: int, col1: int):
    r = row1 - 1
    c = col1 - 1

    if (0 <= r < df.shape[0] and 0 <= c < df.shape[1]) and (not _is_blank(df.iat[r, c])):
        ligne = r + 1
    else:
        rr = r - 1
        while rr >= 0 and (0 <= c < df.shape[1]) and _is_blank(df.iat[rr, c]):
            rr -= 1
        ligne = rr + 1 if rr >= 0 else ""

    if ligne != "" and ligne < 3:
        return ""
    return ligne


def _ligne_mes_prec_df(df: pd.DataFrame, row1: int, col1: int):
    ligne = int(row1) - 14
    if ligne <= 3:
        return ""

    c = col1 - 1

    def empty(r1):
        rr = r1 - 1
        if rr < 0 or rr >= df.shape[0] or c < 0 or c >= df.shape[1]:
            return True
        return _is_blank(df.iat[rr, c])

    if empty(ligne):
        ligne -= 1
    if empty(ligne):
        ligne += 2
    if empty(ligne):
        ligne -= 3

    return "" if empty(ligne) else ligne


def vecteurUnitaire(nomVecteur: str, a: float, b: float, dn: float, dv: float, couleurDefaut_css: str, transp: float):
    if abs(dn) >= dessinfonctions.LimiteReprmm() or abs(dv) >= dessinfonctions.LimiteReprmm():
        return

    dnDessin = dn * float(dessinfonctions.facteurmm())
    dvDessin = -dv * float(dessinfonctions.facteurVmm())

    x2 = a + dnDessin
    y2 = b + dvDessin

    key = _shape_key(nomVecteur)
    sh = dessinfonctions.Shapes.get(key)
    sh = dessinfonctions.CreerSolSiNecessaire(sh, key)

    minx, maxx = (a, x2) if a <= x2 else (x2, a)
    miny, maxy = (b, y2) if b <= y2 else (y2, b)

    pad = 12
    minx -= pad
    maxx += pad
    miny -= pad
    maxy += pad

    sh.kind = "polyline"
    sh.Left = float(minx)
    sh.Top = float(miny)
    sh.Width = float(max(1.0, maxx - minx))
    sh.Height = float(max(1.0, maxy - miny))
    sh.points = [(float(a - minx), float(b - miny)), (float(x2 - minx), float(y2 - miny))]

    sh.stroke = couleurDefaut_css
    sh.stroke_width = 1.2
    sh.transparence = float(transp)
    sh.dash = ""
    sh.zIndex = 150

    sh.arrow = True
    sh.arrow_size = 7


def topoVecteurs(shapes, workbook_path: str):
    df = _df_topo(workbook_path)
    LigneDerDate = _last_contiguous_non_empty_row_in_col(df, col0=0, start_row0=1)

    k = 5
    while k < df.shape[1]:
        nomCible = df.iat[0, k] if df.shape[0] > 0 else ""
        if _is_blank(nomCible):
            break
        nomCible = str(nomCible)

        a = CentreHCible(nomCible)
        b = CentreVCible(nomCible)
        if a == 0 and b == 0:
            k += 6
            continue

        transp = _transp_vecteur_df(df, LigneDerDate, k + 1)
        L = _ligne_der_mes_df(df, LigneDerDate, k + 1)

        L2 = ""
        if L != "":
            L2 = _ligne_mes_prec_df(df, L, k + 1)

            dnReel = _f(df.iat[L - 1, (k + 3)]) if (L - 1) < df.shape[0] and (k + 3) < df.shape[1] else 0.0
            dtReel = _f(df.iat[L - 1, (k + 4)]) if (L - 1) < df.shape[0] and (k + 4) < df.shape[1] else 0.0
            dvReel = _f(df.iat[L - 1, (k + 5)]) if (L - 1) < df.shape[0] and (k + 5) < df.shape[1] else 0.0

            nomVecteur = f"vecteur {nomCible} dN={round(dnReel,1)} dT={round(dtReel,1)} dZ={round(dvReel,1)}"
            vecteurUnitaire(nomVecteur, a, b, dnReel, dvReel, _orange_css(), transp)

        if L2 != "":
            DateL2 = df.iat[L2 - 1, 0] if (L2 - 1) < df.shape[0] else ""

            dn_L = _f(df.iat[L - 1, (k + 3)]) if (L - 1) < df.shape[0] and (k + 3) < df.shape[1] else 0.0
            dv_L = _f(df.iat[L - 1, (k + 5)]) if (L - 1) < df.shape[0] and (k + 5) < df.shape[1] else 0.0

            dn_P = _f(df.iat[L2 - 1, (k + 3)]) if (L2 - 1) < df.shape[0] and (k + 3) < df.shape[1] else 0.0
            dv_P = _f(df.iat[L2 - 1, (k + 5)]) if (L2 - 1) < df.shape[0] and (k + 5) < df.shape[1] else 0.0

            dnR = dn_L - dn_P
            dvR = dv_L - dv_P

            nomVecteur = f"vecteur relatif {nomCible} {DateL2}"
            vecteurUnitaire(nomVecteur, a, b, dnR, dvR, _blue_css(), transp)

        k += 6

    return shapes


def Topo(shapes, workbook_path: str = None):
    if workbook_path is None:
        workbook_path = dessinfonctions.classeurCoupe()

    if not _sheet_exists_topo(workbook_path):
        return shapes

    cibles(shapes, workbook_path)
    topoVecteurs(shapes, workbook_path)
    return shapes
