import os
import math
import pandas as pd
import src.io.excel_reader as excel_reader
import src.io.excel_reader as liredansexcel




# =========================
# CLASSEUR CENTRALISÉ
# =========================

_CLASSEUR_COURANT = "monclasseur.xlsx"


def set_classeur(nom_fichier: str):
    """Permet de changer dynamiquement le classeur plus tard."""
    global _CLASSEUR_COURANT
    _CLASSEUR_COURANT = str(nom_fichier).strip()


def classeurCoupe() -> str:
    """Chemin absolu vers le classeur courant (dans le dossier de l'app)."""
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, _CLASSEUR_COURANT)


def sheetExist(classeur_abs_path: str, sheet_name: str) -> bool:
    try:
        pd.read_excel(classeur_abs_path, sheet_name=sheet_name, nrows=1, engine="openpyxl")
        return True
    except Exception:
        return False


def LignePara(i, j):
    return liredansexcel.LireExcel(classeurCoupe(), "Paramètres", i, j)


# =========================
# SHAPES
# =========================

def NomShape(nom):
    if nom is None:
        return ""
    return str(nom).strip().upper()


class Shape:
    def __init__(self, name: str):
        self.Name = name
        self.Top = 0.0
        self.Left = 0.0
        self.Height = 0.0
        self.Width = 0.0
        # champs optionnels (rendu)
        # kind, fill, border, stroke, stroke_width, points, zIndex, text, dash, arrow, arrow_size ...


Shapes = {}  # dict[str, Shape]


def CreerSolSiNecessaire(shape, nomSOL: str) -> Shape:
    nomSOL = NomShape(nomSOL)
    if shape is None:
        shape = Shape(nomSOL)
        Shapes[nomSOL] = shape
    return shape


# =========================
# PARAMÈTRES / ÉCHELLES
# =========================

def EchVm():
    return LignePara(7, 2)


def echelle():
    # provisoire (équivalent espacement vertical Excel)
    return 20


def echellebis():
    return EchVm()


def facteurmm():
    # conversion mm -> px (provisoire)
    return 1


def facteurh():
    # facteur horizontal topo/tirants (provisoire demandé)
    return 1


def facteurVmm():
    # demandé = 1 pour le moment
    return 1


# =========================
# GÉOMÉTRIE DESSIN
# =========================

def GaucheDessin():
    return 10


def DroiteDessin():
    return 580


def LargeurDessin():
    return DroiteDessin() - GaucheDessin()


def ZHautParoi():
    # cellule Geol (17,3) comme ton code
    return liredansexcel.LireExcel(classeurCoupe(), "Geol", 17, 3)


def ZParoi():
    # tu as dit qu'on peut garder qu'un seul : on aligne
    return ZHautParoi()


def ZDessin(z):
    z_ref = ZHautParoi()
    return (z_ref - z) * (echelle() / echellebis())


def GaucheParoi(shapes: dict):
    for k, s in shapes.items():
        if str(k).lower() == "paroi":
            return s.Left
    return 0


def LargeurParoi(shapes: dict) -> float:
    """Largeur (épaisseur) de la paroi en px."""
    for k, s in shapes.items():
        if NomShape(k) == "PAROI":
            try:
                return float(s.Width)
            except Exception:
                return 0.0
    return 0.0


def DroiteParoi(shapes: dict) -> float:
    """Abscisse du bord droit de la paroi."""
    for k, s in shapes.items():
        if NomShape(k) == "PAROI":
            try:
                return float(s.Left) + float(s.Width)
            except Exception:
                return 0.0
    return 0.0


def MilieuParoi(shapes: dict):
    for k, s in shapes.items():
        if NomShape(k) == "PAROI":
            return s.Left + (s.Width / 2)
    return GaucheDessin() + (LargeurDessin() / 2)


# =========================
# EXCAV / FOUILLE / TIRANTS
# =========================

def DerExcav():
    df = pd.read_excel(classeurCoupe(), sheet_name="Excav", header=None, engine="openpyxl")
    i = 0
    while i < len(df) and pd.notna(df.iat[i, 0]) and df.iat[i, 0] != "":
        i += 1
    if i == 0:
        return 0
    return df.iat[i - 1, 1]


def marge():
    return 0.8


def Transptir(AltTir, Forcing):
    transp = 0.5
    forcing = "" if Forcing is None else str(Forcing).strip().lower()
    if AltTir < DerExcav() + marge() or forcing in ("suppr", "non", "absent"):
        if forcing not in ("present", "oui"):
            transp = 0.9
    return transp


# =========================
# TOPO
# =========================

def Xref():
    return LignePara(3, 2)


def Yref():
    return LignePara(4, 2)


def AngleXn():
    return LignePara(1, 2)


def angle():
    a = AngleXn()
    if a is None or str(a).strip() == "":
        return 0
    return a


def DiamCible():
    return 7


def DeltaJourTopo(L):
    return int(L) - 7


def LimiteReprmm():
    return 1000
