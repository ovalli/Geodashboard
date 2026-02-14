from typing import Optional

from src.render.schema import utils_schema
from src.render.schema import utils_schema as dessinfonctions

from src.render.schema.dessin import (
    CouchesGeol,
    ParoiDessin,
    FouilleDessin,
    FondDeFouilleDessin,
)

from src.render.schema.dessinInclino import inclinoUnique
from src.render.schema.dessinTirants import tirants
from src.render.schema.dessinPlanchers import planchers
from src.render.schema.dessinTopo import Topo


def _pick_sheet(classeur: str, *candidates: str) -> Optional[str]:
    """Retourne le premier nom de feuille existant parmi les candidats."""
    for s in candidates:
        if dessinfonctions.sheetExist(classeur, s):
            return s
    return None


def _draw_inclino_pair(
    *,
    shapes: dict,
    classeur: str,
    sheet_name: str,
    base_shape_name: str,
    couleur: str,
    nom_reel: str,
):
    """Dessine une paire inclino (plein + pointillé) depuis une feuille."""
    if not dessinfonctions.sheetExist(classeur, sheet_name):
        return shapes

    # Courbe principale : dernière colonne (plein)
    shapes = inclinoUnique(
        Inclino=sheet_name,
        nomReel=nom_reel,
        nomShape=f"{base_shape_name}_1",
        Recul=0,
        Zincl=0,
        Col=0,
        StyleDeCourbe="solid",
        epaisseur=2.0,
        transparence=0.0,
        facteurSpecial=1,
        shapes=shapes,
        couleur=couleur,
    )

    # Courbe secondaire : avant-dernière colonne (pointillé)
    shapes = inclinoUnique(
        Inclino=sheet_name,
        nomReel=f"{nom_reel} (col-1)",
        nomShape=f"{base_shape_name}_2",
        Recul=0,
        Zincl=0,
        Col=-1,
        StyleDeCourbe="dashed",
        epaisseur=2.0,
        transparence=0.0,
        facteurSpecial=1,
        shapes=shapes,
        couleur=couleur,
    )

    return shapes


def build_shapes():
    # IMPORTANT : reset pour éviter de garder des shapes d’un run précédent
    dessinfonctions.Shapes.clear()

    # =========================
    # BASE DU SCHÉMA
    # =========================
    shapes = CouchesGeol()
    ParoiDessin()
    shapes = FouilleDessin(shapes)
    shapes = FondDeFouilleDessin(shapes)

    classeur = dessinfonctions.classeurCoupe()

    # =========================
    # INCLINO AVAL — ROUGE
    # Feuille : "Inclino"
    # =========================
    shapes = _draw_inclino_pair(
        shapes=shapes,
        classeur=classeur,
        sheet_name="Inclino",
        base_shape_name="INCLINO",
        couleur="red",
        nom_reel="Inclinomètre",
    )

    # =========================
    # INCLINO AMONT — BLEU CLAIR
    # Feuille : "Inclinoamont"
    # =========================
    shapes = _draw_inclino_pair(
        shapes=shapes,
        classeur=classeur,
        sheet_name="Inclinoamont",
        base_shape_name="INCLINO_AMONT",
        couleur="rgb(80,150,255)",
        nom_reel="Inclinomètre amont",
    )

    # =========================
    # INCLINO AMONT 2 — VERT FONCÉ
    # Feuille : "Inclinoamont2"
    # =========================
    shapes = _draw_inclino_pair(
        shapes=shapes,
        classeur=classeur,
        sheet_name="Inclinoamont2",
        base_shape_name="INCLINO_AMONT2",
        couleur="rgb(20,110,60)",   # vert foncé lisible
        nom_reel="Inclinomètre amont 2",
    )

    # =========================
    # TIRANTS / PLANCHERS / TOPO
    # =========================
    shapes = tirants(shapes, sheet_name="Tirants 1")
    shapes = planchers(shapes, sheet_name="Planchers")

    if dessinfonctions.sheetExist(classeur, "Topo"):
        shapes = Topo(shapes, workbook_path=classeur)

    return shapes
