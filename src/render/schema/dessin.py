from src.render.schema import utils_schema as dessinfonctions
import src.io.excel_reader as liredansexcel


def _soil_color_rgba(idx: int, total: int, alpha: float = 0.38) -> str:
    """
    Couleur fixe par couche : dégradé bleu -> rouge.
    idx: 0..total-1 dans l'ordre de Geol
    """
    if total <= 1:
        t = 0.5
    else:
        t = idx / (total - 1)

    # Bleu -> Rouge (bien visibles)
    r0, g0, b0 = 40, 120, 220
    r1, g1, b1 = 220, 60, 60

    r = int(round(r0 + (r1 - r0) * t))
    g = int(round(g0 + (g1 - g0) * t))
    b = int(round(b0 + (b1 - b0) * t))

    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r},{g},{b},{a})"


def CouchesGeol():
    i = 3
    classeur = dessinfonctions.classeurCoupe()

    # 1) Collecter toutes les couches valides pour connaître le total
    layers = []
    while liredansexcel.LireExcel(classeur, "Geol", i, 1) != "FF":
        nom_cell = liredansexcel.LireExcel(classeur, "Geol", i, 1)

        if nom_cell != "":
            z_haut = liredansexcel.LireExcel(classeur, "Geol", i, 3)
            z_bas = liredansexcel.LireExcel(classeur, "Geol", i + 1, 3)

            if not (z_haut == 0 and z_bas == 0):
                layers.append((nom_cell, z_haut, z_bas))

        i += 2

    total_layers = len(layers)

    # 2) Créer / mettre à jour les shapes, et assigner une couleur
    for idx, (nom_cell, z_haut, z_bas) in enumerate(layers):
        nomSOL = dessinfonctions.NomShape(nom_cell)

        shape = dessinfonctions.Shapes.get(nomSOL)
        shape = dessinfonctions.CreerSolSiNecessaire(shape, nomSOL)

        y_haut = dessinfonctions.ZDessin(z_haut)
        y_bas = dessinfonctions.ZDessin(z_bas)

        shape.Top = min(y_haut, y_bas)
        shape.Height = abs(y_bas - y_haut)

        shape.Left = dessinfonctions.GaucheDessin()
        shape.Width = dessinfonctions.LargeurDessin()

        # ✅ COULEUR PAR COUCHE (bleu -> rouge)
        shape.fill = _soil_color_rgba(idx, total_layers, alpha=0.38)

        # ✅ couches sous tout
        shape.zIndex = 0

        dessinfonctions.Shapes[nomSOL] = shape

    return dessinfonctions.Shapes


def ParoiDessin():
    key = None
    for k in dessinfonctions.Shapes.keys():
        if str(k).lower() == "paroi":
            key = k
            break

    if key is None:
        shape = dessinfonctions.Shapes.get("paroi")
        shape = dessinfonctions.CreerSolSiNecessaire(shape, "paroi")
        key = "paroi"

    paroi = dessinfonctions.Shapes[key]

    largeur_paroi = 6
    centre = dessinfonctions.GaucheDessin() + dessinfonctions.LargeurDessin() / 2
    paroi.Width = largeur_paroi
    paroi.Left = centre - largeur_paroi / 2

    paroi.zIndex = 10
    paroi.fill = "rgba(150,150,150,0.9)"

    return paroi


def FouilleDessin(shapes):
    tops_sols = []
    for name, s in shapes.items():
        n = str(name).lower()
        if n not in ("paroi", "fouille"):
            tops_sols.append(s.Top)

    if not tops_sols:
        return shapes

    top_fouille = min(tops_sols)

    z_bas = dessinfonctions.DerExcav()
    y_bas = dessinfonctions.ZDessin(z_bas)

    height = y_bas - top_fouille
    if height <= 0:
        return shapes

    # créer shape fouille si absente
    key = dessinfonctions.NomShape("Fouille")
    fouille = dessinfonctions.Shapes.get(key)
    fouille = dessinfonctions.CreerSolSiNecessaire(fouille, "Fouille")
    shapes[key] = fouille

    fouille.Top = top_fouille
    fouille.Height = height

    gauche = dessinfonctions.GaucheParoi(shapes)
    fouille.Left = gauche
    fouille.Width = dessinfonctions.DroiteDessin() - gauche

    fouille.fill = "white"
    fouille.border = "1px solid black"
    fouille.zIndex = 5

    return shapes


def FondDeFouilleDessin(shapes):
    return shapes
