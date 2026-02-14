from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil


@dataclass(frozen=True)
class ImportTarget:
    label: str
    dest_path: Path
    backup_dir: Path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_mesures_completes_xlsx(common_data_dir: Path) -> Path | None:
    """
    Cherche le fichier 'Mesures Completes' (accents tolérés) dans data/common_data.
    Retourne le plus récent si plusieurs.
    """
    if not common_data_dir.exists():
        return None

    needles = [
        "mesures completes",
        "mesures complètes",
        "mesures complete",
        "mesures compl",
    ]

    candidates: list[Path] = []
    for p in common_data_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".xlsx":
            stem = p.stem.lower()
            if any(n in stem for n in needles):
                candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def find_inclino_xlsx(common_data_dir: Path) -> Path | None:
    """
    Cherche un fichier inclino dans data/common_data.
    Retourne le plus récent si plusieurs.
    """
    if not common_data_dir.exists():
        return None

    needles = ["inclino", "inclinometr", "inclinomètr"]

    candidates: list[Path] = []
    for p in common_data_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".xlsx":
            stem = p.stem.lower()
            if any(n in stem for n in needles):
                candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def build_topo_target(common_data_dir: Path) -> ImportTarget:
    """
    Destination = le fichier Mesures Completes actuellement utilisé (détecté).
    Si pas trouvé => crée Mesures Completes.xlsx.
    """
    common_data_dir.mkdir(parents=True, exist_ok=True)
    current = find_mesures_completes_xlsx(common_data_dir)
    dest = current if current is not None else (common_data_dir / "Mesures Completes.xlsx")
    backup_dir = common_data_dir / "_import_backups"
    return ImportTarget(label="Topographie", dest_path=dest, backup_dir=backup_dir)


def build_inclino_target(common_data_dir: Path) -> ImportTarget:
    """
    Destination = le fichier inclino actuellement utilisé (détecté).
    Si pas trouvé => crée Inclinometrie.xlsx.
    """
    common_data_dir.mkdir(parents=True, exist_ok=True)
    current = find_inclino_xlsx(common_data_dir)
    dest = current if current is not None else (common_data_dir / "Inclinometrie.xlsx")
    backup_dir = common_data_dir / "_import_backups"
    return ImportTarget(label="Inclinométrie", dest_path=dest, backup_dir=backup_dir)


def replace_file_bytes(dest_path: Path, content: bytes, backup_dir: Path) -> None:
    """
    Remplace dest_path par le contenu (bytes).
    Sauvegarde l'ancien fichier (s'il existe) dans backup_dir avec horodatage.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        backup_name = f"{dest_path.stem}__backup__{_timestamp()}{dest_path.suffix}"
        shutil.copy2(dest_path, backup_dir / backup_name)

    # écriture atomique "best effort" (temp puis replace)
    tmp = dest_path.with_suffix(dest_path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(dest_path)
