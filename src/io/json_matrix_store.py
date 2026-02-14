from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JsonWorkbook:
    """Represents a workbook stored as a folder of sheet JSON files."""
    folder: Path  # e.g. data/common_data/cst_runtime


def _sheet_file(folder: Path, sheet_name: str) -> Path:
    # We keep names as exported: "<SheetName>.json" with spaces preserved if you want.
    # Here we assume filenames match exactly sheet names + ".json".
    return folder / f"{sheet_name}.json"


def read_sheet_matrix(folder: Path, sheet_name: str) -> list[list[Any]]:
    p = _sheet_file(folder, sheet_name)
    if not p.exists():
        raise FileNotFoundError(f"JSON sheet introuvable: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "matrix" in data:
        return data["matrix"]
    raise ValueError(f"Format JSON invalide pour {p} (attendu dict avec clé 'matrix')")


def write_sheet_matrix(folder: Path, sheet_name: str, matrix: list[list[Any]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    p = _sheet_file(folder, sheet_name)
    tmp = p.with_suffix(p.suffix + ".tmp")
    payload = {"sheet": sheet_name, "matrix": matrix}
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def read_cell(folder: Path, sheet_name: str, row1: int, col1: int) -> Any:
    m = read_sheet_matrix(folder, sheet_name)
    r = row1 - 1
    c = col1 - 1
    if r < 0 or c < 0:
        return ""
    if r >= len(m):
        return ""
    if c >= len(m[r]):
        return ""
    v = m[r][c]
    return "" if v is None else v


def write_cell(folder: Path, sheet_name: str, row1: int, col1: int, value: Any) -> None:
    m = read_sheet_matrix(folder, sheet_name)
    r = row1 - 1
    c = col1 - 1
    if r < 0 or c < 0:
        return

    # extend rows
    while len(m) <= r:
        m.append([])

    # extend cols
    while len(m[r]) <= c:
        m[r].append(None)

    m[r][c] = value
    write_sheet_matrix(folder, sheet_name, m)
