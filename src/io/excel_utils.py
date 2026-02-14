from pathlib import Path
import pandas as pd


def list_xlsx_in_folder(folder: Path):
    return sorted(
        f.name for f in folder.iterdir()
        if f.is_file() and f.name.lower().endswith(".xlsx") and not f.name.startswith("~$")
    )


def audit_sheets(abs_xlsx_path: str) -> dict:
    try:
        xls = pd.ExcelFile(abs_xlsx_path, engine="openpyxl")
        sheets = set(xls.sheet_names)
    except Exception:
        return {"_error": True, "sheets": set()}

    return {
        "_error": False,
        "sheets": sheets,
        "inclino": any(s.lower() in ("inclino", "inclinoamont", "inclinoamont2") for s in sheets),
        "topo": "Topo" in sheets,
        "excav": "Excav" in sheets,
        "pz": any(s.lower().startswith("pz") for s in sheets),
    }
