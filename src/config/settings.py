from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Excel
EXCEL_DATA_DIR = BASE_DIR / "data" / "excel_data"
COMMON_DATA_DIR = BASE_DIR / "data" / "common_data"

TRAME_FILE = COMMON_DATA_DIR / "Charges sur Trame.xlsx"

