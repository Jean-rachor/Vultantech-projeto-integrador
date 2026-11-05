from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vulcantech.config import DATABASE_PATH, IMAGE_DIR
from vulcantech.database import init_db, register_source_images


if __name__ == "__main__":
    init_db()
    total = register_source_images(DATABASE_PATH, IMAGE_DIR)
    print(f"SQLite: {DATABASE_PATH} | imagens-fonte novas: {total}")
