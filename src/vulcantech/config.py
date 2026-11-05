import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "Vulcantech_imagens_soldas"
VARIANT_DIR = IMAGE_DIR / "derivadas"
MODEL_PATH = Path(os.getenv("VULCANTECH_MODEL", ROOT / "Docs/Entregaveis/ML/vulcantech_pipeline.joblib"))
DATABASE_PATH = Path(os.getenv("VULCANTECH_DB", ROOT / "vulcantech.sqlite3"))
MAX_UPLOAD_BYTES = int(os.getenv("VULCANTECH_MAX_UPLOAD_BYTES", 12 * 1024 * 1024))
RULE_VERSION = "pendente-de-aprovacao"
