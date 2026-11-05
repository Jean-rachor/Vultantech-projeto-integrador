import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import DATABASE_PATH, IMAGE_DIR


def connect(path: Path = DATABASE_PATH) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


@contextmanager
def session(path: Path = DATABASE_PATH):
    db = connect(path)
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db(path: Path = DATABASE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with session(path) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS image_assets (
            id INTEGER PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
            source_path TEXT NOT NULL, source_type TEXT NOT NULL, label TEXT NOT NULL DEFAULT 'NAO_CLASSIFICADA',
            label_source TEXT, metadata_json TEXT NOT NULL, image_blob BLOB NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS inspections (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, input_kind TEXT NOT NULL,
            image_sha256 TEXT, diagnosis_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
            limitations_json TEXT NOT NULL, decision_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_inspections_created ON inspections(created_at DESC);
        CREATE TABLE IF NOT EXISTS decision_audit (
            id INTEGER PRIMARY KEY, inspection_id TEXT NOT NULL REFERENCES inspections(id),
            criterion_version TEXT NOT NULL, validation_state TEXT NOT NULL,
            action TEXT, rationale TEXT NOT NULL, created_at TEXT NOT NULL
        );
        """)


def register_source_images(path: Path = DATABASE_PATH, image_dir: Path = IMAGE_DIR) -> int:
    init_db(path)
    count = 0
    derivative_data = {}
    for manifest_file in image_dir.glob("*/manifest.json"):
        try:
            for item in json.loads(manifest_file.read_text(encoding="utf-8")):
                derivative_data[item["file"]] = item
        except (OSError, ValueError, KeyError, TypeError):
            continue
    external_data = {}
    external_manifest = image_dir / "externas" / "manifest.json"
    if external_manifest.exists():
        parsed = json.loads(external_manifest.read_text(encoding="utf-8"))
        external_data = {item["filename"]: item for item in parsed if "filename" in item}
    with session(path) as db:
        for file in sorted(image_dir.rglob("*")):
            if (not file.is_file() or file.name == "manifest.json" or
                file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}):
                continue
            data = file.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            relative = file.relative_to(image_dir)
            derivative = derivative_data.get(file.name)
            source_type = "derivada" if derivative else "externa" if relative.parts[0] == "externas" else "projeto"
            metadata = {"classification": "NAO_CLASSIFICADA", "classification_source": None,
                        "image_role": "referencia_solda", "source_type": source_type,
                        "provenance_status": {"projeto": "origem original não documentada",
                                              "derivada": "derivada de imagem do projeto; transformação no manifesto",
                                              "externa": "fonte e licença descritas no manifesto"}[source_type],
                        "exif_available": False}
            if derivative:
                metadata.update(derivative)
            if source_type == "externa":
                metadata.update(external_data.get(file.name, {}))
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS
                with Image.open(file) as im:
                    metadata.update(width=im.width, height=im.height, format=im.format,
                                    mode=im.mode, exif_available=bool(im.getexif()))
                    metadata["exif"] = {TAGS.get(k, str(k)): str(v) for k, v in im.getexif().items()}
            except ImportError:
                metadata["metadata_note"] = "Pillow ausente; dimensões não extraídas."
            db.execute("""INSERT OR IGNORE INTO image_assets
                (filename,sha256,source_path,source_type,label,label_source,metadata_json,image_blob,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (file.name, digest, str(file.relative_to(image_dir.parent)), source_type, "NAO_CLASSIFICADA",
                 None, json.dumps(metadata, ensure_ascii=False), data, datetime.now(timezone.utc).isoformat()))
            inserted = db.execute("SELECT changes()").fetchone()[0]
            count += inserted
            if not inserted:
                db.execute("UPDATE image_assets SET metadata_json=? WHERE sha256=?",
                           (json.dumps(metadata, ensure_ascii=False), digest))
    return count


def reference_images(path: Path = DATABASE_PATH, limit: int = 200) -> list[dict]:
    """Return original project image BLOBs for experimental visual comparison.

    Derivatives and external illustrations are deliberately excluded: the
    former duplicate source images and the latter do not document the same
    capture/process conditions. Labels and metadata are returned as stored.
    """
    if not path.exists():
        return []
    with session(path) as db:
        rows = db.execute(
            """SELECT id, filename, sha256, source_path, source_type, label,
                      label_source, metadata_json, image_blob
               FROM image_assets
               WHERE source_type=?
               ORDER BY id ASC
               LIMIT ?""",
            ("projeto", max(1, min(int(limit), 500))),
        ).fetchall()
    images = []
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"])
        except (TypeError, ValueError):
            metadata = {}
        blob = row["image_blob"]
        if not isinstance(blob, bytes) or not blob:
            continue
        images.append({
            "id": row["id"],
            "filename": row["filename"],
            "sha256": row["sha256"],
            "source_path": row["source_path"],
            "source_type": row["source_type"],
            "label": row["label"],
            "label_source": row["label_source"],
            "metadata": metadata,
            "image_blob": blob,
        })
    return images


def database_status(path: Path = DATABASE_PATH) -> dict:
    try:
        if not path.exists():
            return {"available": False, "image_assets": 0, "project_references": 0,
                    "uploaded_images": 0, "labelled_uploads": 0, "labelled_vision_images": 0}
        with session(path) as db:
            db.execute("SELECT 1").fetchone()
            image_assets = db.execute("SELECT COUNT(*) FROM image_assets").fetchone()[0]
            project_references = db.execute(
                "SELECT COUNT(*) FROM image_assets WHERE source_type=?", ("projeto",)
            ).fetchone()[0]
            uploaded_images = db.execute(
                "SELECT COUNT(*) FROM image_assets WHERE source_type=?", ("upload",)
            ).fetchone()[0]
            labelled_uploads = db.execute(
                "SELECT COUNT(*) FROM image_assets WHERE source_type=? AND label<>? AND label_source IS NOT NULL",
                ("upload", "NAO_CLASSIFICADA"),
            ).fetchone()[0]
            labelled_vision_images = db.execute(
                "SELECT COUNT(*) FROM image_assets WHERE source_type IN ('projeto','upload') AND label<>? AND label_source IS NOT NULL",
                ("NAO_CLASSIFICADA",),
            ).fetchone()[0]
        return {"available": True, "image_assets": image_assets, "project_references": project_references,
                "uploaded_images": uploaded_images, "labelled_uploads": labelled_uploads,
                "labelled_vision_images": labelled_vision_images}
    except (OSError, sqlite3.Error):
        return {"available": False, "image_assets": 0, "project_references": 0,
                "uploaded_images": 0, "labelled_uploads": 0, "labelled_vision_images": 0}


def register_uploaded_image(data: bytes, filename: str, metadata: dict,
                            path: Path = DATABASE_PATH) -> dict:
    """Persist a validated user image in SQLite without treating it as labelled training data."""
    init_db(path)
    digest = hashlib.sha256(data).hexdigest()
    safe_name = Path(filename.replace("\\", "/")).name[:255] or "imagem-enviada"
    stored_metadata = {**metadata, "image_role": "inspecao_enviada",
                       "classification": "NAO_CLASSIFICADA", "classification_source": None,
                       "source_type": "upload", "label_status": "aguardando_rotulo_confiavel"}
    with session(path) as db:
        db.execute("""INSERT OR IGNORE INTO image_assets
            (filename,sha256,source_path,source_type,label,label_source,metadata_json,image_blob,created_at)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (safe_name, digest, f"uploads/{digest}", "upload", "NAO_CLASSIFICADA", None,
             json.dumps(stored_metadata, ensure_ascii=False), data, datetime.now(timezone.utc).isoformat()))
        row = db.execute("SELECT id, filename, sha256, source_type, label, created_at FROM image_assets WHERE sha256=?",
                         (digest,)).fetchone()
    return dict(row)


def label_vision_image(image_id: int, label: str, label_source: str,
                       path: Path = DATABASE_PATH) -> dict | None:
    """Store a human-confirmed class for an original/uploaded image."""
    allowed = {"NORMAL", "POROSIDADE", "TRINCA", "DESCONTINUIDADE"}
    if label not in allowed:
        raise ValueError("Classe visual não permitida.")
    if not path.exists():
        return None
    with session(path) as db:
        row = db.execute("SELECT source_type, metadata_json FROM image_assets WHERE id=?", (image_id,)).fetchone()
        if row is None or row["source_type"] not in {"projeto", "upload"}:
            return None
        try:
            metadata = json.loads(row["metadata_json"])
        except (TypeError, ValueError):
            metadata = {}
        metadata.update(classification=label, classification_source=label_source,
                        label_status="rotulo_humano_registrado",
                        label_confirmed_at=datetime.now(timezone.utc).isoformat())
        db.execute("UPDATE image_assets SET label=?, label_source=?, metadata_json=? WHERE id=?",
                   (label, label_source.strip(), json.dumps(metadata, ensure_ascii=False), image_id))
    return {"id": image_id, "label": label, "label_source": label_source.strip(),
            "label_status": "rotulo_humano_registrado"}


def save_inspection(record: dict, *, input_kind: str, image_sha256: str | None,
                    diagnosis: dict, metrics: dict, limitations: list[str], path: Path = DATABASE_PATH) -> None:
    init_db(path)
    with session(path) as db:
        db.execute("""INSERT INTO inspections VALUES (?,?,?,?,?,?,?,?)""",
                   (record["inspection_id"], record["created_at"], input_kind, image_sha256,
                    json.dumps(diagnosis, ensure_ascii=False), json.dumps(metrics, ensure_ascii=False),
                    json.dumps(limitations, ensure_ascii=False), json.dumps(record, ensure_ascii=False)))
        db.execute("""INSERT INTO decision_audit
                   (inspection_id,criterion_version,validation_state,action,rationale,created_at)
                   VALUES (?,?,?,?,?,?)""",
                   (record["inspection_id"], record["criterion_version"], record["validation_state"],
                    record["action"], record["rationale"], record["created_at"]))


def list_inspections(limit: int = 50, offset: int = 0, path: Path = DATABASE_PATH) -> list[dict]:
    if not path.exists():
        return []
    with session(path) as db:
        rows = db.execute("SELECT * FROM inspections ORDER BY created_at DESC LIMIT ? OFFSET ?",
                           (limit, offset)).fetchall()
    return [{**dict(row), **{key: json.loads(row[f"{key}_json"]) for key in
             ("diagnosis", "metrics", "limitations", "decision")}} for row in rows]
