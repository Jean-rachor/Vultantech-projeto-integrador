import sys
from pathlib import Path

import numpy as np
import pytest
import cv2

from vulcantech.database import init_db, list_inspections, reference_images, register_source_images, save_inspection
from vulcantech.decision import decision_record
from vulcantech.vision import inspect_image, reference_catalog

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Vulcantech_imagens_soldas"))
from gerar_variacoes import white_frame_bbox


def test_corrupt_image_is_rejected():
    with pytest.raises(ValueError):
        inspect_image(b"not-an-image")


def test_only_continuous_white_perimeter_is_cropped():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:2] = 255
    image[-2:] = 255
    image[:, :3] = 255
    image[:, -3:] = 255
    assert white_frame_bbox(image) == (3, 2, 27, 18)
    assert white_frame_bbox(np.zeros((20, 30, 3), dtype=np.uint8)) == (0, 0, 30, 20)


def test_ordinary_image_is_experimental_and_unclassified():
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, 32:64] = (255, 255, 255)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    result = inspect_image(encoded.tobytes())
    assert result["classification"] == "NAO_CLASSIFICADA"
    assert result["haz_visual_experimental"]["temperature_inference"] is False
    assert result["thermal_analysis"]["status"] == "INDISPONIVEL"


def test_reference_catalog_reads_original_blobs_and_excludes_derivatives(tmp_path):
    from vulcantech.config import IMAGE_DIR

    db_path = tmp_path / "references.sqlite3"
    inserted = register_source_images(db_path, IMAGE_DIR)
    refs = reference_images(db_path)
    assert inserted == 62
    assert len(refs) == 20
    assert all(item["source_type"] == "projeto" and item["image_blob"] for item in refs)
    assert all(item["label"] == "NAO_CLASSIFICADA" and item["label_source"] is None for item in refs)
    assert reference_catalog(db_path)["classification_ready"] is False


def test_inspection_compares_against_db_references_without_inventing_a_class(tmp_path):
    from vulcantech.config import IMAGE_DIR

    db_path = tmp_path / "references.sqlite3"
    register_source_images(db_path, IMAGE_DIR)
    result = inspect_image((IMAGE_DIR / "solda_01.jpg").read_bytes(), reference_path=db_path)
    comparison = result["reference_comparison"]
    assert comparison["status"] == "COMPARACAO_EXPERIMENTAL"
    assert comparison["references_available"] == 20
    assert comparison["references_used"] == 20
    assert comparison["top_matches"][0]["filename"] == "solda_01.jpg"
    assert result["classification"] == "NAO_CLASSIFICADA"


def test_empty_reference_database_reports_unavailable_without_classification(tmp_path):
    db_path = tmp_path / "empty.sqlite3"
    init_db(db_path)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    result = inspect_image(encoded.tobytes(), reference_path=db_path)
    assert result["reference_comparison"]["status"] == "INDISPONIVEL"
    assert result["reference_comparison"]["references_used"] == 0
    assert result["classification"] == "NAO_CLASSIFICADA"


def test_history_persists_pending_action_without_uploaded_image(tmp_path):
    db_path = tmp_path / "project.sqlite3"
    init_db(db_path)
    record = decision_record({"status": "NAO_CLASSIFICADA"})
    save_inspection(record, input_kind="optical", image_sha256="abc",
                    diagnosis={"status": "NAO_CLASSIFICADA"}, metrics={"edge_pixel_fraction": 0.1},
                    limitations=["Sem rótulo"], path=db_path)
    rows = list_inspections(path=db_path)
    assert len(rows) == 1
    assert rows[0]["decision"]["action"] is None
    assert rows[0]["image_sha256"] == "abc"
    assert rows[0]["decision"]["validation_state"] == "PENDENTE_APROVACAO"
