import importlib.machinery
import importlib.util

from fastapi.testclient import TestClient

from vulcantech import app as app_module
from vulcantech.config import IMAGE_DIR
from vulcantech.database import list_inspections, save_inspection
from vulcantech.model import load_pipeline


def _real_model_features():
    loader = importlib.machinery.SourceFileLoader("ml_for_api_test", "src/machine.learning.py")
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    row = module.carregar_dados().iloc[0]
    return {name: float(row[name]) for name in load_pipeline()["feature_names"]}


def test_real_prediction_and_image_routes_keep_pending_decision(monkeypatch, tmp_path):
    db_path = tmp_path / "api.sqlite3"

    def save(record, **kwargs):
        save_inspection(record, **kwargs, path=db_path)

    monkeypatch.setattr(app_module, "save_inspection", save)
    monkeypatch.setattr(app_module, "list_inspections",
                        lambda limit, offset: list_inspections(limit, offset, path=db_path))
    with TestClient(app_module.app) as client:
        assert client.get("/").status_code == 200
        health = client.get("/health").json()
        assert health["predictive_model"]["available"] is True
        reference_status = client.get("/references")
        assert reference_status.status_code == 200
        assert reference_status.json()["images_loaded"] == 20
        assert reference_status.json()["classification_ready"] is False
        prediction = client.post("/predict", json={"features": _real_model_features()})
        assert prediction.status_code == 200
        result = prediction.json()
        assert abs(sum(result["probabilities"].values()) - 1) < 1e-6
        assert result["decision"]["action"] is None
        assert result["decision"]["validation_state"] == "PENDENTE_APROVACAO"
        upload = client.post("/inspect", files={"file": ("weld.jpg", (IMAGE_DIR / "solda_01.jpg").read_bytes(), "image/jpeg")})
        assert upload.status_code == 200
        visual = upload.json()
        assert visual["diagnosis"]["status"] == "NAO_CLASSIFICADA"
        assert visual["metrics"]["thermal_analysis"]["status"] == "INDISPONIVEL"
        assert visual["reference_comparison"]["references_used"] == 20
        assert visual["reference_comparison"]["status"] == "COMPARACAO_EXPERIMENTAL"
        thermal = client.post("/inspect?input_type=thermal", files={"file": ("weld.jpg", (IMAGE_DIR / "solda_01.jpg").read_bytes(), "image/jpeg")})
        assert thermal.json()["metrics"]["thermal_analysis"]["status"] == "INDISPONIVEL"
        batch = client.post("/predict-batch", json={"rows": [_real_model_features(), _real_model_features()]})
        assert batch.status_code == 200 and len(batch.json()["results"]) == 2
        assert all(item["decision"]["action"] is None for item in batch.json()["results"])
        invalid = client.post("/predict", json={"features": {}})
        assert invalid.status_code == 422
        assert invalid.json()["decision"]["validation_state"] == "PENDENTE_DADOS_INVALIDOS"
        assert invalid.json()["decision"]["action"] is None
        malformed = client.post("/predict", json={"features": {"bad": "not-a-number"}})
        assert malformed.status_code == 422
        assert malformed.json()["decision"]["validation_state"] == "PENDENTE_DADOS_INVALIDOS"
        invalid_image = client.post("/inspect", files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")})
        assert invalid_image.status_code == 422
        assert invalid_image.json()["decision"]["action"] is None
        mislabeled_jpeg = client.post("/inspect", files={"file": ("image.png", (IMAGE_DIR / "solda_01.jpg").read_bytes(), "image/png")})
        assert mislabeled_jpeg.status_code == 415
        assert mislabeled_jpeg.json()["decision"]["validation_state"] == "PENDENTE_DADOS_INVALIDOS"
        invalid_batch = client.post("/predict-batch", json={"rows": [{}]})
        assert invalid_batch.status_code == 200
        assert invalid_batch.json()["results"][0]["decision"]["validation_state"] == "PENDENTE_DADOS_INVALIDOS"
        assert len(client.get("/inspections").json()["results"]) == 10
