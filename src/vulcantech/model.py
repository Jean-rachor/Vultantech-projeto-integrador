import math
from functools import lru_cache

from .config import MODEL_PATH

CLASS_NAMES = {0: "Operação Normal", 1: "Desgaste Severo de Ferramenta / Fuso",
               2: "Falha de Refrigeração / Superaquecimento",
               3: "Desalinhamento / Trinca Mecânica"}


@lru_cache(maxsize=1)
def load_pipeline():
    import joblib
    return joblib.load(MODEL_PATH)


def model_status() -> dict:
    if not MODEL_PATH.is_file():
        return {"available": False, "reason": "Artefato do modelo não encontrado."}
    try:
        pipeline = load_pipeline()
        selected = pipeline.get("modelo_selecionado") or "Random Forest"
        return {"available": bool(pipeline.get("modelos")), "models": sorted(pipeline.get("modelos", {})),
                "selected_model": selected, "selection_metric": "recall macro temporal"}
    except Exception:
        return {"available": False, "reason": "Artefato incompatível com as dependências instaladas."}


def predict_one(values: dict[str, float]) -> dict:
    if not values or any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Informe valores numéricos finitos para as features do modelo.")
    pipeline = load_pipeline()
    features = pipeline.get("feature_names", [])
    missing, extra = sorted(set(features) - values.keys()), sorted(values.keys() - set(features))
    if missing or extra:
        raise ValueError(f"Features divergentes. Ausentes: {missing}; extras: {extra}.")
    import numpy as np
    scaler = pipeline["scaler"]
    selected = pipeline.get("modelo_selecionado") or "Random Forest"
    model = pipeline["modelos"].get(selected)
    if model is None:
        raise RuntimeError("O artefato não contém o modelo selecionado pelo critério de recall documentado.")
    matrix = np.asarray([[values[name] for name in features]], dtype=float)
    probabilities = model.predict_proba(scaler.transform(matrix))[0]
    class_id = int(model.classes_[int(np.argmax(probabilities))])
    return {"model": selected, "model_selection":
            pipeline.get("criterio_selecao", "Maior recall macro temporal, conforme critério do código existente."),
            "diagnosis": {"class_id": class_id, "label": CLASS_NAMES.get(class_id, "Classe sem rótulo")},
            "probabilities": {CLASS_NAMES.get(int(cls), str(cls)): float(prob)
                              for cls, prob in zip(model.classes_, probabilities)},
            "validation_claim": "Resultados do relatório existente; esta inferência individual não é nova validação."}
