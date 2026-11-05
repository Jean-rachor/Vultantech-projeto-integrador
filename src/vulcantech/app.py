import hashlib
import csv
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import DATABASE_PATH, IMAGE_DIR, MAX_UPLOAD_BYTES, ROOT
from .database import (database_status, init_db, list_inspections,
                       label_vision_image, register_source_images, register_uploaded_image,
                       save_inspection)
from .decision import decision_record
from .model import model_status, predict_one
from .schemas import BatchRequest, ImageLabelRequest, PredictRequest
from .vision import inspect_image, reference_catalog, validate_image


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(init_db)
    status = await run_in_threadpool(database_status)
    if status["available"] and status["project_references"] == 0:
        await run_in_threadpool(register_source_images)
    yield


app = FastAPI(title="VulcanTech Inspection API", version="0.1.0", lifespan=lifespan)
WEB = Path(__file__).resolve().parents[1] / "web" / "index.html"
SENSOR_EXAMPLE = Path(__file__).resolve().parents[2] / "Docs" / "Exemplos" / "predicao_sensores_base.json"
SCADA_CSV = ROOT / "Data" / "dados_sensores_vulcantech.csv"


def scada_rows() -> list[dict]:
    """Read-only historical telemetry used by the clearly marked SCADA demo."""
    if not SCADA_CSV.is_file():
        raise HTTPException(503, "Arquivo histórico de telemetria não encontrado.")
    with SCADA_CSV.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def number(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


SCADA_FIELDS = {
    "temperatura": "temperatura_fuso_c", "vibracao": "vibracao_triaxial_rms_mms",
    "torque": "torque_fuso_nm", "refrigerante": "vazao_refrigerante_lmin",
    "corrente": "corrente_estator_amp", "pressao": "pressao_hidraulica_bar",
    "ruido": "nivel_ruido_db",
}


def record_pending(input_kind: str, reason: str, image_sha256: str | None = None) -> dict:
    diagnosis = {"status": "NAO_CLASSIFICADA", "reason": reason}
    record = decision_record(diagnosis, {"input_validation": "invalid"}, valid=False)
    save_inspection(record, input_kind=input_kind, image_sha256=image_sha256, diagnosis=diagnosis,
                    metrics={"input_validation": "invalid"}, limitations=[reason])
    return record


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    if request.url.path not in {"/inspect", "/predict", "/predict-batch"}:
        return JSONResponse(status_code=422, content={"detail": "Entrada inválida."})
    record = await run_in_threadpool(record_pending, request.url.path.lstrip("/"),
                                     "Entrada ausente ou inválida; diagnóstico e ação permanecem pendentes.")
    return JSONResponse(status_code=422, content={"detail": "Entrada ausente ou inválida.",
                                                   "inspection_id": record["inspection_id"],
                                                   "decision": record})


@app.exception_handler(HTTPException)
async def structured_http_error(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "decision" in exc.detail:
        payload = {**exc.detail, "detail": exc.detail.get("message", "Entrada inválida.")}
        return JSONResponse(status_code=exc.status_code, content=payload, headers=exc.headers)
    return await http_exception_handler(request, exc)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB)


@app.get("/examples/sensor-prediction.json", include_in_schema=False)
def sensor_prediction_example():
    if not SENSOR_EXAMPLE.is_file():
        raise HTTPException(404, "JSON de exemplo não encontrado.")
    return FileResponse(SENSOR_EXAMPLE, media_type="application/json",
                        filename="predicao_sensores_base.json")


@app.get("/health")
def health():
    db = database_status()
    return {"status": "ok", "service": "vulcantech", "predictive_model": model_status(),
            "database": {**db, "path_configured": DATABASE_PATH.name},
            "visual_references": reference_catalog(),
            "image_classifier": {"available": False, "reason":
                "Não há modelo visual de solda treinado.",
                "uploaded_images": db.get("uploaded_images", 0),
                "labelled_uploads": db.get("labelled_uploads", 0),
                "labelled_vision_images": db.get("labelled_vision_images", 0)},
            "industrial_decision_rules": "PENDENTE_APROVACAO",
            "thermal_analysis": "INDISPONIVEL_SEM_CALIBRACAO"}


@app.get("/model-info")
def model_info():
    status = model_status()
    if not status["available"]:
        return status
    from .model import load_pipeline
    pipeline = load_pipeline()
    return {**status, "selected_model": status["selected_model"],
            "selection_criterion": "Maior recall macro temporal no relatório existente.",
            "features": pipeline.get("feature_names", [])}


@app.get("/references")
def references():
    """Report reference availability and provenance without exposing image BLOBs."""
    return reference_catalog()


@app.get("/vision-model-info")
def vision_model_info():
    db = database_status()
    return {"available": False, "task": "classificação visual de solda", "model": None,
            "uploaded_images": db.get("uploaded_images", 0),
            "labelled_uploads": db.get("labelled_uploads", 0),
            "labelled_vision_images": db.get("labelled_vision_images", 0),
            "reason": "As métricas OpenCV e ORB são exploratórias e não classificam defeitos. Os rótulos humanos podem ser registrados para preparar um conjunto de treino.",
            "classes_needed": ["normal", "porosidade", "trinca", "descontinuidade"],
            "next_step": "Rotular imagens originais com classe confirmada e fonte; então treinar e validar um modelo visual independente."}


@app.put("/images/{image_id}/label")
def label_image(image_id: int, request: ImageLabelRequest):
    result = label_vision_image(image_id, request.label, request.label_source)
    if result is None:
        raise HTTPException(404, "Imagem original ou upload não encontrado para rotulagem.")
    return {"success": True, "image": result,
            "message": "Rótulo humano salvo. Um novo modelo ainda precisa ser treinado e validado."}


@app.get("/assets/weld-sample", include_in_schema=False)
def weld_sample():
    sample = IMAGE_DIR / "solda_01.jpg"
    if not sample.is_file():
        raise HTTPException(404, "Imagem de referência não disponível.")
    return FileResponse(sample, media_type="image/jpeg")


@app.get("/scada/overview")
def scada_overview():
    rows = scada_rows()
    latest: dict[str, dict] = {}
    for row in rows:
        latest[row.get("id_celula", "Sem identificação")] = row
    cells = []
    for cell_id, row in sorted(latest.items()):
        fault = int(number(row.get("tipo_falha")) or 0)
        cells.append({"id": cell_id, "status": "CRÍTICO" if fault >= 2 else "ATENÇÃO" if fault == 1 else "NORMAL",
                      "fault_class": fault, "timestamp": row.get("timestamp"),
                      "alloy": row.get("liga_metalica"),
                      "values": {key: number(row.get(column)) for key, column in SCADA_FIELDS.items()}})
    return {"mode": "SIMULACAO", "source": SCADA_CSV.name, "records": len(rows), "cells": cells}


@app.get("/scada/telemetry")
def scada_telemetry(cell: str | None = Query(None), limit: int = Query(72, ge=1, le=500)):
    rows = scada_rows()
    if cell:
        rows = [row for row in rows if row.get("id_celula") == cell]
    rows = rows[-limit:]
    return {"mode": "SIMULACAO", "cell": cell, "source": SCADA_CSV.name,
            "points": [{"timestamp": row.get("timestamp"), **{key: number(row.get(column)) for key, column in SCADA_FIELDS.items()},
                        "fault_class": int(number(row.get("tipo_falha")) or 0)} for row in rows]}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), input_type: Literal["optical", "thermal"] = Query("optical")):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        digest = hashlib.sha256(data).hexdigest() if data else None
        record = await run_in_threadpool(record_pending, "image", "Tipo de arquivo inválido; decisão pendente.", digest)
        raise HTTPException(415, {"message": "Envie uma imagem JPEG, PNG ou WebP.",
                                  "inspection_id": record["inspection_id"], "decision": record})
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data or len(data) > MAX_UPLOAD_BYTES:
        digest = hashlib.sha256(data).hexdigest() if data else None
        record = await run_in_threadpool(record_pending, "image", "Imagem vazia ou acima do limite; decisão pendente.", digest)
        raise HTTPException(413, {"message": "Imagem vazia ou acima do limite permitido.",
                                  "inspection_id": record["inspection_id"], "decision": record})
    digest = hashlib.sha256(data).hexdigest()
    limits = ["Imagem sem rótulo de referência confiável; métricas não confirmam defeito.",
              "Recomendação operacional pendente de regra aprovada."]
    try:
        uploaded_image = await run_in_threadpool(validate_image, data)
    except (ValueError, cv2.error) as exc:
        record = await run_in_threadpool(record_pending, "image", str(exc), digest)
        raise HTTPException(422, {"message": "A imagem não pôde ser validada.",
                                  "inspection_id": record["inspection_id"], "decision": record}) from exc
    actual_content_type = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(uploaded_image["format"])
    if file.content_type != "application/octet-stream" and file.content_type != actual_content_type:
        reason = "O tipo informado não corresponde ao formato real do arquivo."
        record = await run_in_threadpool(record_pending, "image", reason, digest)
        raise HTTPException(415, {"message": reason, "inspection_id": record["inspection_id"],
                                  "decision": record})
    if input_type == "thermal":
        result = {"status": "INDISPONIVEL", "reason":
                  "Imagem térmica não será interpretada sem calibração, escala térmica e parâmetros de captura."}
        diagnosis = {"status": "NAO_CLASSIFICADA", "source": "termografia indisponível"}
        comparison = {"status": "NAO_EXECUTADA", "references_available": 0, "references_used": 0,
                      "reason": "A modalidade termográfica permanece indisponível sem calibração e escala térmica."}
        metrics = {"input_image": uploaded_image, "reference_comparison": comparison,
                   "thermal_analysis": result}
        limits.append(result["reason"])
    else:
        try:
            metrics = await run_in_threadpool(inspect_image, data)
        except (ValueError, cv2.error) as exc:
            record = await run_in_threadpool(record_pending, "image", str(exc), digest)
            raise HTTPException(422, {"message": "A imagem não pôde ser analisada.",
                                      "inspection_id": record["inspection_id"], "decision": record}) from exc
        diagnosis = {"status": "NAO_CLASSIFICADA", "indications":
                     "Métricas geométricas e regiões visuais experimentais; sem confirmação de classe."}
        limits.append("HAZ em imagem comum é análise visual experimental e não mede temperatura.")
        comparison = metrics["reference_comparison"]
        if not comparison.get("references_available"):
            limits.append("Não havia imagens originais legíveis no banco para comparação visual.")
        else:
            limits.append("A comparação ORB é exploratória; as imagens não têm rótulos confiáveis nem captura equivalente validada.")
    decision = decision_record(diagnosis, metrics)
    stored_image = await run_in_threadpool(register_uploaded_image, data, file.filename or "imagem-enviada",
                                           metrics.get("input_image", metrics.get("image", {})))
    await run_in_threadpool(save_inspection, decision, input_kind=input_type, image_sha256=digest,
                            diagnosis=diagnosis, metrics=metrics, limitations=limits)
    return {"success": True, "inspection_id": decision["inspection_id"], "input_image": uploaded_image,
            "stored_image": {"id": stored_image["id"], "sha256": stored_image["sha256"],
                             "source_type": stored_image["source_type"], "label": stored_image["label"]},
            "diagnosis": diagnosis, "metrics": metrics, "reference_comparison": comparison,
            "limitations": limits, "decision": decision}


def predict_result(features: dict[str, float]) -> dict:
    try:
        result = predict_one(features)
    except ValueError as exc:
        record = record_pending("sensor", str(exc))
        raise HTTPException(422, {"message": "Features ausentes, extras ou inválidas.",
                                  "inspection_id": record["inspection_id"], "decision": record}) from exc
    except Exception as exc:
        raise HTTPException(503, "Modelo preditivo indisponível ou artefato incompatível.") from exc
    decision = decision_record(result["diagnosis"], result["probabilities"])
    limits = ["Ação industrial pendente de aprovação. Nenhum comando é emitido."]
    save_inspection(decision, input_kind="sensor", image_sha256=None, diagnosis=result["diagnosis"],
                    metrics={"model": result["model"], "probabilities": result["probabilities"]},
                    limitations=limits)
    return {**result, "decision": decision, "limitations": limits,
            "inspection_id": decision["inspection_id"]}


@app.post("/predict")
async def predict(request: PredictRequest):
    return await run_in_threadpool(predict_result, request.features)


@app.post("/predict-batch")
async def predict_batch(request: BatchRequest):
    results = []
    for row in request.rows:
        try:
            results.append(await run_in_threadpool(predict_result, row))
        except HTTPException as exc:
            if exc.status_code != 422 or not isinstance(exc.detail, dict):
                raise
            results.append({"detail": exc.detail.get("message", "Entrada inválida."),
                            "inspection_id": exc.detail["inspection_id"],
                            "decision": exc.detail["decision"]})
    return {"results": results}


@app.get("/inspections")
def inspections(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return {"results": list_inspections(limit, offset)}
