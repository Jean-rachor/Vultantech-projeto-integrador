from functools import lru_cache
from io import BytesIO
from pathlib import Path
import sqlite3

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import DATABASE_PATH
from .database import reference_images

MAX_IMAGE_PIXELS = 20_000_000
MAX_ANALYSIS_EDGE = 1600
MAX_REFERENCE_EDGE = 1000


def _decode_upload(data: bytes) -> tuple[np.ndarray, tuple[int, int], str]:
    if not data:
        raise ValueError("Arquivo de imagem vazio.")
    try:
        with Image.open(BytesIO(data)) as probe:
            if probe.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Envie uma imagem JPEG, PNG ou WebP.")
            width, height = probe.size
            image_format = probe.format
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Dimensões de imagem inválidas ou acima do limite de análise.")
            probe.verify()
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Arquivo inválido, corrompido ou em formato não suportado.") from exc

    array = np.frombuffer(data, dtype=np.uint8)
    try:
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("O OpenCV não conseguiu decodificar a imagem.") from exc
    if image is None:
        raise ValueError("O OpenCV não conseguiu decodificar a imagem.")
    return image, (width, height), image_format


def validate_image(data: bytes) -> dict:
    image, (width, height), image_format = _decode_upload(data)
    return {"width": width, "height": height, "channels": int(image.shape[2]), "format": image_format}


@lru_cache(maxsize=4)
def _cached_references(path: str) -> tuple[tuple[dict, ...], int]:
    # Inspection writes change the SQLite file too; key only the static catalog
    # by database path so those writes do not decode every reference again.
    decoded = []
    invalid = 0
    detector = cv2.ORB_create(nfeatures=900, scaleFactor=1.2, nlevels=8, fastThreshold=12)
    for item in reference_images(Path(path), limit=200):
        try:
            array = np.frombuffer(item["image_blob"], dtype=np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                invalid += 1
                continue
            height, width = image.shape[:2]
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                invalid += 1
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if max(width, height) > MAX_REFERENCE_EDGE:
                scale = MAX_REFERENCE_EDGE / max(width, height)
                gray = cv2.resize(gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            decoded.append({
                "id": item["id"],
                "filename": item["filename"],
                "label": item["label"],
                "label_source": item["label_source"],
                "source_type": item["source_type"],
                "width": width,
                "height": height,
                "gray": gray,
                "descriptor_count": len(keypoints),
                "descriptors": descriptors,
            })
        except (cv2.error, TypeError, ValueError):
            invalid += 1
    return tuple(decoded), invalid


def load_reference_images(path: Path = DATABASE_PATH) -> tuple[tuple[dict, ...], int]:
    try:
        resolved = path.resolve()
        resolved.stat()
    except OSError:
        return (), 0
    try:
        return _cached_references(str(resolved))
    except (OSError, sqlite3.Error):
        return (), 0


def reference_catalog(path: Path = DATABASE_PATH) -> dict:
    images, invalid = load_reference_images(path)
    return {
        "available": bool(images),
        "source_type": "projeto",
        "images_loaded": len(images),
        "images_unreadable": invalid,
        "labelled_images": sum(bool(item["label_source"]) and item["label"] != "NAO_CLASSIFICADA" for item in images),
        "classification_ready": False,
        "reason": "As imagens disponíveis não têm rótulos confiáveis para validar classes de defeito.",
    }


def _compare_references(gray: np.ndarray, references: tuple[dict, ...], invalid: int) -> dict:
    if not references:
        return {
            "status": "INDISPONIVEL",
            "method": "ORB com teste de razão; comparação experimental, sem classificação.",
            "references_available": 0,
            "references_used": 0,
            "references_unreadable": invalid,
            "top_matches": [],
            "reason": "Não há imagens originais de referência legíveis no banco.",
        }

    detector = cv2.ORB_create(nfeatures=900, scaleFactor=1.2, nlevels=8, fastThreshold=12)
    query_keypoints, query_descriptors = detector.detectAndCompute(gray, None)
    if query_descriptors is None or not query_keypoints:
        return {
            "status": "SEM_DESCRITORES",
            "method": "ORB com teste de razão; comparação experimental, sem classificação.",
            "references_available": len(references),
            "references_used": 0,
            "references_unreadable": invalid,
            "query_descriptors": 0,
            "top_matches": [],
            "reason": "A imagem enviada não contém pontos locais suficientes para comparação ORB.",
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    ranked = []
    for reference in references:
        try:
            ref_descriptors = reference["descriptors"]
            if ref_descriptors is None or not reference["descriptor_count"]:
                continue
            pairs = matcher.knnMatch(query_descriptors, ref_descriptors, k=2)
            good = sum(1 for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance)
            ranked.append({
                "id": reference["id"],
                "filename": reference["filename"],
                "label": reference["label"],
                "label_source": reference["label_source"],
                "source_type": reference["source_type"],
                "ratio_test_matches": good,
                "query_descriptor_match_fraction": round(good / len(query_descriptors), 6),
            })
        except cv2.error:
            continue
    ranked.sort(key=lambda item: (item["ratio_test_matches"], item["query_descriptor_match_fraction"]), reverse=True)
    return {
        "status": "COMPARACAO_EXPERIMENTAL",
        "method": "Descritores ORB e teste de razão Hamming (0,75), ordenados por correspondências locais. Não é confiança nem critério de defeito.",
        "references_available": len(references),
        "references_used": len(ranked),
        "references_unreadable": invalid,
        "query_descriptors": len(query_descriptors),
        "top_matches": ranked[:5],
        "classification_ready": False,
        "reason": "As referências não têm rótulos confiáveis e suas condições de captura não estão validadas como equivalentes.",
    }


def inspect_image(data: bytes, reference_path: Path = DATABASE_PATH) -> dict:
    image, original_size, _ = _decode_upload(data)
    original_width, original_height = original_size
    working = image
    if max(original_width, original_height) > MAX_ANALYSIS_EDGE:
        scale = MAX_ANALYSIS_EDGE / max(original_width, original_height)
        working = cv2.resize(image, (round(original_width * scale), round(original_height * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    sharp = cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32))
    median = float(np.median(sharp))
    low, high = max(0, int(0.66 * median)), min(255, int(1.33 * median))
    canny_high = max(low + 1, high)
    edges = cv2.Canny(sharp, low, canny_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = [float(cv2.contourArea(contour)) for contour in contours if cv2.contourArea(contour) > 0]

    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    visual_regions = {}
    for channel, name in ((1, "saturation"), (2, "value")):
        plane = hsv[:, :, channel]
        threshold, mask = cv2.threshold(plane, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        visual_regions[name] = {
            "otsu_threshold": float(threshold),
            "pixel_fraction": round(float(np.count_nonzero(mask)) / mask.size, 6),
        }

    references, invalid_references = load_reference_images(reference_path)
    comparison = _compare_references(gray, references, invalid_references)
    working_height, working_width = working.shape[:2]
    return {
        "image": {
            "width": original_width,
            "height": original_height,
            "channels": 3,
            "analysis_width": working_width,
            "analysis_height": working_height,
            "resized_for_analysis": (working_width, working_height) != (original_width, original_height),
        },
        "classification": "NAO_CLASSIFICADA",
        "geometric_experimental": {
            "edge_pixel_fraction": round(float(np.count_nonzero(edges)) / edges.size, 6),
            "contour_count": len(contours),
            "nonzero_contour_count": len(areas),
            "contour_area_sum_px2": round(sum(areas), 3),
            "canny_thresholds": {"low": low, "high": canny_high},
        },
        "haz_visual_experimental": {
            "status": "ANALISE_VISUAL_EXPERIMENTAL",
            "method": "HSV com threshold Otsu em saturação e valor; regiões visuais sem inferência térmica.",
            "regions": visual_regions,
            "temperature_inference": False,
        },
        "reference_comparison": comparison,
        "thermal_analysis": {
            "status": "INDISPONIVEL",
            "reason": "Não há imagem termográfica calibrada nem escala térmica/parâmetros de captura.",
        },
    }
