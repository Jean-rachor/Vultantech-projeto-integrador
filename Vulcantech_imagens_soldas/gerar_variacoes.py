"""Cria variações leves de iluminação, mantendo a morfologia e sem criar rótulos."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Vulcantech_imagens_soldas"
OUTPUT = SOURCE / "derivadas"


def white_frame_bbox(image: np.ndarray, threshold: int = 248) -> tuple[int, int, int, int]:
    """Remove somente faixas brancas contínuas no perímetro; retorna bbox integral se ausentes."""
    white = np.all(image >= threshold, axis=2)
    h, w = white.shape
    top, bottom, left, right = 0, h, 0, w
    while top < bottom and white[top].mean() > 0.995: top += 1
    while bottom > top and white[bottom - 1].mean() > 0.995: bottom -= 1
    while left < right and white[:, left].mean() > 0.995: left += 1
    while right > left and white[:, right - 1].mean() > 0.995: right -= 1
    return left, top, right, bottom


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source in sorted(SOURCE.glob("solda_*")):
        if not source.is_file() or source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        raw = source.read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        x1, y1, x2, y2 = white_frame_bbox(image)
        prepared = image[y1:y2, x1:x2]
        source_hash = hashlib.sha256(raw).hexdigest()
        for suffix, factor in (("luz097", 0.97), ("luz103", 1.03)):
            variant = cv2.convertScaleAbs(prepared, alpha=factor, beta=0)
            name = f"{source.stem}__{suffix}.jpg"
            target = OUTPUT / name
            ok, encoded = cv2.imencode(".jpg", variant, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            target.write_bytes(encoded.tobytes())
            manifest.append({"file": name, "source_file": source.name, "source_sha256": source_hash,
                "derived_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "transform": {"perimeter_white_crop_bbox_xyxy": [x1, y1, x2, y2],
                              "brightness_factor": factor, "jpeg_quality": 95},
                "label": "NAO_CLASSIFICADA", "label_source": None,
                "generated_at": datetime.now(timezone.utc).isoformat()})
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Variações criadas: {len(manifest)} | rótulos: NAO_CLASSIFICADA | saída: {OUTPUT}")


if __name__ == "__main__":
    main()
