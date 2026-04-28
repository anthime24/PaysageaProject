from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import json
import numpy as np


Point = Tuple[float, float]


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_zone_dict(
    zone_id: int,
    mode: str,
    label: str,
    bbox,
    centroid,
    area_pixels: int,
    polygon_points: Sequence[Point],
    mask_rle: Dict[str, Any],
) -> Dict[str, Any]:
    x, y, w, h = bbox
    cx, cy = centroid

    return {
        "zone_id": int(zone_id),
        "mode": mode,
        "label": label,
        "bbox": [float(x), float(y), float(w), float(h)],
        "centroid": [float(cx), float(cy)],
        "area_pixels": int(area_pixels),
        "polygon_points": [[float(px), float(py)] for px, py in polygon_points],
        "mask_rle": {
            "size": list(mask_rle["size"]),
            "counts": mask_rle["counts"],
        },
        "created_at": _now_iso_utc(),
    }


def export_user_zones(
    zones: Sequence[Dict[str, Any]],
    output_json_path: Path,
    image_id: str,
    image_filename: str,
    image_size: Tuple[int, int],
    version: str = "user_zone_v1",
) -> None:
    width, height = image_size

    payload = {
        "version": version,
        "image_id": image_id,
        "image_filename": image_filename,
        "image_size": [int(width), int(height)],
        "zones": list(zones),
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_mask_png(mask_ref: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # On sauvegarde en 0/255 pour être bien lisible
    img = (mask_ref > 0).astype("uint8") * 255
    cv2.imwrite(str(path), img)


def save_overlay_png(ref_img: np.ndarray, mask_ref: np.ndarray, path: Path) -> None:
    """
    Sauvegarde une image overlay de la zone sur l'image de référence.

    Si le masque est vide, on sauvegarde simplement l'image originale.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if ref_img is None or ref_img.size == 0:
        print("save_overlay_png: image de référence vide, rien à sauvegarder.")
        return

    overlay = ref_img.copy()

    # masque binaire 2D
    mask_bool = (mask_ref > 0)
    if mask_bool.shape[:2] != overlay.shape[:2]:
        print(
            "save_overlay_png: dimensions masque / image incohérentes, "
            f"mask={mask_bool.shape}, img={overlay.shape}"
        )
        # on ne tente pas l'overlay si les tailles ne concordent pas
        ok = cv2.imwrite(str(path), np.ascontiguousarray(overlay))
        if not ok:
            print(f"save_overlay_png: échec de l'écriture du fichier {path}")
        return

    if mask_bool.any():
        color = np.array([0, 0, 255], dtype="uint8")  # rouge en BGR
        # overlay[mask_bool] a la forme (N, 3) -> compatible avec color (3,)
        overlay[mask_bool] = (0.4 * overlay[mask_bool] + 0.6 * color).astype("uint8")
    else:
        print("save_overlay_png: masque vide, overlay identique à l'image de référence.")

    ok = cv2.imwrite(str(path), np.ascontiguousarray(overlay))
    if not ok:
        print(f"save_overlay_png: échec de l'écriture du fichier {path}")

