from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from pycocotools import mask as mask_utils


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # x, y, w, h


@dataclass
class CoordinateMapper:
    """
    Gère la conversion entre :
    - espace de référence (image préprocessée)
    - espace d'affichage (image upscalée pour l'utilisateur)
    """

    ref_width: int
    ref_height: int
    disp_width: int
    disp_height: int

    @classmethod
    def from_shapes(cls, ref_shape: Tuple[int, int], disp_shape: Tuple[int, int]) -> "CoordinateMapper":
        """ref_shape et disp_shape sont au format (height, width)."""
        ref_h, ref_w = ref_shape
        disp_h, disp_w = disp_shape
        return cls(ref_width=ref_w, ref_height=ref_h, disp_width=disp_w, disp_height=disp_h)

    @property
    def scale_x(self) -> float:
        return self.disp_width / self.ref_width

    @property
    def scale_y(self) -> float:
        return self.disp_height / self.ref_height

    # --- Points ---

    def to_ref_point(self, pt_disp: Point) -> Point:
        """Convertit un point (x, y) de l'espace affichage vers l'espace référence."""
        x_d, y_d = pt_disp
        x_r = x_d / self.scale_x
        y_r = y_d / self.scale_y
        # On reste en float pour garder la précision, les arrondis sont gérés plus tard
        return float(x_r), float(y_r)

    def to_disp_point(self, pt_ref: Point) -> Point:
        """Convertit un point (x, y) de l'espace référence vers l'espace affichage."""
        x_r, y_r = pt_ref
        x_d = x_r * self.scale_x
        y_d = y_r * self.scale_y
        return float(x_d), float(y_d)

    # --- Masques ---

    def mask_display_to_ref(self, mask_disp: np.ndarray) -> np.ndarray:
        """
        Convertit un masque binaire exprimé dans l'espace affichage vers l'espace référence.
        """
        ref_size = (self.ref_width, self.ref_height)  # (width, height) pour cv2.resize
        mask_resized = cv2.resize(mask_disp.astype("uint8"), ref_size, interpolation=cv2.INTER_NEAREST)
        return (mask_resized > 0).astype("uint8")


def polygon_points_display_to_ref(points_disp: Sequence[Point], mapper: CoordinateMapper) -> List[Point]:
    """Mappe tous les points d'un polygone de l'espace affichage vers l'espace référence."""
    return [mapper.to_ref_point(p) for p in points_disp]


def polygon_to_mask_ref(
    points_disp: Sequence[Point],
    mapper: CoordinateMapper,
    ref_shape: Tuple[int, int],
) -> np.ndarray:
    """
    À partir d'un polygone cliqué en affichage, génère un masque binaire dans l'espace référence.
    """
    if len(points_disp) < 3:
        return np.zeros(ref_shape, dtype="uint8")

    ref_points = polygon_points_display_to_ref(points_disp, mapper)
    ref_points_int = np.round(np.array(ref_points), 0).astype("int32")

    mask = np.zeros(ref_shape, dtype="uint8")
    cv2.fillPoly(mask, [ref_points_int], color=1)
    return mask


def compute_bbox_from_mask(mask_ref: np.ndarray) -> BBox:
    """
    Calcule la bounding box au format [x, y, width, height] en coordonnées référence.
    """
    ys, xs = np.where(mask_ref > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0, 0.0, 0.0, 0.0

    x_min = float(xs.min())
    y_min = float(ys.min())
    x_max = float(xs.max())
    y_max = float(ys.max())

    w = float(x_max - x_min + 1)
    h = float(y_max - y_min + 1)
    return x_min, y_min, w, h


def compute_centroid_from_mask(mask_ref: np.ndarray) -> Point:
    """
    Calcule le centroïde (x, y) en coordonnées référence.
    """
    ys, xs = np.where(mask_ref > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0, 0.0

    cx = float(xs.mean())
    cy = float(ys.mean())
    return cx, cy


def compute_area_pixels(mask_ref: np.ndarray) -> int:
    """Nombre de pixels à 1 dans le masque."""
    return int((mask_ref > 0).sum())


def mask_to_coco_rle(mask_ref: np.ndarray) -> dict:
    """
    Encode un masque binaire au format RLE compatible COCO / pycocotools.

    Retourne un dict avec :
    - "size": [height, width]
    - "counts": string (utf-8)
    """
    if mask_ref.dtype != np.uint8:
        mask_ref = mask_ref.astype("uint8")
    if mask_ref.ndim != 2:
        raise ValueError("mask_to_coco_rle attend un masque 2D.")

    rle = mask_utils.encode(np.asfortranarray(mask_ref))
    # pycocotools renvoie bytes pour counts -> convertir en str pour JSON
    rle["counts"] = rle["counts"].decode("utf-8")
    return {"size": [int(mask_ref.shape[0]), int(mask_ref.shape[1])], "counts": rle["counts"]}


def ensure_uint8_mask(mask: np.ndarray) -> np.ndarray:
    """Normalise un masque quelconque en masque 0/1 uint8."""
    return (mask > 0).astype("uint8")

