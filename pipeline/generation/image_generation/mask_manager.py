"""
Gestion des masques individuels par plante.

Modes: "random" (bbox aléatoire) ou "fixed" (zone_hint prédéfini).
Noir = ne pas modifier, Blanc = modifier (zone à inpaint).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Iterable, List

import numpy as np
from PIL import Image, ImageFilter

from .config import PLACEMENT_MODE

# Mapping zone_hint -> (y_min, y_max, x_min, x_max) en ratio 0-1 (mode "fixed")
# y=0 haut, y=1 bas
ZONE_HINT_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "foreground_left": (0.70, 0.90, 0.05, 0.30),
    "foreground_right": (0.70, 0.90, 0.70, 0.95),
    "foreground_center": (0.75, 0.95, 0.35, 0.65),
    "midground_left": (0.50, 0.70, 0.05, 0.30),
    "midground_right": (0.50, 0.70, 0.70, 0.95),
    "midground_center": (0.55, 0.75, 0.35, 0.65),
    "middle_left": (0.50, 0.70, 0.05, 0.30),
    "middle_right": (0.50, 0.70, 0.70, 0.95),
    "middle_center": (0.55, 0.75, 0.35, 0.65),
    "background_left": (0.35, 0.50, 0.10, 0.35),
    "background_right": (0.35, 0.50, 0.65, 0.90),
    "background_center": (0.35, 0.50, 0.35, 0.65),
}
DEFAULT_ZONE = (0.60, 0.80, 0.40, 0.60)


def create_manual_test_mask(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    cx: int = 160,
    cy: int = 320,
    radius: int = 40,
) -> Path:
    """
    Crée un masque circulaire pour test manuel.
    Blanc = zone à inpaint, Noir = conserver.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    mask = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    mask[inside] = 255
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).convert("L").save(out)
    return out


@dataclass
class MaskResult:
    """Résultat : chemin du masque + bbox."""

    mask_path: str
    bbox: list[int]  # [x1, y1, x2, y2]


def _bbox_intersection_area(a: list[int], b: list[int]) -> int:
    """Aire d'intersection entre deux bboxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return int((ix2 - ix1) * (iy2 - iy1))


def _create_random_bbox(w: int, h: int, plant_id: str, plant_index: int) -> tuple[int, int, int, int]:
    """
    Génère une bbox aléatoire dans la zone jardin (évite le ciel).

    - margin = 20px
    - bbox_w = int(w * 0.18), bbox_h = int(h * 0.18)
    - y_min >= 45% de la hauteur (évite le ciel)
    """
    margin = 20
    bbox_w = max(32, int(w * 0.18))
    bbox_h = max(32, int(h * 0.18))

    y_min_allowed = int(h * 0.45)
    x_min_allowed = margin
    x_max_allowed = max(x_min_allowed, w - bbox_w - margin)
    y_max_allowed = max(y_min_allowed, h - bbox_h - margin)

    rng = np.random.default_rng(seed=hash(plant_id) % (2**32) + plant_index * 1000)
    x1 = int(rng.integers(x_min_allowed, x_max_allowed + 1))
    y1 = int(rng.integers(y_min_allowed, y_max_allowed + 1))
    x2 = min(x1 + bbox_w, w)
    y2 = min(y1 + bbox_h, h)

    return x1, y1, x2, y2


class MaskManager:
    """
    Crée et sauvegarde les masques par plante.
    Blanc = zone à inpaint, Noir = conserver.
    """

    def __init__(self, masks_dir: Union[str, Path], use_ellipse: bool = False):
        self.masks_dir = Path(masks_dir)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.use_ellipse = use_ellipse
        self._plant_counter = 0

    def create_mask(
        self,
        image_path: Union[str, Path],
        plant_id: str,
        zone_hint: str = "midground_center",
    ) -> MaskResult:
        """
        Crée un masque pour une plante et le sauvegarde.
        """
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        if PLACEMENT_MODE == "random":
            x1, y1, x2, y2 = _create_random_bbox(w, h, plant_id, self._plant_counter)
            self._plant_counter += 1
        else:
            ratios = ZONE_HINT_REGIONS.get(
                (zone_hint or "").lower().strip(),
                DEFAULT_ZONE,
            )
            y_min_r, y_max_r, x_min_r, x_max_r = ratios
            y1, y2 = int(y_min_r * h), int(y_max_r * h)
            x1, x2 = int(x_min_r * w), int(x_max_r * w)
            y2, x2 = max(y2, y1 + 32), max(x2, x1 + 32)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        mask_pil = Image.fromarray(mask).convert("L")
        mask_path = self.masks_dir / f"{plant_id}.png"
        mask_pil.save(mask_path)

        return MaskResult(mask_path=str(mask_path), bbox=[x1, y1, x2, y2])

    # ------------------------------------------------------------------
    # Nouveau : masque individuel pour pipeline séquentiel
    # ------------------------------------------------------------------

    def create_individual_plant_mask(
        self,
        image_path: Union[str, Path],
        plant: dict,
        plant_index: int,
        already_placed: Iterable[list[int]],
        plantable_zones_mask: Image.Image | None = None,
    ) -> MaskResult:
        """
        Crée un masque individuel pour une plante.

        - Utilise zone_hint pour la position de base
        - Adapte la taille selon height_cm/width_cm (perspective approximative)
        - Évite au mieux les chevauchements avec already_placed (décalage)
        - Si plantable_zones_mask est fourni, intersecte avec ce masque
          et garde la plus grande zone plantable dans la bbox.
        """
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        zone_hint = (plant.get("zone_hint") or "midground_center").lower().strip()
        ratios = ZONE_HINT_REGIONS.get(zone_hint, DEFAULT_ZONE)
        y_min_r, y_max_r, x_min_r, x_max_r = ratios
        base_y1, base_y2 = int(y_min_r * h), int(y_max_r * h)
        base_x1, base_x2 = int(x_min_r * w), int(x_max_r * w)

        # Heuristique de hauteur selon zone (foreground / midground / background)
        zone_prefix = zone_hint.split("_")[0]
        if zone_prefix.startswith("foreground"):
            min_h, max_h = 0.15 * h, 0.25 * h
        elif zone_prefix.startswith("background"):
            min_h, max_h = 0.06 * h, 0.12 * h
        else:  # midground / middle / défaut
            min_h, max_h = 0.10 * h, 0.18 * h

        # Mapping rudimentaire height_cm -> facteur [0,1]
        height_cm = float(plant.get("height_cm") or 0.0)
        if height_cm <= 0:
            t = 0.5
        else:
            # Clamp 30–200cm
            t = max(0.0, min(1.0, (height_cm - 30.0) / (200.0 - 30.0)))
        bbox_h = int(min_h + t * (max_h - min_h))

        # Largeur : basée soit sur width_cm, soit sur ratio largeur/hauteur
        width_cm = float(plant.get("width_cm") or 0.0)
        if width_cm > 0 and height_cm > 0:
            aspect = max(0.5, min(2.0, width_cm / max(height_cm, 1.0)))
        else:
            aspect = 0.8
        bbox_w = int(bbox_h * aspect)

        # Centrer dans la zone_hint
        zone_cx = (base_x1 + base_x2) // 2
        zone_cy = (base_y1 + base_y2) // 2
        x1 = max(0, zone_cx - bbox_w // 2)
        x2 = min(w, x1 + bbox_w)
        y2 = min(h, zone_cy + bbox_h // 2)
        y1 = max(0, y2 - bbox_h)

        # Collision avoidance simple: décaler horizontalement si forte intersection
        bbox = [x1, y1, x2, y2]
        placed: List[list[int]] = list(already_placed)
        max_shift = int(0.15 * w)
        step = max(10, int(0.04 * w))
        if placed:
            def too_much_overlap(b: list[int]) -> bool:
                area_b = (b[2] - b[0]) * (b[3] - b[1])
                if area_b <= 0:
                    return False
                for other in placed:
                    inter = _bbox_intersection_area(b, other)
                    if inter / area_b > 0.35:
                        return True
                return False

            if too_much_overlap(bbox):
                # Essayer de décaler à gauche puis à droite
                for delta in range(step, max_shift + step, step):
                    cand_left = [max(0, x1 - delta), y1, max(0, x1 - delta) + bbox_w, y2]
                    cand_left[2] = min(w, cand_left[2])
                    if not too_much_overlap(cand_left):
                        bbox = cand_left
                        break
                    cand_right = [min(w - bbox_w, x1 + delta), y1, min(w, x1 + delta + bbox_w), y2]
                    if not too_much_overlap(cand_right):
                        bbox = cand_right
                        break

        x1, y1, x2, y2 = bbox

        # Masque rectangulaire de base
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        # Intersection avec masque plantable externe si fourni
        if plantable_zones_mask is not None:
            pm = plantable_zones_mask.convert("L").resize((w, h), Image.NEAREST)
            pm_arr = np.array(pm)
            plantable = (pm_arr >= 128)

            # 1) Essai direct : bbox actuelle ∩ plantable
            intersection = (mask > 0) & plantable
            inter_pixels = int(np.count_nonzero(intersection))

            # 2) Si vide : repositionner la bbox DANS une zone blanche
            if inter_pixels == 0:
                ys, xs = np.where(plantable)
                if len(xs) > 0:
                    # Choix pseudo-aléatoire mais stable (par plant_id + index)
                    seed = (hash(str(plant.get("plant_id", ""))) + int(plant_index) * 10007) & 0xFFFFFFFF
                    rng = np.random.default_rng(seed=seed)

                    # Plusieurs tentatives pour garantir au moins quelques pixels blancs
                    for _ in range(12):
                        k = int(rng.integers(0, len(xs)))
                        cx = int(xs[k])
                        cy = int(ys[k])

                        nx1 = max(0, cx - bbox_w // 2)
                        ny2 = min(h, cy + bbox_h // 2)
                        ny1 = max(0, ny2 - bbox_h)
                        nx2 = min(w, nx1 + bbox_w)

                        bbox_mask = np.zeros((h, w), dtype=np.uint8)
                        bbox_mask[ny1:ny2, nx1:nx2] = 1
                        inter = (bbox_mask > 0) & plantable
                        if np.count_nonzero(inter) > 0:
                            x1, y1, x2, y2 = nx1, ny1, nx2, ny2
                            mask = (inter.astype(np.uint8) * 255)
                            break
                # si plantable est vide -> on garde le mask initial (fallback)
            else:
                mask = (intersection.astype(np.uint8) * 255)

        # Lissage très léger des bords (pour debug/overlay, BFL recevra une version binarisée)
        mask_pil = Image.fromarray(mask, mode="L")
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=1))
        # Binarisation stricte pour l'API BFL (0/255)
        mask_arr = np.array(mask_pil)
        mask_bin = np.where(mask_arr >= 128, 255, 0).astype(np.uint8)
        mask_pil = Image.fromarray(mask_bin, mode="L")

        mask_path = self.masks_dir / f"plant_{plant.get('plant_id', plant_index):s}.png"
        mask_pil.save(mask_path)

        return MaskResult(mask_path=str(mask_path), bbox=[x1, y1, x2, y2])

    def create_combined_mask(
        self,
        image_path: Union[str, Path],
        plants: list[dict],
        output_path: Union[str, Path],
    ) -> list[dict]:
        """
        Crée un masque unique combinant toutes les zones de plantation.
        Retourne la liste des plantes avec leurs bboxes calculées.
        """
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        updated_plants = []

        for i, plant in enumerate(plants):
            zone_hint = plant.get("zone_hint", "midground_center")
            ratios = ZONE_HINT_REGIONS.get(zone_hint.lower().strip(), DEFAULT_ZONE)
            y1, y2 = int(ratios[0] * h), int(ratios[1] * h)
            x1, x2 = int(ratios[2] * w), int(ratios[3] * w)
            y2, x2 = max(y2, y1 + 32), max(x2, x1 + 32)
            
            combined_mask[y1:y2, x1:x2] = 255
            
            plant_copy = plant.copy()
            plant_copy["bbox"] = [x1, y1, x2, y2]
            updated_plants.append(plant_copy)

        Image.fromarray(combined_mask).convert("L").save(output_path)
        return updated_plants
