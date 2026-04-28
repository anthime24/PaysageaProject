"""
Générateur de zones plantables amélioré pour Garden AI.

Ce module remplace/complète plantable_mask.py avec une approche plus robuste :
1. Segmentation multi-méthode (couleur HSV + gradient + texture)
2. Zones plantables intelligentes le long des bords de pelouse
3. Masques par zone_hint précis et fiables
4. Compatible avec le code de zones plantables de ton collègue (quand disponible)

Convention BFL :
    blanc (255) = zone à MODIFIER (inpaint)
    noir (0)    = zone à CONSERVER

Auteur: Garden AI
"""
from __future__ import annotations

from pathlib import Path
from typing import Union, Optional
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Types & constantes
# ---------------------------------------------------------------------------

PlantableResult = dict  # {"mask": Image.Image, "white_pct": float, "zones": list[dict]}

# Zones prédéfinies (y_min, y_max, x_min, x_max) en ratio 0-1
# y=0 haut de l'image, y=1 bas
ZONE_DEFINITIONS: dict[str, tuple[float, float, float, float]] = {
    # Bords classiques de jardin
    "foreground_left":    (0.72, 0.95, 0.02, 0.28),
    "foreground_right":   (0.72, 0.95, 0.72, 0.98),
    "foreground_center":  (0.75, 0.97, 0.35, 0.65),
    "midground_left":     (0.50, 0.72, 0.02, 0.28),
    "midground_right":    (0.50, 0.72, 0.72, 0.98),
    "midground_center":   (0.52, 0.75, 0.35, 0.65),
    "background_left":    (0.30, 0.52, 0.05, 0.30),
    "background_right":   (0.30, 0.52, 0.70, 0.95),
    "background_center":  (0.30, 0.52, 0.35, 0.65),
    # Aliases
    "middle_left":        (0.50, 0.72, 0.02, 0.28),
    "middle_right":       (0.50, 0.72, 0.72, 0.98),
    "middle_center":      (0.52, 0.75, 0.35, 0.65),
    # Bords entiers
    "border_left":        (0.35, 0.95, 0.00, 0.20),
    "border_right":       (0.35, 0.95, 0.80, 1.00),
    "border_bottom":      (0.80, 1.00, 0.05, 0.95),
    "border_top":         (0.30, 0.50, 0.10, 0.90),
}

DEFAULT_ZONE = (0.55, 0.80, 0.38, 0.62)

# Seuils HSV
SKY_RATIO = 0.30          # 30% supérieur = ciel potentiel
MIN_WHITE_PCT = 5.0       # En dessous = masque trop petit → fallback
MAX_WHITE_PCT_SAFE = 50.0 # Au-delà = risque de redesign complet


# ---------------------------------------------------------------------------
# Utilitaires HSV
# ---------------------------------------------------------------------------

def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB [0-255] -> HSV (H:0-360, S:0-1, V:0-1)."""
    r = rgb[..., 0] / 255.0
    g = rgb[..., 1] / 255.0
    b = rgb[..., 2] / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    v = mx
    s = np.where(mx > 0, delta / mx, 0.0)
    safe_d = np.where(delta > 1e-8, delta, 1.0)
    h = np.zeros_like(r)
    c = delta > 1e-8
    h = np.where(c & (mx == r), 60 * (((g - b) / safe_d) % 6), h)
    h = np.where(c & (mx == g), 60 * ((b - r) / safe_d + 2), h)
    h = np.where(c & (mx == b), 60 * ((r - g) / safe_d + 4), h)
    return np.stack([h, s, v], axis=-1)


def _morph_close(arr: np.ndarray, size: int = 5) -> np.ndarray:
    """Fermeture morphologique (dilatation puis érosion)."""
    pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    pil = pil.filter(ImageFilter.MaxFilter(size))
    pil = pil.filter(ImageFilter.MinFilter(size))
    return (np.array(pil) >= 128).astype(np.uint8)


def _morph_open(arr: np.ndarray, size: int = 3) -> np.ndarray:
    """Ouverture morphologique (érosion puis dilatation)."""
    pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    pil = pil.filter(ImageFilter.MinFilter(size))
    pil = pil.filter(ImageFilter.MaxFilter(size))
    return (np.array(pil) >= 128).astype(np.uint8)


def _dilate(arr: np.ndarray, radius: int) -> np.ndarray:
    """Dilatation binaire."""
    pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    for _ in range(min(radius, 60)):
        pil = pil.filter(ImageFilter.MaxFilter(3))
    return (np.array(pil) >= 128).astype(np.uint8)


def _erode(arr: np.ndarray, radius: int) -> np.ndarray:
    """Érosion binaire."""
    pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    for _ in range(min(radius, 60)):
        pil = pil.filter(ImageFilter.MinFilter(3))
    return (np.array(pil) >= 128).astype(np.uint8)


def _remove_small_components(arr: np.ndarray, min_area_ratio: float = 0.003) -> np.ndarray:
    """Supprime les composantes connexes trop petites."""
    if not HAS_SCIPY:
        return arr
    labeled, n = ndimage.label(arr)
    if n == 0:
        return arr
    total = arr.size
    min_area = int(min_area_ratio * total)
    out = arr.copy()
    for i in range(1, n + 1):
        if np.sum(labeled == i) < min_area:
            out[labeled == i] = 0
    return out


def _white_pct(mask: np.ndarray) -> float:
    """% de pixels blancs dans le masque binaire."""
    return 100.0 * np.sum(mask >= 128) / mask.size if mask.size > 0 else 0.0


# ---------------------------------------------------------------------------
# Détection des zones non-plantables
# ---------------------------------------------------------------------------

def _detect_sky(hsv: np.ndarray, sky_ratio: float = SKY_RATIO) -> np.ndarray:
    """Détecte le ciel (zone haute + bleu/blanc)."""
    h_img = hsv.shape[0]
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sky = np.zeros(hsv.shape[:2], dtype=bool)
    # Zone haute systématique
    sky[:int(h_img * sky_ratio), :] = True
    # Bleu (H 190-260, S faible)
    sky |= (hh >= 190) & (hh <= 265) & (ss < 0.45)
    # Blanc/gris (très lumineux, désaturé)
    sky |= (vv > 0.87) & (ss < 0.12)
    return sky.astype(np.uint8)


def _detect_lawn(hsv: np.ndarray) -> np.ndarray:
    """Détecte la pelouse (vert HSV)."""
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lawn = (hh >= 65) & (hh <= 175) & (ss > 0.08) & (vv > 0.08)
    return lawn.astype(np.uint8)


def _detect_hardscape(hsv: np.ndarray) -> np.ndarray:
    """Détecte les surfaces dures : terrasse bois, dallage, piscine."""
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Bois/terrasse (orange/brun)
    wood = (hh >= 10) & (hh <= 45) & (ss > 0.12) & (vv > 0.25)
    # Dallage gris (faible saturation, luminosité moyenne)
    paving = (ss < 0.12) & (vv > 0.25) & (vv < 0.80)
    # Piscine (cyan)
    pool = (hh >= 165) & (hh <= 205) & (ss > 0.18)
    return (wood | paving | pool).astype(np.uint8)


# ---------------------------------------------------------------------------
# Génération du masque plantable principal
# ---------------------------------------------------------------------------

def generate_smart_plantable_mask(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    border_width_px: int = 70,
    include_flower_beds: bool = True,
    max_white_pct: float = MAX_WHITE_PCT_SAFE,
    external_zones: Optional[list[dict]] = None,
) -> PlantableResult:
    """
    Génère un masque plantable intelligent.

    Stratégie :
    1. Segmente ciel, pelouse, surfaces dures
    2. Zones plantables = bandes le long des bords de pelouse + zones non-dures non-ciel
    3. Si zones externes fournies (par le code du collègue), les prioritise
    4. Réduit si masque trop large pour éviter redesign

    Args:
        image_path: Image source
        output_path: Chemin de sauvegarde du masque (optionnel)
        border_width_px: Largeur de la bande plantable autour de la pelouse (px)
        include_flower_beds: Inclure les zones de massifs (non-pelouse, non-dur)
        max_white_pct: % max de blanc avant réduction du masque
        external_zones: Zones plantables fournies par un autre module (list de dicts
                        avec keys: x1, y1, x2, y2 en pixels, ou ratio 0-1 si
                        "is_ratio"=True). Prioritaires si fournies.

    Returns:
        dict avec:
            "mask": Image.Image (L, blanc=plantable)
            "white_pct": float
            "zones": list[dict] (bboxes des zones détectées)
            "used_fallback": bool
            "method": str
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    h_img, w_img = arr.shape[:2]
    hsv = _rgb_to_hsv(arr)

    # --- Priorité : zones externes du collègue ---
    if external_zones:
        return _build_mask_from_external_zones(img, external_zones, output_path)

    # --- 1. Segmentation ---
    sky = _detect_sky(hsv)
    lawn = _detect_lawn(hsv)
    hard = _detect_hardscape(hsv)

    non_plantable = sky | hard

    # --- 2. Zones plantables ---
    # a) Bandes plantables = proche des bords de pelouse (intérieur et bords de massifs)
    non_lawn_ground = (~sky.astype(bool) & ~lawn.astype(bool) & ~hard.astype(bool)).astype(np.uint8)

    # Dilatation de la zone non-pelouse sur la pelouse = "bordures"
    W = min(border_width_px, max(40, w_img // 18))
    dilated_non_lawn = _dilate(non_lawn_ground, W)
    border_band = (dilated_non_lawn.astype(bool) & lawn.astype(bool)).astype(np.uint8)

    # b) Massifs existants (zones non-dures, non-ciel, non-pelouse)
    flower_beds = non_lawn_ground if include_flower_beds else np.zeros_like(non_lawn_ground)

    # c) Union
    plantable = (border_band | flower_beds) & ~non_plantable.astype(bool)
    plantable = plantable.astype(np.uint8)

    # --- 3. Nettoyage morphologique ---
    plantable = _morph_close(plantable, size=7)
    plantable = _morph_open(plantable, size=3)
    plantable = _remove_small_components(plantable)

    # --- 4. Vérification % et réduction si trop large ---
    pct = _white_pct(plantable * 255)
    used_fallback = False
    method = "smart_border"

    if pct < MIN_WHITE_PCT:
        # Fallback : bandes fixes en bas de l'image
        plantable, pct, used_fallback, method = _fallback_bottom_bands(h_img, w_img, sky)

    elif pct > max_white_pct:
        # Réduire : garder seulement les contours
        plantable = _reduce_to_contour(plantable, target_pct=max_white_pct)
        pct = _white_pct(plantable * 255)
        method = "smart_border_reduced"

    # --- 5. Construire image masque finale ---
    mask_np = (plantable * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_np, mode="L")

    # Détecter les bboxes des zones plantables
    zones = _extract_zone_bboxes(plantable, h_img, w_img)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(output_path)

    return {
        "mask": mask_img,
        "white_pct": round(pct, 1),
        "zones": zones,
        "used_fallback": used_fallback,
        "method": method,
    }


def _build_mask_from_external_zones(
    img: Image.Image,
    zones: list[dict],
    output_path: Optional[Union[str, Path]] = None,
) -> PlantableResult:
    """Construit le masque depuis les zones fournies par le code du collègue."""
    w, h = img.size
    mask_np = np.zeros((h, w), dtype=np.uint8)

    for z in zones:
        if z.get("is_ratio", False):
            x1 = int(z["x1"] * w)
            y1 = int(z["y1"] * h)
            x2 = int(z["x2"] * w)
            y2 = int(z["y2"] * h)
        else:
            x1, y1, x2, y2 = int(z["x1"]), int(z["y1"]), int(z["x2"]), int(z["y2"])
        mask_np[y1:y2, x1:x2] = 255

    mask_img = Image.fromarray(mask_np, mode="L")
    pct = _white_pct(mask_np)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(output_path)

    return {
        "mask": mask_img,
        "white_pct": round(pct, 1),
        "zones": zones,
        "used_fallback": False,
        "method": "external_zones",
    }


def _fallback_bottom_bands(
    h: int, w: int, sky: np.ndarray
) -> tuple[np.ndarray, float, bool, str]:
    """Fallback : bandes plantables en bas de l'image (évite le ciel)."""
    sky_cut = int(np.argmax(np.mean(sky, axis=1) < 0.5)) if np.any(np.mean(sky, axis=1) < 0.5) else int(h * 0.35)
    sky_cut = max(sky_cut, int(h * 0.30))

    arr = np.zeros((h, w), dtype=np.uint8)
    # Bande gauche
    arr[sky_cut:, :int(w * 0.22)] = 1
    # Bande droite
    arr[sky_cut:, int(w * 0.78):] = 1
    # Bande bas centre
    arr[int(h * 0.75):, int(w * 0.10):int(w * 0.90)] = 1

    pct = _white_pct(arr * 255)
    return arr, pct, True, "fallback_bands"


def _reduce_to_contour(arr: np.ndarray, target_pct: float = 45.0) -> np.ndarray:
    """Réduit le masque par érosion progressive jusqu'à target_pct."""
    current = arr.copy()
    for _ in range(30):
        pct = _white_pct(current * 255)
        if pct <= target_pct:
            break
        pil = Image.fromarray((current * 255).astype(np.uint8), mode="L")
        pil = pil.filter(ImageFilter.MinFilter(3))
        current = (np.array(pil) >= 128).astype(np.uint8)
    return current


def _extract_zone_bboxes(arr: np.ndarray, h: int, w: int, min_area: int = 500) -> list[dict]:
    """Extrait les bboxes des composantes connexes du masque."""
    if not HAS_SCIPY:
        return []
    labeled, n = ndimage.label(arr)
    bboxes = []
    for i in range(1, n + 1):
        coords = np.where(labeled == i)
        if len(coords[0]) < min_area:
            continue
        y1, y2 = int(coords[0].min()), int(coords[0].max())
        x1, x2 = int(coords[1].min()), int(coords[1].max())
        bboxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "area": len(coords[0])})
    bboxes.sort(key=lambda b: -b["area"])
    return bboxes


# ---------------------------------------------------------------------------
# Masque par zone_hint (pour inpainting individuel par plante)
# ---------------------------------------------------------------------------

def create_zone_mask(
    image_path: Union[str, Path],
    zone_hint: str,
    output_path: Optional[Union[str, Path]] = None,
    blend_with_plantable: bool = True,
    external_plantable_mask: Optional[Union[str, Path, Image.Image]] = None,
) -> tuple[Image.Image, list[int]]:
    """
    Crée un masque pour une zone spécifique (zone_hint).

    Si blend_with_plantable=True, intersecte avec le masque plantable
    pour s'assurer qu'on ne peint que des zones plantables réelles.

    Args:
        image_path: Image source
        zone_hint: Identifiant de zone (ex: "foreground_left")
        output_path: Chemin de sauvegarde
        blend_with_plantable: Croiser avec le masque plantable
        external_plantable_mask: Masque plantable externe (du collègue)

    Returns:
        (mask_pil, [x1, y1, x2, y2])
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    ratios = ZONE_DEFINITIONS.get(zone_hint.lower().strip(), DEFAULT_ZONE)
    y1_r, y2_r, x1_r, x2_r = ratios
    y1 = int(y1_r * h)
    y2 = max(int(y2_r * h), y1 + 40)
    x1 = int(x1_r * w)
    x2 = max(int(x2_r * w), x1 + 40)

    mask_np = np.zeros((h, w), dtype=np.uint8)
    mask_np[y1:y2, x1:x2] = 255

    # Intersection avec masque plantable si disponible
    if blend_with_plantable:
        if external_plantable_mask is not None:
            if isinstance(external_plantable_mask, (str, Path)):
                pm = Image.open(external_plantable_mask).convert("L")
            else:
                pm = external_plantable_mask.convert("L")
            pm_arr = np.array(pm.resize((w, h), Image.NEAREST))
            # On garde la zone_hint + on élargit si intersection trop petite
            intersection = mask_np & (pm_arr >= 128).astype(np.uint8) * 255
            inter_pct = _white_pct(intersection)
            if inter_pct > 1.0:
                mask_np = intersection
            # sinon on garde la zone_hint telle quelle
        else:
            # Générer masque plantable à la volée
            result = generate_smart_plantable_mask(image_path)
            pm_arr = np.array(result["mask"])
            intersection = mask_np & (pm_arr >= 128).astype(np.uint8) * 255
            inter_pct = _white_pct(intersection)
            if inter_pct > 1.0:
                mask_np = intersection

    mask_pil = Image.fromarray(mask_np, mode="L")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mask_pil.save(output_path)

    return mask_pil, [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Masque combiné pour plusieurs plantes (génération globale)
# ---------------------------------------------------------------------------

def create_combined_plantable_mask(
    image_path: Union[str, Path],
    plants: list[dict],
    output_path: Optional[Union[str, Path]] = None,
    external_plantable_mask: Optional[Union[str, Path]] = None,
    external_zones: Optional[list[dict]] = None,
) -> tuple[Image.Image, list[dict]]:
    """
    Crée un masque combiné pour toutes les plantes.

    Stratégie :
    - Si external_zones fournies (code collègue) → priorité absolue
    - Sinon → génération intelligente + zones_hint par plante

    Args:
        image_path: Image source
        plants: Liste de dicts plantes (avec "zone_hint" optionnel)
        output_path: Sauvegarde
        external_plantable_mask: Masque plantable calculé par le collègue
        external_zones: Zones plantables calculées par le collègue

    Returns:
        (combined_mask_pil, plants_with_bboxes)
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Générer le masque plantable de base
    result = generate_smart_plantable_mask(
        image_path,
        external_zones=external_zones,
    )
    base_mask = np.array(result["mask"])

    # Si masque externe fourni, l'utiliser comme base
    if external_plantable_mask:
        ext = Image.open(external_plantable_mask).convert("L").resize((w, h), Image.NEAREST)
        base_mask = np.array(ext)

    combined = np.zeros((h, w), dtype=np.uint8)
    plants_out = []

    for i, plant in enumerate(plants):
        zone_hint = plant.get("zone_hint", "midground_center")
        ratios = ZONE_DEFINITIONS.get(zone_hint.lower().strip(), DEFAULT_ZONE)
        y1_r, y2_r, x1_r, x2_r = ratios

        y1 = int(y1_r * h)
        y2 = max(int(y2_r * h), y1 + 40)
        x1 = int(x1_r * w)
        x2 = max(int(x2_r * w), x1 + 40)

        # Zone locale de la plante
        zone_mask = np.zeros((h, w), dtype=np.uint8)
        zone_mask[y1:y2, x1:x2] = 255

        # Intersect avec masque plantable
        intersection = zone_mask & (base_mask >= 128).astype(np.uint8) * 255
        inter_pct = _white_pct(intersection)

        if inter_pct >= 1.0:
            # On a une vraie zone plantable → l'utiliser
            combined |= (intersection >= 128).astype(np.uint8) * 255
            final_bbox = _compute_bbox_from_mask(intersection, x1, y1, x2, y2)
        else:
            # Pas d'intersection → utiliser la zone_hint brute (l'IA fera de son mieux)
            combined |= (zone_mask >= 128).astype(np.uint8) * 255
            final_bbox = [x1, y1, x2, y2]

        p_copy = plant.copy()
        p_copy["bbox"] = final_bbox
        plants_out.append(p_copy)

    # Binariser
    combined = np.where(combined >= 128, 255, 0).astype(np.uint8)

    # Vérif % final
    pct = _white_pct(combined)
    if pct > MAX_WHITE_PCT_SAFE:
        # Réduire
        combined = _reduce_to_contour((combined >= 128).astype(np.uint8), target_pct=MAX_WHITE_PCT_SAFE)
        combined = (combined * 255).astype(np.uint8)

    mask_pil = Image.fromarray(combined, mode="L")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mask_pil.save(output_path)

    return mask_pil, plants_out


def _compute_bbox_from_mask(mask: np.ndarray, fallback_x1: int, fallback_y1: int,
                             fallback_x2: int, fallback_y2: int) -> list[int]:
    """Calcule la bbox réelle d'un masque."""
    coords = np.where(mask >= 128)
    if len(coords[0]) == 0:
        return [fallback_x1, fallback_y1, fallback_x2, fallback_y2]
    return [int(coords[1].min()), int(coords[0].min()),
            int(coords[1].max()), int(coords[0].max())]


# ---------------------------------------------------------------------------
# Visualisation debug (overlay masque sur image)
# ---------------------------------------------------------------------------

def debug_overlay(
    image_path: Union[str, Path],
    mask: Union[np.ndarray, Image.Image],
    output_path: Union[str, Path],
    alpha: float = 0.45,
    color: tuple[int, int, int] = (0, 255, 100),
) -> None:
    """
    Génère une image debug avec le masque plantable superposé en vert.

    Utile pour vérifier visuellement les zones avant d'envoyer à BFL.
    """
    img = Image.open(image_path).convert("RGB").copy()
    arr = np.array(img)

    if isinstance(mask, Image.Image):
        mask_arr = np.array(mask.convert("L").resize(img.size, Image.NEAREST))
    else:
        mask_arr = mask

    plantable = mask_arr >= 128
    overlay = arr.copy()
    overlay[plantable] = (
        (1 - alpha) * arr[plantable] + alpha * np.array(color)
    ).astype(np.uint8)

    result = Image.fromarray(overlay)

    # Ajouter un contour rouge sur les bords du masque
    draw = ImageDraw.Draw(result)
    contour_mask = Image.fromarray(mask_arr, mode="L").filter(ImageFilter.FIND_EDGES)
    contour_arr = np.array(contour_mask)
    contour_coords = np.where(contour_arr > 30)
    for y, x in zip(contour_coords[0], contour_coords[1]):
        draw.point((x, y), fill=(255, 50, 50))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"[DEBUG] Overlay sauvegardé : {output_path}")


# ---------------------------------------------------------------------------
# Interface de compatibilité avec le code du collègue
# ---------------------------------------------------------------------------

def inject_external_plantable_zones(
    zones: list[dict],
    image_size: tuple[int, int],
) -> Image.Image:
    """
    Point d'entrée pour le code du collègue.

    Ton collègue appellera cette fonction avec ses zones détectées.
    Le format attendu est une liste de dicts :
    [
        {"x1": 0.1, "y1": 0.6, "x2": 0.4, "y2": 0.9, "is_ratio": True},
        {"x1": 120, "y1": 300, "x2": 400, "y2": 500, "is_ratio": False},
        ...
    ]

    Returns:
        Image.Image (masque L, blanc=plantable)
    """
    w, h = image_size
    mask_np = np.zeros((h, w), dtype=np.uint8)
    for z in zones:
        if z.get("is_ratio", True):
            x1 = int(z["x1"] * w)
            y1 = int(z["y1"] * h)
            x2 = int(z["x2"] * w)
            y2 = int(z["y2"] * h)
        else:
            x1, y1, x2, y2 = int(z["x1"]), int(z["y1"]), int(z["x2"]), int(z["y2"])
        mask_np[y1:y2, x1:x2] = 255
    return Image.fromarray(mask_np, mode="L")
