"""
Masque plantable pour design jardin global.

blanc = zone à modifier (BFL)
noir = zone à conserver (BFL)

Améliorations:
- Ciel (haut clair/bleu) en noir
- Bordures: bande le long des transitions pelouse/non-pelouse (distance transform)
- Morphologie: close/open, suppression petites composantes
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageFilter

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def compute_mask_white_percent(mask: Union[str, Path, Image.Image]) -> float:
    """Calcule le % de pixels blancs (>= 128). Returns 0.0-100.0."""
    if isinstance(mask, (str, Path)):
        mask = Image.open(mask).convert("L")
    elif isinstance(mask, Image.Image):
        mask = mask.convert("L")
    arr = np.array(mask)
    total = arr.size
    white = np.sum(arr >= 128)
    return 100.0 * white / total if total > 0 else 0.0


def create_fallback_mask_exclude_sky(
    image_path: Union[str, Path],
    sky_ratio: float = 0.35,
) -> Image.Image:
    """Fallback: tout blanc sauf les sky_ratio*100% supérieurs (ciel noir)."""
    img = Image.open(image_path).convert("RGB")
    h, w = img.size[1], img.size[0]
    sky_cut = int(h * sky_ratio)
    mask_np = np.ones((h, w), dtype=np.uint8) * 255
    mask_np[:sky_cut, :] = 0
    return Image.fromarray(mask_np, mode="L")


def create_border_mask(
    mask: Union[str, Path, Image.Image],
    erosion_pixels: int = 15,
    output_path: Union[str, Path] | None = None,
) -> tuple[Image.Image, float]:
    """
    Crée un masque "bordures" : contour des zones plantables (original - érodé).
    Utilisé quand white_pct > 60% pour éviter redesign complet.
    Returns: (border_mask_pil, white_pct)
    """
    if isinstance(mask, (str, Path)):
        mask = Image.open(mask).convert("L")
    elif isinstance(mask, Image.Image):
        mask = mask.convert("L")
    arr_orig = (np.array(mask) >= 128).astype(np.uint8)

    # Érosion progressive
    eroded = arr_orig.copy()
    pil_tmp = Image.fromarray((eroded * 255).astype(np.uint8), mode="L")
    for _ in range(max(1, erosion_pixels)):
        pil_tmp = pil_tmp.filter(ImageFilter.MinFilter(3))
    arr_eroded = (np.array(pil_tmp) >= 128).astype(np.uint8)

    # Bordure = original - intérieur (érodé)
    border = (arr_orig == 1) & (arr_eroded == 0)
    border_np = (border.astype(np.uint8) * 255).astype(np.uint8)
    mask_border = Image.fromarray(border_np, mode="L")

    white_pct = compute_mask_white_percent(mask_border)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mask_border.save(output_path)
    return mask_border, white_pct


def reduce_mask_to_borders(
    mask: Union[str, Path, Image.Image],
    max_white_percent: float = 45.0,
    output_path: Union[str, Path] | None = None,
) -> tuple[Image.Image, float]:
    """
    Réduit le masque (érosion) pour éviter un masque trop large (>60% = redesign).
    Garde une bande le long des bords des zones blanches.
    Returns: (mask_reduced, white_pct)
    """
    if isinstance(mask, (str, Path)):
        mask = Image.open(mask).convert("L")
    elif isinstance(mask, Image.Image):
        mask = mask.convert("L")
    arr = np.array(mask)
    white_bin = (arr >= 128).astype(np.uint8)

    # Éroder progressivement jusqu'à white_pct < max_white_percent
    for _ in range(25):
        pct = 100.0 * np.sum(white_bin) / white_bin.size
        if pct <= max_white_percent:
            break
        # Érosion 1px
        pil = Image.fromarray((white_bin * 255).astype(np.uint8), mode="L")
        pil = pil.filter(ImageFilter.MinFilter(3))
        white_bin = (np.array(pil) >= 128).astype(np.uint8)

    mask_reduced = Image.fromarray((white_bin * 255).astype(np.uint8), mode="L")
    white_pct = compute_mask_white_percent(mask_reduced)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mask_reduced.save(output_path)
    return mask_reduced, white_pct


def create_fallback_mask_full(image_path: Union[str, Path]) -> Image.Image:
    """Fallback: masque plein (toute l'image en blanc)."""
    img = Image.open(image_path).convert("RGB")
    h, w = img.size[1], img.size[0]
    return Image.new("L", (w, h), 255)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB [0-255] -> HSV. H: 0-360, S: 0-1, V: 0-1."""
    r, g, b = rgb[..., 0] / 255.0, rgb[..., 1] / 255.0, rgb[..., 2] / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    v = mx
    delta = mx - mn
    s = np.where(mx > 0, delta / mx, 0.0)
    h = np.zeros_like(r)
    cond = delta > 1e-8
    safe_delta = np.where(cond, delta, 1.0)
    h = np.where(cond & (mx == r), 60 * (((g - b) / safe_delta) % 6), h)
    h = np.where(cond & (mx == g), 60 * ((b - r) / safe_delta + 2), h)
    h = np.where(cond & (mx == b), 60 * ((r - g) / safe_delta + 4), h)
    return np.stack([h, s, v], axis=-1)


def _dilate_binary(arr: np.ndarray, radius: int) -> np.ndarray:
    """Dilate binaire (radius en pixels)."""
    if HAS_SCIPY:
        from scipy.ndimage import binary_dilation
        struct = ndimage.generate_binary_structure(2, 1)
        out = arr.astype(bool)
        for _ in range(min(radius, 50)):
            out = binary_dilation(out, structure=struct)
        return out.astype(np.uint8)
    # Fallback PIL: ~1px par itération
    pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    for _ in range(min(radius, 40)):
        pil = pil.filter(ImageFilter.MaxFilter(3))
    return (np.array(pil) >= 128).astype(np.uint8)


def generate_plantable_mask(
    image_path: Union[str, Path],
    exclude_lawn: bool = True,
    output_path: Union[str, Path] | None = None,
    min_white_percent: float = 5.0,
    border_width_px: int = 60,
) -> tuple[Image.Image, float, bool]:
    """
    Génère plantable_mask : blanc=plantable (zone à modifier), noir=non-plantable.

    - Ciel (zone haute claire/bleue) -> noir
    - Si exclude_lawn: bordures = bande le long des transitions pelouse/arbustes
    - Piscine, terrasse bois -> noir
    - Morphologie: close/open, suppression petites composantes

    Returns:
        (mask_pil, white_percent, used_fallback)
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    h_img, w_img = arr.shape[:2]
    hsv = _rgb_to_hsv(arr)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # 1. Ciel: zone haute très claire ou bleue
    sky_ratio = 0.35
    sky_cut = int(h_img * sky_ratio)
    is_sky = np.zeros((h_img, w_img), dtype=bool)
    is_sky[:sky_cut, :] = True
    # Affiner: bleu (H 200-260) ou très clair (V > 0.9)
    is_sky |= (hh >= 200) & (hh <= 260) & (ss < 0.4)
    is_sky |= (vv > 0.88) & (ss < 0.15)
    non_plantable = is_sky.copy()

    # 2. Pelouse (HSV vert)
    is_lawn = (hh >= 70) & (hh <= 170) & (ss > 0.1) & (vv > 0.1)
    is_non_lawn_ground = ~is_sky & ~is_lawn

    # 3. Piscine (cyan)
    is_pool = (hh >= 170) & (hh <= 200) & (ss > 0.2)
    non_plantable |= is_pool

    # 4. Terrasse bois (orange/brun)
    is_wood = (hh >= 10) & (hh <= 45) & (ss > 0.15) & (vv > 0.3)
    non_plantable |= is_wood

    if exclude_lawn:
        # Bordures: bande le long des transitions pelouse / non-pelouse
        # plantable = non_lawn_ground | (bande de largeur W le long du bord)
        non_lawn_bin = is_non_lawn_ground.astype(np.uint8)
        W = min(border_width_px, max(40, w_img // 20))
        dilated_non_lawn = _dilate_binary(non_lawn_bin, W)
        # Band = zone dans pelouse qui est proche de non-pelouse
        band = dilated_non_lawn & is_lawn
        plantable = is_non_lawn_ground | band
    else:
        plantable = ~non_plantable

    mask_np = np.where(plantable, 255, 0).astype(np.uint8)

    # 5. Morphologie: close (remplir trous) puis open (enlever petits îlots)
    mask_pil = Image.fromarray(mask_np, mode="L")
    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(5))
    mask_pil = mask_pil.filter(ImageFilter.MinFilter(3))
    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))

    # 6. Suppression petites composantes (< 0.5% de l'image)
    if HAS_SCIPY:
        labeled, num_feat = ndimage.label(np.array(mask_pil) >= 128)
        if num_feat > 0:
            min_area = int(0.005 * h_img * w_img)
            for i in range(1, num_feat + 1):
                if np.sum(labeled == i) < min_area:
                    mask_np[labeled == i] = 0
            mask_pil = Image.fromarray(mask_np, mode="L")

    white_pct = compute_mask_white_percent(mask_pil)
    used_fallback = False

    if white_pct < min_white_percent:
        # Fallback: tout sauf ciel
        mask_pil = create_fallback_mask_exclude_sky(image_path)
        white_pct = compute_mask_white_percent(mask_pil)
        used_fallback = True

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mask_pil.save(output_path)

    return mask_pil, white_pct, used_fallback
