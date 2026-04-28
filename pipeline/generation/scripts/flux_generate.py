#!/usr/bin/env python3
"""
Script minimal : génère plantable_mask.png puis inpaint FLUX.

Usage:
    python -m garden_ai.scripts.flux_generate

Input:  data/garden.jpg
Output: outputs/final_garden.png, outputs/plantable_mask.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0-255] to HSV. H: 0-360, S: 0-1, V: 0-1."""
    r, g, b = rgb[..., 0] / 255.0, rgb[..., 1] / 255.0, rgb[..., 2] / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    v = mx
    delta = mx - mn
    s = np.where(mx > 0, delta / mx, 0.0)
    h = np.zeros_like(r)
    cond = delta > 0
    h = np.where(cond & (mx == r), 60 * (((g - b) / delta) % 6), h)
    h = np.where(cond & (mx == g), 60 * ((b - r) / delta + 2), h)
    h = np.where(cond & (mx == b), 60 * ((r - g) / delta + 4), h)
    return np.stack([h, s, v], axis=-1)


def generate_plantable_mask(image_path: Path) -> Image.Image:
    """
    Génère un masque plantable (blanc=plantable, noir=protégé).

    - HSV thresholding : herbe/sol (vert/brun) dans la moitié inférieure
    - Exclut le ciel (35% supérieur)
    - Exclut piscine (cyan) et terrasse bois (orange/brun planches)
    - Morphologie close/open pour lisser
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    h_img, w_img = arr.shape[:2]

    hsv = _rgb_to_hsv(arr)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Masque ciel : exclure les 35% supérieurs
    sky_cut = int(h_img * 0.35)
    not_sky = np.zeros((h_img, w_img), dtype=bool)
    not_sky[sky_cut:, :] = True

    # Herbe : H 80-160, S > 0.15, V > 0.15
    is_green = (hh >= 80) & (hh <= 160) & (ss > 0.15) & (vv > 0.15)

    # Sol/brun : H 15-50, S > 0.1, V > 0.1
    is_brown = (hh >= 15) & (hh <= 50) & (ss > 0.1) & (vv > 0.1)

    # Vert plus large (feuillages, pelouse)
    is_green_wide = (hh >= 70) & (hh <= 170) & (ss > 0.12) & (vv > 0.12)

    plantable = (is_green | is_brown | is_green_wide) & not_sky

    # Exclure piscine (cyan) : H 170-200, S élevé
    is_pool = (hh >= 170) & (hh <= 200) & (ss > 0.3)
    plantable = plantable & ~is_pool

    # Exclure terrasse bois (orange/brun planches) : H 10-40, S modéré, zones assez claires
    is_wood = (hh >= 10) & (hh <= 45) & (ss > 0.2) & (ss < 0.7) & (vv > 0.35)
    plantable = plantable & ~is_wood

    mask = np.where(plantable, 255, 0).astype(np.uint8)
    mask_pil = Image.fromarray(mask, mode="L")

    # Morphologie : close (remplir trous) puis open (enlever petits bruits)
    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(5))  # dilation
    mask_pil = mask_pil.filter(ImageFilter.MinFilter(5))  # erosion -> close
    mask_pil = mask_pil.filter(ImageFilter.MinFilter(5))  # erosion
    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(5))  # dilation -> open

    return mask_pil


def main() -> int:
    print("=== FLUX Generate (minimal) ===\n")

    image_path = DATA_DIR / "garden.jpg"
    if not image_path.exists():
        print(f"❌ Image non trouvée : {image_path}", file=sys.stderr)
        return 1

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = OUTPUTS_DIR / "plantable_mask.png"
    out_path = OUTPUTS_DIR / "final_garden.png"

    # 1. Générer le masque plantable
    print("1. Génération masque plantable...")
    mask = generate_plantable_mask(image_path)
    mask.save(mask_path)
    print(f"   ✓ {mask_path}")

    # 2. Inpaint FLUX
    print("2. Inpaint FLUX...")
    prompt = (
        "Add a natural mediterranean garden with multiple plants: "
        "lavender, rosemary, olive tree, shrubs, "
        "keep existing architecture unchanged, photorealistic, "
        "consistent lighting, no text, no labels"
    )

    try:
        from garden_ai.image_generation.bfl_provider import inpaint
        inpaint(
            image_path=image_path,
            mask_path=mask_path,
            prompt=prompt,
            out_path=out_path,
            seed=42,
        )
    except RuntimeError as e:
        if "BFL_API_KEY" in str(e):
            print(f"❌ {e}", file=sys.stderr)
            print("   Définir : export BFL_API_KEY='votre_clé'", file=sys.stderr)
            return 1
        raise

    print(f"   ✓ {out_path}")
    print("\n=== Terminé ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
