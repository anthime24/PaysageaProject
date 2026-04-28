"""
Provider MOCK - génère faux inpaint visuel sans API.

Rendu final : texture végétale dans la zone (masque ou bbox), SANS aucun texte.
Les labels existent uniquement dans preview_boxes.png (debug).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageFilter


def _fake_vegetation_inpaint(
    img: np.ndarray,
    mask: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Applique une texture végétale (bruit vert + flou + variation) dans la zone masquée.
    Pas de texte, pas de labels.
    """
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]

    # Bruit vert/brun organique
    noise_r = rng.normal(0.5, 0.15, (h, w)).astype(np.float32)
    noise_g = rng.normal(0.6, 0.12, (h, w)).astype(np.float32)
    noise_b = rng.normal(0.3, 0.1, (h, w)).astype(np.float32)

    # Variations par zone (simule feuillage)
    yy, xx = np.ogrid[:h, :w]
    freq = 0.02 + rng.random() * 0.03
    variation = np.sin(xx * freq) * np.cos(yy * freq) * 0.1
    noise_r = np.clip(noise_r + variation, 0, 1)
    noise_g = np.clip(noise_g + variation * 1.2, 0, 1)
    noise_b = np.clip(noise_b + variation * 0.5, 0, 1)

    # Combiner en texture végétale
    texture = np.stack([noise_r, noise_g, noise_b], axis=-1)
    texture = (texture * 255).astype(np.uint8)

    # Lisser la texture (feuillage flou)
    tex_pil = Image.fromarray(texture)
    tex_pil = tex_pil.filter(ImageFilter.GaussianBlur(radius=3))
    tex_pil = tex_pil.filter(ImageFilter.SMOOTH_MORE)
    texture = np.array(tex_pil)

    # Mélanger avec l'image originale en bordure (anti-aliasing)
    mask_f = mask.astype(np.float32) / 255.0
    if mask_f.ndim == 2:
        mask_f = mask_f[:, :, np.newaxis]
    # Adoucir les bords du masque
    mask_pil = Image.fromarray((mask_f[:, :, 0] * 255).astype(np.uint8))
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=2))
    mask_f = np.array(mask_pil).astype(np.float32) / 255.0
    if mask_f.ndim == 2:
        mask_f = mask_f[:, :, np.newaxis]

    out = img.astype(np.float32) * (1 - mask_f) + texture.astype(np.float32) * mask_f
    return np.clip(out, 0, 255).astype(np.uint8)


def inpaint_mock(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    prompt: str,
    out_path: Union[str, Path],
    plant_name: str | None = None,
    bbox: list[int] | None = None,
    **kwargs,
) -> None:
    """
    Faux inpaint : applique texture végétale dans la zone masque/bbox.
    AUCUN TEXTE dans l'image finale.
    """
    print("   [MOCK MODE] Génération visuelle simulée (texture végétale)")
    src = Path(image_path).resolve()
    dst = Path(out_path).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(src).convert("RGB")
    img_arr = np.array(img)
    mask_img = Image.open(mask_path).convert("L")

    if mask_img.size != img.size:
        mask_img = mask_img.resize(img.size, Image.Resampling.NEAREST)
    mask_arr = np.array(mask_img)

    # Utiliser masque si valide, sinon bbox
    if np.any(mask_arr > 0):
        mask_use = mask_arr
    else:
        # Fallback bbox : créer masque depuis bbox
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            mask_use = np.zeros((img_arr.shape[0], img_arr.shape[1]), dtype=np.uint8)
            mask_use[y1:y2, x1:x2] = 255
        else:
            # Pas de zone valide : copier l'image telle quelle
            img.save(dst)
            return

    seed = kwargs.get("seed", 42)
    result = _fake_vegetation_inpaint(img_arr, mask_use, seed=seed)
    Image.fromarray(result).save(dst)
    if plant_name:
        print(f"   [MOCK] Zone modifiée (sans label) : {plant_name}")


def create_preview_boxes(
    image_path: Union[str, Path],
    plants: list[dict],
    out_path: Union[str, Path],
) -> None:
    """
    Debug uniquement : dessine bbox + labels sur l'image.
    Ce fichier ne doit JAMAIS être utilisé comme image finale.
    """
    from PIL import ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        font = ImageFont.load_default()

    colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    for i, plant in enumerate(plants):
        bbox = plant.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = plant.get("plant_id", f"plant_{i}") + " - " + plant.get("name", "")
        draw.text((x1 + 5, y1 + 5), label, fill=color, font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"   [DEBUG] Preview sauvegardé : {out_path}")
