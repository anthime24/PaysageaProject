from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageFilter


def feather_mask(mask: Image.Image, radius: int = 5) -> Image.Image:
    """
    Applique un flou gaussien doux sur les bords d'un masque binaire.

    Le masque d'entrée doit être en mode "L" avec des valeurs 0/255.
    Le masque de sortie reste en "L" avec des valeurs 0–255 mais
    avec une transition douce sur les bords (utile pour le compositing).
    """
    if mask.mode != "L":
        mask = mask.convert("L")
    if radius <= 0:
        return mask
    # Normaliser en [0,1], flouter, puis remapper en [0,255]
    arr = np.array(mask, dtype=np.float32) / 255.0
    pil = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return blurred


def composite_with_mask(
    original: Union[Image.Image, str, Path],
    generated: Union[Image.Image, str, Path],
    mask: Union[Image.Image, str, Path],
    feather_radius: int = 3,
) -> Image.Image:
    """
    Fusionne original et generated en utilisant mask.

    - Dans la zone blanche du masque : pixels générés
    - Dans la zone noire : pixels originaux
    - Transition douce sur les bords (feathering via feather_mask)

    IMPORTANT:
    - original doit être l'image utilisée comme base pour cet inpaint
      (c.-à-d. l'image d'entrée du call BFL courant), pas la toute première
      photo, afin de préserver les plantes déjà ajoutées tout en évitant
      la dégradation itérative hors masque.
    """
    if not isinstance(original, Image.Image):
        original = Image.open(original).convert("RGB")
    else:
        original = original.convert("RGB")

    if not isinstance(generated, Image.Image):
        generated = Image.open(generated).convert("RGB")
    else:
        generated = generated.convert("RGB")

    if not isinstance(mask, Image.Image):
        mask = Image.open(mask).convert("L")
    else:
        mask = mask.convert("L")

    # S'assurer que toutes les images ont la même taille
    w, h = original.size
    if generated.size != (w, h):
        generated = generated.resize((w, h), Image.LANCZOS)
    if mask.size != (w, h):
        mask = mask.resize((w, h), Image.NEAREST)

    # Feather du masque pour une transition douce
    if feather_radius > 0:
        mask = feather_mask(mask, radius=feather_radius)

    orig_arr = np.array(original, dtype=np.float32)
    gen_arr = np.array(generated, dtype=np.float32)
    alpha = np.array(mask, dtype=np.float32) / 255.0  # 0 = original, 1 = generated
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha_3 = alpha[..., None]

    blended = orig_arr * (1.0 - alpha_3) + gen_arr * alpha_3
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")

