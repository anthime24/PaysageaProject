"""
Utilitaires image pour le pipeline Garden AI.

Responsabilité unique : chargement, sauvegarde et redimensionnement d'images.
Séparation stricte des responsabilités pour faciliter les tests unitaires.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Charge une image depuis un fichier.

    Returns:
        Tableau numpy RGB (H, W, 3), dtype uint8.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image non trouvée : {path}")
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(arr: np.ndarray, path: Union[str, Path]) -> None:
    """
    Sauvegarde un tableau numpy en image (PNG ou JPG).

    Accepte grayscale (H, W) ou RGB (H, W, 3).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        Image.fromarray(arr.astype(np.uint8)).save(path)
    else:
        Image.fromarray(arr.astype(np.uint8)).save(path)


def resize_to_shape(
    img: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """
    Redimensionne l'image à la taille cible (bilinear).

    Utile pour ramener les sorties des modèles (ex: depth) à la taille originale.
    """
    pil = Image.fromarray(img)
    pil = pil.resize((target_width, target_height), Image.Resampling.BILINEAR)
    return np.array(pil)


def resize_to_max_side(img: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """
    Redimensionne en conservant le ratio (côté max = max_side).

    Réduit la charge GPU/CPU pour les modèles tout en gardant les proportions.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    pil = Image.fromarray(img)
    pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.array(pil)
