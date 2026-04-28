"""
Découpe du masque plantable en N régions (une par plante).

Utilise k-means sur les coordonnées des pixels plantables pour créer
des zones spatiales distinctes. Tient compte de la profondeur pour
placer les plantes (grandes en arrière-plan, petites au premier plan).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def split_plantable_mask(
    plantable_mask: np.ndarray,
    depth_map: np.ndarray,
    n_regions: int,
    depth_weight: float = 0.3,
) -> list[np.ndarray]:
    """
    Découpe le masque plantable en N régions pour placer N plantes.

    Utilise k-means sur (x, y, depth_normalized) pour créer des zones
    qui respectent la profondeur (plantes plus hautes en arrière-plan).

    Args:
        plantable_mask: Masque binaire (H, W), 255 = plantable
        depth_map: Carte de profondeur (H, W)
        n_regions: Nombre de régions ( = nombre de plantes)
        depth_weight: Poids de la profondeur vs position (0 = ignorer profondeur)

    Returns:
        Liste de N masques binaires (H, W), un par région
    """
    h, w = plantable_mask.shape
    coords = np.argwhere(plantable_mask > 0)  # (N_pixels, 2) -> (y, x)

    if len(coords) == 0:
        return [np.zeros_like(plantable_mask) for _ in range(n_regions)]

    if n_regions == 1:
        return [plantable_mask.copy()]

    # Features pour le clustering : (y, x) + profondeur normalisée
    y_vals = coords[:, 0].astype(np.float32) / h
    x_vals = coords[:, 1].astype(np.float32) / w

    depth_vals = depth_map[coords[:, 0], coords[:, 1]].astype(np.float32)
    depth_vals = depth_vals / (depth_vals.max() + 1e-6)

    features = np.column_stack([y_vals, x_vals, depth_weight * depth_vals])

    kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    region_masks = []
    for k in range(n_regions):
        mask = np.zeros((h, w), dtype=np.uint8)
        region_pixels = coords[labels == k]
        if len(region_pixels) > 0:
            mask[region_pixels[:, 0], region_pixels[:, 1]] = 255
        region_masks.append(mask)

    return region_masks


def order_regions_by_depth(
    region_masks: list[np.ndarray],
    depth_map: np.ndarray,
) -> list[np.ndarray]:
    """
    Trie les régions par profondeur moyenne (du fond vers l'avant).

    Les régions plus "lointaines" (profondeur élevée) sont rendues en premier,
    celles du premier plan en dernier (pour un compositing correct).
    """
    depths = []
    for mask in region_masks:
        pixels = np.argwhere(mask > 0)
        if len(pixels) > 0:
            mean_depth = depth_map[pixels[:, 0], pixels[:, 1]].mean()
        else:
            mean_depth = 0
        depths.append(mean_depth)

    order = np.argsort(depths)[::-1]  # du plus loin au plus proche
    return [region_masks[i] for i in order]
