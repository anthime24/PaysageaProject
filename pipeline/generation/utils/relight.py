"""
Post-traitement : transformation jour -> nuit réaliste.

- Exposition + contraste
- Teinte froide (bleu) dans les ombres
- Vignettage
- Spots lumineux chauds autour de chaque plante (halo sur le sol)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image


def relight_to_night(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    strength: float = 0.85,
    light_intensity: float = 0.5,
    plants: list[dict[str, Any]] | None = None,
    add_warm_lights: bool = True,
    seed: int = 42,
) -> Path:
    """
    Transforme une image jour en version nuit réaliste.

    Args:
        input_path: Image source
        output_path: Image de sortie
        strength: Intensité assombrissement (0-1)
        light_intensity: Intensité des spots (0-1)
        plants: Liste des plantes avec bbox pour spots (optionnel)
        add_warm_lights: Ajouter spots lumineux
        seed: Seed pour reproductibilité

    Returns:
        output_path
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"[RELIGHT] Image non trouvée : {input_path}")

    print("[RELIGHT] Chargement image...")
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    # 1. Baisse exposition (gamma)
    gamma = 1.0 + strength * 2.0
    arr = np.power(arr, gamma)

    # 2. Teinte froide (plus de bleu dans les ombres)
    luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    shadow_factor = 1.0 - np.clip(luminance * 1.5, 0, 1)  # plus froid dans les ombres
    arr[:, :, 0] *= 0.85 - shadow_factor * 0.1
    arr[:, :, 1] *= 0.92 - shadow_factor * 0.05
    arr[:, :, 2] = np.minimum(arr[:, :, 2] * (1.15 + shadow_factor * 0.1), 1.0)

    # 3. Contraste doux
    mean_val = arr.mean()
    arr = (arr - mean_val) * 1.08 + mean_val

    # 4. Vignettage
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vignette = 1.0 - 0.25 * (dist / max_dist) ** 2
    arr *= vignette[:, :, np.newaxis]

    arr = np.clip(arr, 0, 1)

    # 5. Spots lumineux chauds
    if add_warm_lights and light_intensity > 0:
        rng = random.Random(seed)

        # Spots autour de chaque plante (halo sur le sol)
        if plants:
            for plant in plants:
                bbox = plant.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                # Centre du halo : sous la plante (sol)
                cx_spot = (x1 + x2) // 2
                cy_spot = min(y2 + (y2 - y1) // 2, h - 1)  # halo en bas
                sigma = max((x2 - x1 + y2 - y1) // 2, 40) * (0.8 + rng.random() * 0.4)
                intensity = light_intensity * rng.uniform(0.04, 0.12)

                yy_arr, xx_arr = np.ogrid[:h, :w]
                gauss = np.exp(-((xx_arr - cx_spot) ** 2 + (yy_arr - cy_spot) ** 2) / (2 * sigma**2))
                arr[:, :, 0] += gauss * intensity * 1.3
                arr[:, :, 1] += gauss * intensity * 0.95
                arr[:, :, 2] += gauss * intensity * 0.4

        # Quelques spots aléatoires supplémentaires (éclairage jardin)
        n_extra = rng.randint(2, 5)
        for _ in range(n_extra):
            cx = rng.randint(w // 8, 7 * w // 8)
            cy = rng.randint(h // 2, 7 * h // 8)  # plutôt en bas (sol)
            sigma = rng.uniform(30, 80)
            intensity = light_intensity * rng.uniform(0.02, 0.06)
            yy_arr, xx_arr = np.ogrid[:h, :w]
            gauss = np.exp(-((xx_arr - cx) ** 2 + (yy_arr - cy) ** 2) / (2 * sigma**2))
            arr[:, :, 0] += gauss * intensity * 1.2
            arr[:, :, 1] += gauss * intensity * 0.9
            arr[:, :, 2] += gauss * intensity * 0.4

        arr = np.clip(arr, 0, 1)

    out_arr = (arr * 255).astype(np.uint8)
    Image.fromarray(out_arr).save(output_path)
    print(f"[RELIGHT] Sauvegardé : {output_path}")
    return output_path
