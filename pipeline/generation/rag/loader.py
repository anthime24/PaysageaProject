"""
Charge les plantes depuis un fichier JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import Plant


def load_plants(plants_path: str | Path) -> list[Plant]:
    """
    Charge et normalise les plantes depuis un fichier JSON.

    Accepte :
    - Liste de plantes : [...]
    - Objet avec clé "garden" : {"garden": [...]}

    Returns:
        Liste de Plant validées
    """
    path = Path(plants_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier plantes non trouvé : {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        raw_plants = data
    elif isinstance(data, dict) and "garden" in data:
        raw_plants = data["garden"]
    else:
        raise ValueError(
            "Format JSON invalide : attendu liste de plantes ou {\"garden\": [...]}"
        )

    plants = []
    for i, p in enumerate(raw_plants):
        if not isinstance(p, dict):
            continue
        try:
            plant = Plant(
                plant_id=str(p.get("plant_id", f"plant_{i:02d}")),
                name=str(p.get("name", "")),
                type=str(p.get("type", "")),
                height_cm=int(p.get("height_cm", 0)),
                width_cm=int(p.get("width_cm", 0)),
                density=str(p.get("density", "")),
                color=str(p.get("color", "")),
                climate=str(p.get("climate", "")),
                sun_exposure=str(p.get("sun_exposure", "")),
                season=str(p.get("season", "")),
                water_needs=str(p.get("water_needs", "")),
                zone_hint=str(p.get("zone_hint", "")),
                style_tags=p.get("style_tags", []) if isinstance(p.get("style_tags"), list) else [],
                reason=str(p.get("reason", "")),
            )
            plants.append(plant)
        except Exception as e:
            continue
    return plants
