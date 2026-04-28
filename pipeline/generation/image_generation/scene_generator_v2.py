"""
Routeur de génération — Garden AI v3.
Point d'entrée unique depuis l'UI. Délègue entièrement à scene_generator.py.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .scene_generator import generate_scene


def dispatch_generation(
    image_path: str | Path,
    rag_json_path: str | Path,
    outputs_dir: str | Path = "outputs",
    mode: str = "sequential",
    time_of_day: str = "day",
    night_light_intensity: float = 0.5,
    max_plants: int = 6,
    external_plantable_mask_path: str | Path | None = None,
    debug: bool = True,
) -> dict[str, Any]:
    return generate_scene(
        image_path=image_path,
        rag_json_path=rag_json_path,
        outputs_dir=outputs_dir,
        mode=mode,
        time_of_day=time_of_day,
        night_light_intensity=night_light_intensity,
        debug=debug,
        max_plants=max_plants,
        external_plantable_mask_path=external_plantable_mask_path,
    )
