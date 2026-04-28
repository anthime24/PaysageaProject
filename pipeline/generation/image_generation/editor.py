"""
Édition de scène : remove, replace, add plante.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Union

from .mask_manager import MaskManager
from .prompt_builder import build_prompt
from .scene_generator import inpaint


def _maybe_relight_night(
    outputs_dir: Path,
    source_path: Path,
    night_name: str,
    time_of_day: str,
    plants: list[dict] | None = None,
    light_intensity: float = 0.5,
) -> None:
    """Si time_of_day == 'night', appelle relight_to_night."""
    if time_of_day != "night":
        return
    if not source_path.exists():
        print(f"[RELIGHT] Image non trouvée, relight ignoré : {source_path}")
        return
    try:
        from ..utils.relight import relight_to_night
        relight_to_night(
            source_path,
            outputs_dir / night_name,
            light_intensity=light_intensity,
            plants=plants or [],
        )
    except Exception as e:
        print(f"[RELIGHT] Erreur : {e}")


def _load_scene(outputs_dir: Union[str, Path]) -> tuple[dict, Path]:
    """Charge scene.json et retourne (scene_dict, chemin image finale)."""
    outputs_dir = Path(outputs_dir)
    scene_path = outputs_dir / "scene.json"
    if not scene_path.exists():
        raise FileNotFoundError(f"scene.json non trouvé : {scene_path}")

    with open(scene_path, encoding="utf-8") as f:
        scene = json.load(f)

    final_path = Path(scene.get("final_image", str(outputs_dir / "final_garden.png")))
    if not final_path.is_absolute():
        final_path = outputs_dir / final_path
    if not final_path.exists():
        raise FileNotFoundError(f"Image finale non trouvée : {final_path}")

    return scene, final_path


def _get_plant(scene: dict, plant_id: str) -> dict:
    for p in scene.get("plants", []):
        if p.get("plant_id") == plant_id:
            return p
    raise ValueError(f"Plante non trouvée : {plant_id}")


def remove_plant(
    outputs_dir: Union[str, Path],
    plant_id: str,
    time_of_day: str = "day",
    night_light_intensity: float = 0.5,
) -> str:
    """
    Supprime une plante (inpaint avec sol/herbe).

    Returns:
        Chemin de final_garden_edited.png
    """
    scene, image_path = _load_scene(outputs_dir)
    plant = _get_plant(scene, plant_id)
    mask_path = Path(plant["mask_path"])
    if not mask_path.exists():
        raise FileNotFoundError(f"Masque non trouvé : {mask_path}")

    prompt = "clean grass, garden soil, no plant, natural ground, daylight, photorealistic"
    out_path = Path(outputs_dir) / "final_garden_edited.png"

    print(f"   Remove {plant_id}...")
    inpaint(
        image_path=image_path,
        mask_path=mask_path,
        prompt=prompt,
        out_path=out_path,
        seed=100,
    )

    new_plants = [p for p in scene["plants"] if p["plant_id"] != plant_id]
    scene["final_image"] = str(out_path)
    scene["plants"] = new_plants
    with open(Path(outputs_dir) / "scene.json", "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2, ensure_ascii=False)

    _maybe_relight_night(
        Path(outputs_dir), out_path, "final_garden_edited_night.png",
        time_of_day, plants=new_plants, light_intensity=night_light_intensity,
    )
    return str(out_path)


def replace_plant(
    outputs_dir: Union[str, Path],
    plant_id: str,
    new_plant_dict: dict[str, Any],
    time_of_day: str = "day",
    night_light_intensity: float = 0.5,
) -> str:
    """
    Remplace une plante par une autre (même masque, nouveau prompt).

    Returns:
        Chemin de final_garden_edited.png
    """
    scene, image_path = _load_scene(outputs_dir)
    plant = _get_plant(scene, plant_id)
    mask_path = Path(plant["mask_path"])
    if not mask_path.exists():
        raise FileNotFoundError(f"Masque non trouvé : {mask_path}")

    prompt = build_prompt(new_plant_dict)
    out_path = Path(outputs_dir) / "final_garden_edited.png"

    print(f"   Replace {plant_id} par {new_plant_dict.get('name', '?')}...")
    inpaint(
        image_path=image_path,
        mask_path=mask_path,
        prompt=prompt,
        out_path=out_path,
        seed=200,
    )

    new_plant = {
        "plant_id": new_plant_dict.get("plant_id", plant_id),
        "name": new_plant_dict.get("name", "plant"),
        "mask_path": str(mask_path),
        "zone_hint": new_plant_dict.get("zone_hint", plant.get("zone_hint", "")),
        "prompt_used": prompt,
        "bbox": plant.get("bbox", [0, 0, 0, 0]),
    }
    new_plants = [
        new_plant if p["plant_id"] == plant_id else p
        for p in scene["plants"]
    ]
    scene["final_image"] = str(out_path)
    scene["plants"] = new_plants
    with open(Path(outputs_dir) / "scene.json", "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2, ensure_ascii=False)

    _maybe_relight_night(
        Path(outputs_dir), out_path, "final_garden_edited_night.png",
        time_of_day, plants=new_plants, light_intensity=night_light_intensity,
    )
    return str(out_path)


def add_plant(
    outputs_dir: Union[str, Path],
    new_plant_dict: dict[str, Any],
    time_of_day: str = "day",
    night_light_intensity: float = 0.5,
) -> str:
    """
    Ajoute une nouvelle plante (nouveau masque + inpaint).

    Returns:
        Chemin de final_garden_edited.png
    """
    outputs_dir = Path(outputs_dir)
    scene, image_path = _load_scene(outputs_dir)

    plant_id = new_plant_dict.get("plant_id", f"plant_{len(scene['plants']) + 1:02d}")
    zone_hint = new_plant_dict.get("zone_hint", "midground_center")

    mask_manager = MaskManager(masks_dir=outputs_dir / "masks")
    mask_res = mask_manager.create_mask(
        image_path=image_path,
        plant_id=plant_id,
        zone_hint=zone_hint,
    )
    prompt = build_prompt(new_plant_dict)
    out_path = outputs_dir / "final_garden_edited.png"

    print(f"   Add {plant_id}...")
    inpaint(
        image_path=image_path,
        mask_path=mask_res.mask_path,
        prompt=prompt,
        out_path=out_path,
        seed=300,
    )

    new_plant = {
        "plant_id": plant_id,
        "name": new_plant_dict.get("name", "plant"),
        "mask_path": mask_res.mask_path,
        "zone_hint": zone_hint,
        "prompt_used": prompt,
        "bbox": mask_res.bbox,
    }
    scene["plants"] = scene["plants"] + [new_plant]
    scene["final_image"] = str(out_path)
    with open(outputs_dir / "scene.json", "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2, ensure_ascii=False)

    _maybe_relight_night(
        outputs_dir, out_path, "final_garden_edited_night.png",
        time_of_day, plants=scene["plants"], light_intensity=night_light_intensity,
    )
    return str(out_path)
