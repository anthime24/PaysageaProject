from __future__ import annotations

from pathlib import Path
from typing import Any, Union, List, Dict

import json
import os

from PIL import Image

from .blend_utils import composite_with_mask
from .mask_manager import MaskManager, MaskResult
from .scene_generator import inpaint
from .utils_rag import load_rag
from .prompt_builder import build_single_plant_inpaint_prompt, build_global_context
from .prompt_with_image import build_prompt_with_image_ref
from .config import BFL_STEPS, BFL_GUIDANCE, BFL_STRENGTH


def _zone_sort_key(zone_hint: str) -> int:
    """Classe les plantes par profondeur : background → midground → foreground."""
    z = (zone_hint or "").lower()
    prefix = z.split("_")[0]
    if prefix.startswith("background"):
        return 0
    if prefix.startswith("midground") or prefix.startswith("middle"):
        return 1
    if prefix.startswith("foreground"):
        return 2
    return 1


def _strength_for_mask(mask_path: Path) -> float:
    """Choix heuristique du strength BFL en fonction de la taille du masque."""
    mask = Image.open(mask_path).convert("L")
    arr = (mask.size[0] * mask.size[1])
    white = (Image.eval(mask, lambda v: 255 if v >= 128 else 0).histogram()[255])
    white_pct = white / max(arr, 1)
    if white_pct < 0.10:
        return min(0.95, BFL_STRENGTH)
    if white_pct < 0.25:
        return min(0.85, BFL_STRENGTH)
    return min(0.75, BFL_STRENGTH)


def generate_garden_plant_by_plant(
    image_path: str | Path,
    rag_json_path: str | Path,
    outputs_dir: str | Path = "outputs",
    external_plantable_zones: list[dict] | None = None,
    external_plantable_mask_path: str | Path | None = None,
    debug: bool = True,
    max_plants: int = 6,
) -> dict:
    """
    Pipeline séquentiel plante par plante.

    Pour chaque plante du RAG (dans l'ordre de profondeur) :
      1. Crée un masque individuel basé sur zone_hint + zones plantables
      2. Appelle BFL inpaint sur l'image courante
      3. Post-fusion : préserve l'image hors masque (à partir de l'image d'entrée de l'étape)
      4. Sauvegarde l'état intermédiaire
      5. Passe à la plante suivante sur l'image résultante
    """
    image_path = Path(image_path)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    steps_dir = outputs_dir / "steps"
    steps_dir.mkdir(exist_ok=True)
    masks_dir = outputs_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Image non trouvée : {image_path}")

    metadata, plants = load_rag(rag_json_path)
    if not plants:
        raise ValueError("Aucune plante dans le RAG pour le pipeline plante par plante.")

    # Trier les plantes par profondeur (background -> midground -> foreground)
    plants_sorted = sorted(
        plants[:max_plants],
        key=lambda p: _zone_sort_key(p.get("zone_hint", "")),
    )

    original_img = Image.open(image_path).convert("RGB")
    current_img_path = image_path  # image courante utilisée comme base pour BFL

    mask_manager = MaskManager(masks_dir=masks_dir)
    already_placed: List[list[int]] = []
    steps: List[Dict[str, Any]] = []

    global_context = build_global_context(metadata) if metadata else ""

    plantable_mask_img: Image.Image | None = None
    if external_plantable_mask_path:
        p = Path(external_plantable_mask_path)
        if not p.is_absolute():
            p = (Path(__file__).resolve().parent.parent / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Masque plantable externe introuvable : {p}")
        plantable_mask_img = Image.open(p).convert("L")

    for idx, plant in enumerate(plants_sorted):
        plant_id = plant.get("plant_id", f"plant_{idx+1:02d}")
        print(f"🌱 Étape {idx+1}/{len(plants_sorted)} — {plant.get('name', plant_id)}")

        # 1. Masque individuel
        mask_result: MaskResult = mask_manager.create_individual_plant_mask(
            image_path=current_img_path,
            plant=plant,
            plant_index=idx,
            already_placed=already_placed,
            plantable_zones_mask=plantable_mask_img,
        )
        mask_path = Path(mask_result.mask_path)
        already_placed.append(mask_result.bbox)

        # 2. Prompt individuel
        surrounding = ", ".join(p.get("name", p.get("plant_id", "")) for p in plants_sorted[:idx]) or ""
        prompt = build_prompt_with_image_ref(
            plant=plant,
            metadata=metadata,
            surrounding_context=surrounding,
            iteration=idx,
            project_root=Path(__file__).resolve().parent.parent,
        )

        # 3. Appel BFL inpaint sur l'image courante
        raw_out = steps_dir / f"step_{idx+1:02d}_{plant_id}_raw.png"
        strength = _strength_for_mask(mask_path)
        # Seed stable mais différente par plante pour diversité
        base_seed = int(os.environ.get("DEBUG_SEED", "42"))
        seed = base_seed + idx

        inpaint(
            image_path=current_img_path,
            mask_path=mask_path,
            prompt=prompt,
            out_path=raw_out,
            seed=seed,
            steps=BFL_STEPS,
            guidance=BFL_GUIDANCE,
            strength=strength,
        )

        # 4. Post-fusion : preserve hors masque, à partir de l'image d'entrée de l'étape
        composite_out = steps_dir / f"step_{idx+1:02d}_{plant_id}.png"
        composed = composite_with_mask(
            original=Image.open(current_img_path).convert("RGB"),
            generated=Image.open(raw_out).convert("RGB"),
            mask=Image.open(mask_path).convert("L"),
            feather_radius=3,
        )
        composed.save(composite_out)
        current_img_path = composite_out

        steps.append(
            {
                "index": idx,
                "plant_id": plant_id,
                "name": plant.get("name", plant_id),
                "mask_path": str(mask_path),
                "raw_path": str(raw_out),
                "composite_path": str(composite_out),
                "bbox": mask_result.bbox,
                "prompt": prompt,
                "seed": seed,
                "strength": strength,
            }
        )

    # Image finale
    final_path = outputs_dir / "final_garden.png"
    Image.open(current_img_path).convert("RGB").save(final_path)

    scene = {
        "mode": "sequential",
        "input_image": str(image_path),
        "final_image": str(final_path),
        "metadata": metadata,
        "global_context": global_context,
        "plants": plants_sorted,
        "steps": steps,
    }

    with open(outputs_dir / "scene_sequential.json", "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2, ensure_ascii=False)

    return scene

