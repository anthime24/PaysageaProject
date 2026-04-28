"""
prompt_with_image.py — Enrichit les prompts BFL avec les images de référence.

Pour chaque plante qui a un image_path :
1. Encode l'image en base64
2. Appelle Claude Vision pour décrire la plante visuellement
3. Injecte la description dans le prompt BFL

Utilisé par plant_by_plant_generator.py si use_image_refs=True.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any


def _encode_image(image_path: str | Path) -> tuple[str, str]:
    """Encode une image en base64. Retourne (base64_data, media_type)."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def describe_plant_image(image_path: str | Path, plant_name: str) -> str:
    """
    Appelle Claude Vision pour décrire visuellement une plante.
    Retourne une description en anglais pour le prompt BFL.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()

        img_data, media_type = _encode_image(image_path)

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Describe this {plant_name} plant in 1-2 sentences for a photorealistic image generation prompt. "
                                "Focus on: leaf shape/color, flower color/shape, overall form, height impression. "
                                "Be specific and visual. English only. No intro, just the description."
                            ),
                        },
                    ],
                }
            ],
        )
        return message.content[0].text.strip()

    except Exception as e:
        print(f"[prompt_with_image] Claude Vision failed for {plant_name}: {e}")
        return ""


def build_prompt_with_image_ref(
    plant: dict[str, Any],
    metadata: dict[str, Any],
    surrounding_context: str = "",
    iteration: int = 0,
    project_root: str | Path | None = None,
) -> str:
    """
    Construit un prompt BFL enrichi avec description visuelle depuis l'image de référence.

    Si image_path existe et est valide → décrit via Claude Vision → prompt très précis.
    Sinon → fallback sur prompt_builder.build_single_plant_inpaint_prompt().
    """
    from .prompt_builder import build_single_plant_inpaint_prompt, _get_visual

    name = plant.get("name") or plant.get("species") or "garden plant"
    image_path = plant.get("image_path", "")

    # Résoudre le chemin relatif depuis la racine du projet
    if image_path and project_root:
        full_path = Path(project_root) / image_path
    elif image_path:
        full_path = Path(image_path)
    else:
        full_path = None

    # Obtenir la description visuelle
    visual_description = ""
    if full_path and full_path.exists():
        print(f"[prompt_with_image] 🖼️  Image trouvée pour {name}, appel Claude Vision...")
        visual_description = describe_plant_image(full_path, name)
        if visual_description:
            print(f"[prompt_with_image] ✅ Description: {visual_description[:80]}...")
    else:
        # Fallback sur la visual DB
        visual_description = _get_visual(plant)
        print(f"[prompt_with_image] ℹ️  Pas d'image pour {name}, fallback visual DB.")

    # Construction du prompt
    color = (plant.get("color") or "").replace("_", " ")
    color_hint = ""
    if color and visual_description and color.lower() not in visual_description.lower():
        color_hint = f"Dominant color: {color}. "

    style = (
        metadata.get("style")
        or metadata.get("climat")
        or metadata.get("climate")
        or ""
    )
    style_desc = f"Maintain {style} garden aesthetic. " if style else ""

    surrounding = ""
    if surrounding_context and iteration > 0:
        surrounding = f"Harmonize with existing plants: {surrounding_context}. "

    prompt = (
        f"Add a single {name} planted naturally in the ground, "
        f"ONLY inside the white masked area. "
        f"{visual_description}. "
        f"{color_hint}"
        f"Photorealistic, match existing garden lighting, perspective and shadows. "
        f"Roots firmly in the soil, no floating effect, no dark halo at base. "
        f"{style_desc}"
        f"{surrounding}"
        f"CRITICAL: Do NOT modify ANYTHING outside the masked area. "
        f"No text, no labels, no people."
    )
    return " ".join(prompt.split())
