"""
Génération de jardin par inpainting.

Supporte la génération globale (legacy), plante par plante,
et la suppression d'une plante (inpaint sol/herbe).
"""
from __future__ import annotations

import torch
import numpy as np
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
)
from PIL import Image



def build_plant_prompt(plant: dict) -> str:
    """
    Construit le prompt pour une plante à partir des infos RAG.

    Args:
        plant: dict avec au minimum 'name' ou 'species', optionnellement
               'common_name', 'description', 'height', etc.

    Returns:
        Prompt optimisé pour la génération
    """
    name = plant.get("name") or plant.get("species") or plant.get("common_name", "plant")
    base = f"realistic {name}, natural garden plant, high detail, daylight, photorealistic"
    if plant.get("description"):
        base = f"{base}, {plant['description']}"
    return base


class GardenGenerator:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth",
            torch_dtype=self.dtype,
        )

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=self.dtype,
        ).to(self.device)

    def generate(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        plantable_mask: np.ndarray,
        prompt: str,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """
        Génération globale (legacy) : tout le jardin en un seul pass.
        """
        pil_image = Image.fromarray(image)
        pil_depth = Image.fromarray(depth_map)
        pil_mask = Image.fromarray(plantable_mask)

        result = self.pipe(
            prompt=prompt,
            image=pil_image,
            mask_image=pil_mask,
            control_image=pil_depth,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        return result

    def generate_single_plant(
        self,
        canvas: np.ndarray | Image.Image,
        region_mask: np.ndarray,
        plant: dict | str,
        depth_map: np.ndarray,
        num_inference_steps: int = 35,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        max_side: int = 768,
    ) -> Image.Image:
        """
        Génère une seule plante dans une région du jardin.

        Args:
            canvas: Image courante (numpy RGB ou PIL)
            region_mask: Masque binaire de la zone à remplir (255 = zone)
            plant: Plante (dict RAG) ou string pour le prompt
            depth_map: Carte de profondeur
            num_inference_steps, guidance_scale: Paramètres SD
            seed: Seed pour reproductibilité
            max_side: Côté max pour redimensionnement (0 = pas de resize)

        Returns:
            Image PIL avec la nouvelle plante composée sur le canvas
        """
        if isinstance(canvas, np.ndarray):
            canvas_arr = canvas
        else:
            canvas_arr = np.array(canvas)

        h, w = canvas_arr.shape[:2]
        if max_side > 0 and max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            from PIL import Image as PILImage
            pil_canvas = PILImage.fromarray(canvas_arr).resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            pil_depth = PILImage.fromarray(depth_map).resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            pil_mask = PILImage.fromarray(region_mask).resize((new_w, new_h), PILImage.Resampling.NEAREST)
            need_resize_back = True
        else:
            pil_canvas = Image.fromarray(canvas_arr)
            pil_depth = Image.fromarray(depth_map)
            pil_mask = Image.fromarray(region_mask)
            need_resize_back = False

        if isinstance(plant, str):
            prompt = plant
        else:
            prompt = build_plant_prompt(plant)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            image=pil_canvas,
            mask_image=pil_mask,
            control_image=pil_depth,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        if need_resize_back:
            result = result.resize((w, h), Image.Resampling.LANCZOS)
        return result

    def remove_plant(
        self,
        canvas: np.ndarray | Image.Image,
        plant_mask: np.ndarray,
        depth_map: np.ndarray,
        prompt: str = "garden soil, grass, natural ground, empty garden bed, daylight, photorealistic",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> Image.Image:
        """
        Remplace une plante par du sol/herbe (pour l'action "retirer" au clic).

        Args:
            canvas: Image courante
            plant_mask: Masque de la plante à retirer (255 = zone)
            depth_map: Carte de profondeur
            prompt: Prompt pour remplir la zone (sol/herbe par défaut)
            num_inference_steps, guidance_scale: Paramètres SD
            seed: Seed pour reproductibilité

        Returns:
            Image PIL avec la zone remplie par du sol
        """
        if isinstance(canvas, np.ndarray):
            pil_canvas = Image.fromarray(canvas)
        else:
            pil_canvas = canvas

        pil_depth = Image.fromarray(depth_map)
        pil_mask = Image.fromarray(plant_mask)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            image=pil_canvas,
            mask_image=pil_mask,
            control_image=pil_depth,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        return result
