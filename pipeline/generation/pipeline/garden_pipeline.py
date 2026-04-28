"""
Pipeline principal : image de base + liste de plantes → jardin généré + masques individuels.

Conçu pour être branché au RAG plus tard. Pour l'instant, la liste de plantes
est passée en entrée (mock ou JSON depuis vos collègues).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from depth import get_depth_estimator
from generation.garden_generation import GardenGenerator
from segmentation.sam_segmentation import GardenSegmenter
from utils.image_utils import save_image
from utils.region_splitter import split_plantable_mask


@dataclass
class PlantPlacement:
    """Placement d'une plante avec son masque cliquable."""

    plant: dict  # infos RAG (name, species, etc.)
    mask: np.ndarray  # masque binaire (H, W) - zone cliquable
    region_index: int


@dataclass
class GardenResult:
    """
    Résultat du pipeline : image finale + masques individuels.

    Permet au frontend de :
    - Afficher l'image générée
    - Créer des zones cliquables par plante (via plant_placements[].mask)
    - Savoir quelle plante est où (plant_placements[].plant)
    """

    final_image: np.ndarray  # (H, W, 3) RGB
    plant_placements: list[PlantPlacement] = field(default_factory=list)
    plantable_mask: np.ndarray | None = None
    depth_map: np.ndarray | None = None

    def save(
        self,
        output_dir: Union[str, Path],
        base_name: str = "garden",
    ) -> dict[str, Path]:
        """
        Sauvegarde l'image finale et tous les masques.

        Returns:
            Chemins des fichiers créés (pour le frontend)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Image finale
        final_path = output_dir / f"{base_name}.png"
        save_image(self.final_image, final_path)
        paths["final_image"] = final_path

        # Masque combiné (optionnel) : chaque plante a une valeur 1, 2, 3...
        if self.plant_placements:
            labeled_mask = np.zeros(
                (self.final_image.shape[0], self.final_image.shape[1]),
                dtype=np.uint8,
            )
            for i, placement in enumerate(self.plant_placements):
                labeled_mask[placement.mask > 0] = i + 1

            labeled_path = output_dir / f"{base_name}_labeled_mask.png"
            save_image(labeled_mask, labeled_path)
            paths["labeled_mask"] = labeled_path

        # Masques individuels (pour debug ou usage alternatif)
        masks_dir = output_dir / "plant_masks"
        masks_dir.mkdir(exist_ok=True)
        paths["plant_masks"] = []
        for i, placement in enumerate(self.plant_placements):
            safe_name = "".join(
                c for c in str(placement.plant.get("name", "plant"))
                if c.isalnum() or c in "_-"
            ) or "plant"
            mask_path = masks_dir / f"plant_{i}_{safe_name}.png"
            save_image(placement.mask, mask_path)
            paths["plant_masks"].append(mask_path)

        # Manifest JSON-friendly pour le frontend
        manifest = {
            "final_image": str(paths["final_image"]),
            "labeled_mask": str(paths.get("labeled_mask", "")),
            "plants": [
                {
                    "index": i,
                    "name": p.plant.get("name") or p.plant.get("species", "plant"),
                    "mask_file": str(paths["plant_masks"][i]),
                }
                for i, p in enumerate(self.plant_placements)
            ],
        }

        import json
        manifest_path = output_dir / f"{base_name}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        paths["manifest"] = manifest_path

        return paths


class GardenPipeline:
    """
    Pipeline complet : photo de base + liste de plantes → jardin + masques.

    Flux :
    1. Charger l'image de base (fond du jardin)
    2. Segmentation (SAM) → masque plantable
    3. Estimation de profondeur
    4. Découpage du masque en N régions (N = nombre de plantes)
    5. Ordre de rendu (arrière-plan → premier plan)
    6. Génération plante par plante (inpainting)
    7. Sortie : image finale + masques individuels
    """

    def __init__(
        self,
        segmenter: GardenSegmenter | None = None,
        depth_estimator=None,
        generator: GardenGenerator | None = None,
    ):
        self.segmenter = segmenter or GardenSegmenter()
        self.depth_estimator = depth_estimator or get_depth_estimator()
        self.generator = generator or GardenGenerator()

    def run(
        self,
        image: Union[str, Path, np.ndarray],
        plants: list[dict | str],
        depth_weight: float = 0.3,
        max_side: int = 768,
    ) -> GardenResult:
        """
        Exécute le pipeline complet.

        Args:
            image: Chemin ou array de l'image de base (fond du jardin)
            plants: Liste de plantes (dict RAG ou strings). Chaque élément :
                    - dict: {"name": "lavande", "species": "Lavandula", ...}
                    - str: prompt brut
            depth_weight: Poids de la profondeur pour le découpage des régions
            max_side: Côté max pour la génération SD (0 = pas de resize)

        Returns:
            GardenResult avec image finale et plant_placements (masques cliquables)

        Raises:
            FileNotFoundError: Image ou checkpoint SAM non trouvé
            ValueError: Image invalide
        """
        # 1. Charger l'image + 2. Segmentation (SAM attend un chemin)
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image non trouvée : {path}")
            base_image, masks = self.segmenter.segment(str(image))
        else:
            base_image = np.asarray(image)
            if base_image.ndim == 2:
                base_image = np.stack([base_image] * 3, axis=-1)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                Image.fromarray(base_image).save(tmp.name)
                base_image, masks = self.segmenter.segment(tmp.name)

        plantable_mask = self.segmenter.extract_plantable_mask(base_image, masks)

        if not plants:
            return GardenResult(
                final_image=base_image,
                plant_placements=[],
                plantable_mask=plantable_mask,
                depth_map=None,
            )

        # 3. Profondeur
        depth_map = self.depth_estimator.predict(base_image)

        # 4. Découpage en régions
        n_plants = len(plants)
        region_masks = split_plantable_mask(
            plantable_mask, depth_map, n_plants, depth_weight=depth_weight
        )

        # 5. Ordre de rendu (fond → avant)
        ordered_masks, order_indices = self._order_regions_with_indices(
            region_masks, depth_map
        )
        ordered_plants = [plants[i] for i in order_indices]

        # 6. Génération plante par plante
        canvas = base_image.copy()
        plant_placements: list[PlantPlacement] = []

        for i, (region_mask, plant) in enumerate(zip(ordered_masks, ordered_plants)):
            if region_mask.max() == 0:
                continue

            result_pil = self.generator.generate_single_plant(
                canvas=canvas,
                region_mask=region_mask,
                plant=plant,
                depth_map=depth_map,
                seed=42 + i,
                max_side=max_side,
            )
            canvas = np.array(result_pil)

            plant_dict = plant if isinstance(plant, dict) else {"name": str(plant)}
            plant_placements.append(
                PlantPlacement(plant=plant_dict, mask=region_mask.copy(), region_index=i)
            )

        return GardenResult(
            final_image=canvas,
            plant_placements=plant_placements,
            plantable_mask=plantable_mask,
            depth_map=depth_map,
        )

    def _order_regions_with_indices(
        self,
        region_masks: list[np.ndarray],
        depth_map: np.ndarray,
    ) -> tuple[list[np.ndarray], list[int]]:
        """Régions triées par profondeur (fond → avant) + indices originaux."""
        depths = []
        for mask in region_masks:
            pixels = np.argwhere(mask > 0)
            mean_depth = (
                depth_map[pixels[:, 0], pixels[:, 1]].mean() if len(pixels) > 0 else 0
            )
            depths.append(mean_depth)

        order = np.argsort(depths)[::-1]
        ordered_masks = [region_masks[i] for i in order]
        order_indices = order.tolist()
        return ordered_masks, order_indices

    def remove_plant(
        self,
        result: GardenResult,
        plant_index: int,
    ) -> GardenResult:
        """
        Retire une plante du jardin (remplace par du sol/herbe).

        Utile quand l'utilisateur clique sur une plante pour la supprimer.

        Args:
            result: Résultat précédent du pipeline
            plant_index: Index de la plante à retirer (0, 1, 2...)

        Returns:
            Nouveau GardenResult sans cette plante
        """
        if plant_index < 0 or plant_index >= len(result.plant_placements):
            raise ValueError(
                f"Index plante invalide : {plant_index} "
                f"(doit être entre 0 et {len(result.plant_placements) - 1})"
            )
        if result.depth_map is None:
            raise ValueError("depth_map requis pour remove_plant")

        placement = result.plant_placements[plant_index]
        new_image = self.generator.remove_plant(
            canvas=result.final_image,
            plant_mask=placement.mask,
            depth_map=result.depth_map,
            seed=100 + plant_index,
        )
        new_image_arr = np.array(new_image)

        new_placements = [
            p for i, p in enumerate(result.plant_placements) if i != plant_index
        ]
        return GardenResult(
            final_image=new_image_arr,
            plant_placements=new_placements,
            plantable_mask=result.plantable_mask,
            depth_map=result.depth_map,
        )
