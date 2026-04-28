"""
Segmentation du jardin avec SAM (Segment Anything Model).

Le checkpoint doit être téléchargé et placé dans le projet.
Cherche dans : projet/, checkpoints/, ou chemin absolu.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

# Chemins possibles pour le checkpoint SAM
DEFAULT_CHECKPOINT = "sam_vit_h_4b8939.pth"
CHECKPOINT_SEARCH_PATHS = [
    Path.cwd() / DEFAULT_CHECKPOINT,
    Path(__file__).resolve().parent.parent / DEFAULT_CHECKPOINT,
    Path(__file__).resolve().parent.parent / "checkpoints" / DEFAULT_CHECKPOINT,
]


def resolve_sam_checkpoint(checkpoint: str | Path | None = None) -> Path:
    """
    Résout le chemin du checkpoint SAM.

    Raises:
        FileNotFoundError: Si le fichier n'existe nulle part
    """
    if checkpoint is not None:
        p = Path(checkpoint)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"Checkpoint SAM non trouvé : {p}\n"
            "Téléchargez-le depuis : "
            "https://github.com/facebookresearch/segment-anything#model-checkpoints"
        )

    for path in CHECKPOINT_SEARCH_PATHS:
        if path.exists():
            return path

    searched = ", ".join(str(p) for p in CHECKPOINT_SEARCH_PATHS)
    raise FileNotFoundError(
        f"Checkpoint SAM non trouvé. Chemins cherchés : {searched}\n"
        "Téléchargez sam_vit_h_4b8939.pth depuis : "
        "https://github.com/facebookresearch/segment-anything#model-checkpoints"
    )


class GardenSegmenter:
    def __init__(self, sam_checkpoint: str | Path | None = None):
        checkpoint_path = resolve_sam_checkpoint(sam_checkpoint)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint_path))
        sam.to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def segment(self, image_path: str | Path):
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image non trouvée : {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Impossible de charger l'image : {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image)
        return image, masks

    def extract_plantable_mask(self, image, masks):
        h, w, _ = image.shape
        plantable = np.zeros((h, w), dtype=np.uint8)

        for m in masks:
            seg = m["segmentation"]
            y_indices = np.where(seg)[0]

            # heuristique simple : zone basse = sol
            if len(y_indices) > 0 and np.mean(y_indices) > h * 0.55:
                plantable[seg] = 255

        return plantable


if __name__ == "__main__":
    from pathlib import Path
    for p in [Path("data ") / "garden.jpg", Path("data") / "garden.jpg"]:
        if p.exists():
            segmenter = GardenSegmenter()
            image, masks = segmenter.segment(str(p))
            plantable = segmenter.extract_plantable_mask(image, masks)
            cv2.imwrite("plantable_mask.png", plantable)
            print(f"Segmentation OK : {len(masks)} masques, plantable sauvegardé")
            break
    else:
        print("❌ Aucune image trouvée dans data/ ou data /")
