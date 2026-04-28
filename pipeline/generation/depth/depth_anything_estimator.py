"""
Estimateur de profondeur basé sur Depth-Anything (LiheYoung/depth_anything_*).

Utilise le dépôt Depth-Anything pour une meilleure estimation de la profondeur
(en particulier paysages/jardins). Interface compatible avec DepthEstimator :
predict(image) -> carte (H, W) uint8 0-255.

Configuration:
  - DEPTH_ANYTHING_ROOT : chemin vers le dossier Depth-Anything-main (défaut: ~/Downloads/Depth-Anything-main)
  - Option encoder : vitl (défaut), vitb, vits (vitl = meilleure qualité, plus lent)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

# Encoder par défaut : vitl = meilleure qualité pour paysages
_DEFAULT_ENCODER = "vitl"
_PRETRAINED_PREFIX = "LiheYoung/depth_anything_{}14"


def _get_depth_anything_root() -> Path:
    root = os.environ.get("DEPTH_ANYTHING_ROOT", "")
    if not root:
        root = Path.home() / "Downloads" / "Depth-Anything-main"
    return Path(root).resolve()


def _ensure_depth_anything_import():
    """Ajoute DEPTH_ANYTHING_ROOT au path et importe depth_anything (dpt, transform)."""
    root = _get_depth_anything_root()
    if not root.exists():
        raise FileNotFoundError(
            f"Depth-Anything non trouvé : {root}. "
            "Définir DEPTH_ANYTHING_ROOT ou placer le dossier Depth-Anything-main dans ~/Downloads."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


class DepthAnythingEstimator:
    """
    Estimation de profondeur via Depth-Anything (ViT-L par défaut).
    Même interface que DepthEstimator : predict(image) -> (H, W) uint8 0-255.
    """

    def __init__(
        self,
        encoder: str = _DEFAULT_ENCODER,
        depth_anything_root: Union[str, Path, None] = None,
    ):
        """
        Args:
            encoder: 'vits' | 'vitb' | 'vitl' (vitl = meilleure qualité).
            depth_anything_root: Chemin vers Depth-Anything-main (sinon DEPTH_ANYTHING_ROOT / défaut).
        """
        self.encoder = encoder
        self._root = Path(depth_anything_root) if depth_anything_root else _get_depth_anything_root()
        self._model = None
        self._transform = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self) -> None:
        _ensure_depth_anything_import()
        # Charger le modèle depuis le dépôt (torch.hub peut nécessiter cwd = root)
        prev_cwd = os.getcwd()
        try:
            if str(self._root) not in sys.path:
                sys.path.insert(0, str(self._root))
            os.chdir(self._root)
            from depth_anything.dpt import DepthAnything
            from torchvision.transforms import Compose
            import cv2
            from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

            self._cv2 = cv2
            self._Compose = Compose
            self._Resize = Resize
            self._NormalizeImage = NormalizeImage
            self._PrepareForNet = PrepareForNet
            self._DepthAnything = DepthAnything

            model = DepthAnything.from_pretrained(
                _PRETRAINED_PREFIX.format(self.encoder)
            ).to(self._device).eval()

            transform = Compose([
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
            self._model = model
            self._transform = transform
        finally:
            os.chdir(prev_cwd)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Prédit la carte de profondeur pour une image RGB.

        Args:
            image: numpy (H, W, 3) RGB, valeurs 0-255 (uint8 ou float).

        Returns:
            depth_map: (H, W) uint8, 0 = loin, 255 = proche (même convention que MiDaS normalisé).
        """
        if self._model is None or self._transform is None:
            self._load_model()
        cv2 = self._cv2
        h, w = image.shape[:2]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        # Depth-Anything attend RGB [0,1]
        if image.shape[2] == 3:
            rgb = image
        else:
            rgb = image[:, :, :3].copy()

        sample = self._transform({"image": rgb})["image"]
        x = torch.from_numpy(sample).unsqueeze(0).to(self._device)
        with torch.no_grad():
            depth = self._model(x)
        depth = F.interpolate(
            depth.unsqueeze(1), (h, w), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
        # Normaliser 0-255 (comme DepthEstimator MiDaS)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min < 1e-6:
            depth_norm = np.zeros_like(depth, dtype=np.float32)
        else:
            depth_norm = (depth - d_min) / (d_max - d_min) * 255.0
        return depth_norm.astype(np.uint8)


def get_depth_estimator(use_depth_anything: bool | None = None) -> "object":
    """
    Retourne un estimateur de profondeur.

    Si use_depth_anything=True ou USE_DEPTH_ANYTHING=1 : utilise Depth-Anything (si disponible).
    Sinon : utilise MiDaS (DepthEstimator).

    Returns:
        DepthAnythingEstimator ou DepthEstimator (même interface .predict(image)).
    """
    if use_depth_anything is None:
        use_depth_anything = os.environ.get("USE_DEPTH_ANYTHING", "").lower() in ("1", "true", "yes")
    if use_depth_anything:
        try:
            return DepthAnythingEstimator()
        except Exception as e:
            import warnings
            from .depth_estimation import DepthEstimator
            warnings.warn(
                f"Depth-Anything indisponible ({e}), fallback MiDaS.",
                UserWarning,
                stacklevel=2,
            )
            return DepthEstimator()
    from .depth_estimation import DepthEstimator
    return DepthEstimator()
