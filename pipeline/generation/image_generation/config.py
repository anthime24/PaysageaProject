"""
Configuration du module image_generation.

Clé API BFL : définir BFL_API_KEY en variable d'environnement.
"""
from __future__ import annotations

import os

BFL_API_URL = "https://api.bfl.ai/v1/flux-pro-1.0-fill"

# Mode de placement des plantes: "random" (bbox aléatoire) ou "fixed" (zone_hint)
PLACEMENT_MODE = os.environ.get("PLACEMENT_MODE", "fixed")
# Defaults add-only : guidance/steps modérés, strength pour préserver la base
BFL_STEPS = 35
BFL_GUIDANCE = 35
BFL_STRENGTH = 0.92


def get_api_key() -> str:
    """
    Retourne la clé API BFL depuis l'environnement.

    Raises:
        RuntimeError: Si BFL_API_KEY n'est pas définie
    """
    key = os.environ.get("BFL_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "BFL_API_KEY non définie.\n"
            "Définir la variable : export BFL_API_KEY='votre_clé'\n"
            "Ou dans .env / configuration du projet."
        )
    return key
