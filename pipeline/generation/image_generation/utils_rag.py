"""
Utilitaires pour normaliser le JSON RAG.

Fonction load_rag(path) -> (metadata, plants) qui accepte:
- dict avec clé "garden"
- liste directe de plantes
- ignore "image_report" et autres clés non utilisées
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union


# Clés acceptées par l'API inpaint (jamais: plant_name, plant_id, zone_hint, bbox)
ALLOWED_INPAINT_KWARGS = {"seed", "steps", "guidance", "strength"}


def _extract_plants(data: Any) -> tuple[list[dict] | None, str]:
    """
    Extrait la liste de plantes et le format détecté.
    Returns: (plants_list, format_label) avec format_label in ("A", "B", "C")
    """
    if isinstance(data, list):
        return data, "B"
    if isinstance(data, dict):
        for key in ["garden", "jardin", "plants", "recommendations", "results", "items"]:
            val = data.get(key)
            if isinstance(val, list):
                return val, "A" if key in ("garden", "jardin") else "C"
    return None, "?"


def _extract_metadata(data: Any) -> dict:
    """Extrait metadata depuis un dict."""
    if not isinstance(data, dict):
        return {}
    # Support clé "metadata" ou "infos" (format collègue RAG)
    for meta_key in ("metadata", "infos"):
        if meta_key in data and isinstance(data[meta_key], dict):
            return dict(data[meta_key])
    result = {}
    for key in ["style", "climat", "climate", "season", "sun_exposure", "water_constraint", "description"]:
        if key in data:
            result[key] = data[key]
    return result


def validate_rag_schema(plants: list[dict]) -> None:
    """
    Valide et log le schéma RAG.
    - Clés manquantes critiques (plant_id, name)
    - Warning si zone_hint manquant
    """
    keys_ok = True
    for i, p in enumerate(plants):
        if not isinstance(p, dict):
            continue
        if not p.get("plant_id"):
            print(f"[RAG] ⚠ plant[{i}] sans plant_id")
            keys_ok = False
        if not p.get("name") and not p.get("species"):
            print(f"[RAG] ⚠ plant[{i}] sans name/species")
            keys_ok = False
        if not p.get("zone_hint"):
            print(f"[RAG] ⚠ plant[{i}] sans zone_hint (utilisera défaut)")
    if keys_ok:
        print("[RAG] keys ok")


def load_rag(path: Union[str, Path]) -> tuple[dict, list[dict]]:
    """
    Charge et normalise le JSON RAG.

    Accepte:
    - dict avec clé "garden" (ex: jardin_complet.json)
    - liste directe de plantes
    - ignore "image_report" et autres clés

    Returns:
        (metadata, plants) - plants normalisés avec plant_id, name, zone_hint, etc.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier RAG non trouvé : {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    metadata = _extract_metadata(data) if isinstance(data, dict) else {}

    plants_raw, format_label = _extract_plants(data)
    if plants_raw is None:
        keys = list(data.keys()) if isinstance(data, dict) else "liste"
        print(f"[RAG] format détecté: ?, 0 plantes (clés racine: {keys})")
        raise ValueError(
            f"Aucune liste de plantes trouvée. Clés: {keys}\n"
            "Attendu: 'garden', 'plants', ou liste directe"
        )

    if not isinstance(plants_raw, list):
        raise ValueError(f"plants doit être une liste, reçu: {type(plants_raw)}")

    if len(plants_raw) == 0:
        keys = list(data.keys()) if isinstance(data, dict) else "liste"
        print(f"[RAG] format détecté: {format_label}, 0 plantes (clés racine: {keys})")
    else:
        print(f"[RAG] format détecté: {format_label}, {len(plants_raw)} plantes")

    plants = []
    for i, plant in enumerate(plants_raw):
        if not isinstance(plant, dict):
            continue
        plant_id = plant.get("plant_id") or f"plant_{i+1:02d}"
        name = plant.get("name") or plant.get("species") or "plant"
        normalized = {
            "plant_id": plant_id,
            "name": name,
            "type": plant.get("type", ""),
            "height_cm": plant.get("height_cm", 0),
            "width_cm": plant.get("width_cm", 0),
            "density": plant.get("density", "medium"),
            "color": plant.get("color", ""),
            "climate": plant.get("climate", ""),
            "sun_exposure": plant.get("sun_exposure", ""),
            "season": plant.get("season", ""),
            "water_needs": plant.get("water_needs", ""),
            "zone_hint": plant.get("zone_hint", "midground_center"),
            "style_tags": plant.get("style_tags", []),
            "reason": plant.get("reason", ""),
        }
        for k in ["soil_preference", "maintenance_level", "price_range", "colors", "visual_signature", "image_refs"]:
            if k in plant and plant[k] is not None:
                normalized[k] = plant[k]
        plants.append(normalized)

    validate_rag_schema(plants)
    return metadata, plants


def load_rag_output(path: Union[str, Path]) -> dict[str, Any]:
    """
    Alias pour compatibilité: retourne {"metadata": ..., "garden": ...}.
    Préférer load_rag(path) -> (metadata, plants).
    """
    metadata, plants = load_rag(path)
    return {"metadata": metadata, "garden": plants}
