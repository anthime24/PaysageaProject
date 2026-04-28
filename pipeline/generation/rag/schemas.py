"""
Schémas Pydantic pour valider plante, query et output.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Plant(BaseModel):
    """Schéma d'une plante (entrée)."""
    plant_id: str
    name: str
    type: str = ""
    height_cm: int = 0
    width_cm: int = 0
    density: str = ""
    color: str = ""
    climate: str = ""
    sun_exposure: str = ""
    season: str = ""
    water_needs: str = ""
    zone_hint: str = ""
    style_tags: list[str] = Field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=False)


class Query(BaseModel):
    """Schéma de la requête utilisateur."""
    style: str = ""
    climat: str = ""
    sun_exposure: str = ""
    season: str = ""
    water_constraint: str = ""
    description: str = ""


class OutputMetadata(BaseModel):
    """Métadonnées de sortie."""
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    description: str = ""
    style: str = ""
    climat: str = ""


class RAGOutput(BaseModel):
    """Format de sortie RAG conforme à la spec."""
    metadata: OutputMetadata
    garden: list[dict[str, Any]]
