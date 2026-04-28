"""
Retrieval + filtres + scoring.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import unicodedata

if TYPE_CHECKING:
    from .schemas import Plant, Query


def _normalize(s: str) -> str:
    """Normalise pour comparaison souple : accents, majuscules, espaces."""
    s = unicodedata.normalize("NFD", (s or "").lower().strip())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace(" ", "_").replace("-", "_")


def _climate_match(plant_climate: str, query_climat: str) -> bool:
    if not query_climat:
        return True
    return _normalize(plant_climate) == _normalize(query_climat)


def _sun_match(plant_sun: str, query_sun: str) -> bool:
    if not query_sun:
        return True
    # Gérer les variantes : "plein_soleil", "Plein soleil", "soleil"
    p = _normalize(plant_sun)
    q = _normalize(query_sun)
    if p == q:
        return True
    # "soleil" matche "plein_soleil"
    if q in p or p in q:
        return True
    return False


def _season_match(plant_season: str, query_season: str) -> bool:
    if not query_season:
        return True
    p = _normalize(plant_season)
    q = _normalize(query_season)
    if p == "toutes_saisons":
        return True
    return p == q


# Ordre eau : faible < moyen < fort
WATER_ORDER = {"faible": 0, "moyen": 1, "modere": 1, "fort": 2, "eleve": 2}


def _water_compatible(plant_water: str, constraint: str) -> bool:
    WATER_ORDER_LOCAL = WATER_ORDER
    if not constraint:
        return True
    c_level = WATER_ORDER_LOCAL.get(_normalize(constraint), 1)
    p_level = WATER_ORDER_LOCAL.get(_normalize(plant_water), 1)
    return p_level <= c_level


def apply_filters(
    plants: list["Plant"],
    query: "Query",
) -> list["Plant"]:
    """Applique les filtres climat, sun_exposure, season, water_constraint."""
    result = []
    for p in plants:
        if not _water_compatible(p.water_needs, query.water_constraint):
            continue
        # if not _climate_match(p.climate, query.climat):  # désactivé
        #     continue
        if not _sun_match(p.sun_exposure, query.sun_exposure):
            continue
        if not _season_match(p.season, query.season):
            continue
        result.append(p)
    return result


def compute_score(
    plant: "Plant",
    query: "Query",
    embedding_distance: float,
) -> float:
    """
    score_total = -embedding_distance (plus proche = mieux)
                  + bonus style_tag match
                  + bonus saison match
                  - pénalité eau incompatible (déjà filtré)
    """
    # Convertir distance cosine en score (0-1, 1=meilleur)
    score = 1.0 - min(embedding_distance, 1.0)

    # Bonus style_tag
    q_style = (query.style or "").lower()
    if q_style and q_style in [t.lower() for t in plant.style_tags]:
        score += 0.2
    for tag in plant.style_tags:
        if tag.lower() in (query.description or "").lower():
            score += 0.1

    # Bonus saison
    if _season_match(plant.season, query.season):
        score += 0.15

    return score
