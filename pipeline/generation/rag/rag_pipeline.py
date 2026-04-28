"""
Pipeline RAG complet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .index import CHROMA_PATH, build_index, load_index, query_embeddings
from .loader import load_plants
from .retrieve import apply_filters, compute_score
from .schemas import OutputMetadata, Query, RAGOutput


def _query_text(query: Query) -> str:
    """Construit le texte de requête pour les embeddings."""
    parts = [
        query.description,
        query.style,
        query.climat,
        query.season,
        query.sun_exposure,
    ]
    return " ".join(str(p) for p in parts if p)


def run_rag(
    query: Query | dict,
    plants_path: str | Path,
    top_k: int = 6,
    index_path: Path | None = None,
    rebuild_index: bool = False,
) -> dict[str, Any]:
    """
    Exécute le pipeline RAG et retourne le JSON de sortie conforme.

    Args:
        query: Requête (Query ou dict)
        plants_path: Chemin du fichier plantes JSON
        top_k: Nombre de plantes à retourner
        index_path: Chemin ChromaDB (optionnel)
        rebuild_index: Reconstruire l'index même s'il existe

    Returns:
        Dict conforme au format output (metadata + garden)
    """
    if isinstance(query, dict):
        query = Query(**query)

    plants = load_plants(plants_path)
    plants_by_id = {p.plant_id: p for p in plants}

    idx_path = index_path or CHROMA_PATH
    try:
        if rebuild_index or not idx_path.exists():
            col = build_index(plants, idx_path)
        else:
            col = load_index(idx_path)
    except FileNotFoundError:
        col = build_index(plants, idx_path)

    q_text = _query_text(query)
    retrieval_results = query_embeddings(col, q_text, n_results=min(100, len(plants) * 2))

    # Récupérer les plantes par id
    retrieved_plants = []
    for plant_id, dist in retrieval_results:
        if plant_id in plants_by_id:
            retrieved_plants.append((plants_by_id[plant_id], dist))

    # Filtrer
    filtered = apply_filters([p for p, _ in retrieved_plants], query)

    # Réordonner selon distance (garder ordre retrieval pour les filtrés)
    filtered_with_dist = []
    seen = set()
    for p, d in retrieved_plants:
        if p in filtered and p.plant_id not in seen:
            filtered_with_dist.append((p, d))
            seen.add(p.plant_id)

    # Scoring
    scored = [
        (p, compute_score(p, query, d))
        for p, d in filtered_with_dist
    ]
    scored.sort(key=lambda x: -x[1])
    top_plants = [p for p, _ in scored[:top_k]]

    # Si pas assez après filtres, compléter avec les meilleurs par embedding
    if len(top_plants) < top_k:
        for p, _ in scored:
            if p not in top_plants and len(top_plants) < top_k:
                top_plants.append(p)

    garden = [p.to_dict() for p in top_plants]
    metadata = OutputMetadata(
        description=query.description,
        style=query.style,
        climat=query.climat,
    )

    output = RAGOutput(metadata=metadata, garden=garden)
    return output.model_dump()
