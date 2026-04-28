"""
Index ChromaDB pour les embeddings des plantes.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import Plant

CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_db"


def _get_embeddings_model():
    """Charge sentence-transformers (all-MiniLM-L6-v2)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _text_for_embedding(plant: "Plant") -> str:
    """Construit le texte à encoder pour une plante."""
    parts = [
        plant.name,
        plant.type,
        plant.color,
        plant.reason,
        " ".join(plant.style_tags),
        plant.climate,
        plant.season,
        plant.sun_exposure,
    ]
    return " ".join(str(p) for p in parts if p)


def build_index(plants: list["Plant"], persist_path: Path | None = None) -> "ChromaCollection":
    """
    Construit l'index ChromaDB à partir des plantes.
    """
    import chromadb
    from chromadb.config import Settings

    path = persist_path or CHROMA_PATH
    path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    try:
        client.delete_collection(name="plants")
    except Exception:
        pass
    collection = client.create_collection(name="plants", metadata={"hnsw:space": "cosine"})

    model = _get_embeddings_model()
    ids = [p.plant_id for p in plants]
    texts = [_text_for_embedding(p) for p in plants]
    embeddings = model.encode(texts, show_progress_bar=len(plants) > 100).tolist()

    if ids and embeddings:
        collection.add(ids=ids, embeddings=embeddings, documents=texts)
    return collection


def load_index(persist_path: Path | None = None) -> "ChromaCollection":
    """Charge l'index existant."""
    import chromadb

    path = persist_path or CHROMA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Index non trouvé : {path}. Exécuter build_index d'abord.")

    client = chromadb.PersistentClient(path=str(path))
    return client.get_collection(name="plants")


def query_embeddings(
    collection: "ChromaCollection",
    query_text: str,
    n_results: int = 50,
) -> list[tuple[str, float]]:
    """
    Requête par similarité. Retourne [(plant_id, distance), ...].
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_embedding = model.encode([query_text]).tolist()

    results = collection.query(query_embeddings=q_embedding, n_results=n_results)
    ids = results["ids"][0]
    distances = results["distances"][0]
    return list(zip(ids, distances))
