"""
Mini-RAG local pour recommandation de plantes de jardin.
"""
from .schemas import Plant, Query, OutputMetadata, RAGOutput
from .loader import load_plants
from .rag_pipeline import run_rag

__all__ = ["Plant", "Query", "OutputMetadata", "RAGOutput", "load_plants", "run_rag"]
