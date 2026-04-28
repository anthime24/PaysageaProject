"""
Démo RAG : pipeline ou chargement jardin_complet.json.

Usage:
    python -m garden_ai.rag.demo                    # pipeline → rag_output.json
    python -m garden_ai.rag.demo --load data/jardin_complet.json  # charge et valide
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = [PROJECT_ROOT / "data", PROJECT_ROOT / "data "]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _find_file(name: str) -> Path:
    for d in DATA_DIRS:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} non trouvé dans data/ ou data /")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", metavar="PATH", help="Charge un fichier RAG existant (jardin_complet.json) et affiche le nombre de plantes")
    args = parser.parse_args()

    if args.load:
        path = Path(args.load)
        if not path.exists():
            print(f"❌ Fichier non trouvé : {path}", file=sys.stderr)
            sys.exit(1)
        from ..image_generation.utils_rag import load_rag
        metadata, plants = load_rag(path)
        print(f"✅ {len(plants)} plantes détectées depuis {path}")
        return

    print("=== RAG Jardin - Démo ===\n")

    try:
        plants_path = _find_file("plants.json")
    except FileNotFoundError:
        print("❌ plants.json non trouvé dans data/ ou data /")
        print("   Créez data/plants.json avec une liste de plantes (voir data/plants.json exemple)")
        sys.exit(1)

    query = {
        "style": "potager",
        "climat": "tempere",
        "sun_exposure": "plein_soleil",
        "season": "printemps",
        "water_constraint": "moyen",
        "description": "fleuri",
    }

    print(f"📂 Plantes : {plants_path}")
    print(f"🔍 Requête : {query}\n")

    from .rag_pipeline import run_rag

    output = run_rag(query=query, plants_path=plants_path, top_k=6, rebuild_index=True)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "rag_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Sortie : {out_path}")
    print(f"   {len(output['garden'])} plantes recommandées")


if __name__ == "__main__":
    main()
