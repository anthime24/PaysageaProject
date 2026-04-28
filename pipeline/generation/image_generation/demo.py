"""
Démo : génère final_garden.png depuis RAG (jardin_complet.json ou rag_output.json).

Usage:
    python -m garden_ai.image_generation.demo
    python -m garden_ai.image_generation.demo --rag data/rag_output.json
    RAG_JSON=data/rag_output.json python -m garden_ai.image_generation.demo
"""
from __future__ import annotations

import argparse
import os
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
    raise FileNotFoundError(f"{name} non trouvé dans data/ ou data /\nChemins: {[str(d / name) for d in DATA_DIRS]}")


def main():
    parser = argparse.ArgumentParser(description="Garden AI - Génération depuis RAG")
    parser.add_argument("--rag", default=None, help="Fichier RAG JSON (défaut: jardin_complet.json puis rag_output.json)")
    parser.add_argument("--image", default=None, help="Image source (défaut: data/garden.jpg)")
    args = parser.parse_args()

    rag_path = args.rag or os.environ.get("RAG_JSON")
    if rag_path:
        rag_path = Path(rag_path)
        if not rag_path.is_absolute() and not rag_path.exists():
            rag_path = PROJECT_ROOT / rag_path
    else:
        try:
            rag_path = _find_file("jardin_complet.json")
        except FileNotFoundError:
            try:
                rag_path = _find_file("rag_output.json")
            except FileNotFoundError:
                rag_path = PROJECT_ROOT / "data" / "rag_output.json"
        rag_path = Path(rag_path)

    image_path = args.image or _find_file("garden.jpg")

    if not rag_path.exists():
        print(f"❌ RAG non trouvé : {rag_path}", file=sys.stderr)
        sys.exit(1)
    if not Path(image_path).exists():
        print(f"❌ Image non trouvée : {image_path}", file=sys.stderr)
        sys.exit(1)

    print("=== Garden AI - Génération depuis RAG ===\n")
    print(f"📷 Image : {image_path}")
    print(f"🌿 RAG   : {rag_path}\n")

    from .scene_generator import generate_scene

    try:
        scene = generate_scene(
            image_path=image_path,
            rag_json_path=rag_path,
            outputs_dir=OUTPUTS_DIR,
        )
        print(f"\n✅ {len(scene['plants'])} plantes générées")
        print(f"   Sortie : {OUTPUTS_DIR}/final_garden.png")
    except Exception as e:
        print(f"❌ Erreur : {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
