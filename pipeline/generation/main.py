"""
Garden AI - Point d'entrée principal.

Appelle la démo image_generation (BFL FLUX Fill PRO).
Usage:
    python main.py
    # ou depuis le parent de garden_ai :
    python -m garden_ai.image_generation.demo
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    print("=== Garden AI - Interface Streamlit ===")
    print("Pour lancer l'interface visuelle, utilisez la commande suivante :")
    print("streamlit run garden_ai/ui/app.py")
    print("\nLancement de la démo CLI par défaut...")
    from image_generation.demo import main
    main()
