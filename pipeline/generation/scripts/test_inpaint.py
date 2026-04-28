#!/usr/bin/env python3
"""
Test FLUX Fill avec masque manuel (petit cercle).

Usage:
    python -m garden_ai.scripts.test_inpaint
    python -m garden_ai.scripts.test_inpaint --plant lavender

Crée outputs/masks/test_mask.png (cercle r=40 à x=160,y=320)
et outputs/test_inpaint.png. Aucun texte/label sur l'image finale.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Test inpaint avec masque manuel")
    parser.add_argument("--plant", default="lavender", help="Nom de la plante")
    parser.add_argument("--image", default=None, help="Image source")
    parser.add_argument("--cx", type=int, default=160, help="Centre X du masque")
    parser.add_argument("--cy", type=int, default=320, help="Centre Y du masque")
    parser.add_argument("--radius", type=int, default=40, help="Rayon du masque")
    args = parser.parse_args()

    image_path = Path(args.image or str(DATA_DIR / "garden.jpg"))
    if not image_path.exists():
        print(f"❌ Image non trouvée : {image_path}", file=sys.stderr)
        return 1

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    masks_dir = OUTPUTS_DIR / "masks"
    mask_path = masks_dir / "test_mask.png"
    out_path = OUTPUTS_DIR / "test_inpaint.png"

    print("=== Test Inpaint (masque manuel) ===\n")
    print(f"📷 Image : {image_path}")
    print(f"🌿 Plante : {args.plant}")
    print(f"⭕ Masque : cercle r={args.radius} à ({args.cx}, {args.cy})\n")

    # 1. Créer le masque circulaire
    from garden_ai.image_generation.mask_manager import create_manual_test_mask
    create_manual_test_mask(
        image_path=image_path,
        output_path=mask_path,
        cx=args.cx,
        cy=args.cy,
        radius=args.radius,
    )
    print(f"✓ Masque : {mask_path}")

    # 2. Prompt fort (aucun texte/label dans le prompt final)
    from garden_ai.image_generation.prompt_builder import build_inpaint_prompt
    prompt = build_inpaint_prompt(args.plant)
    print(f"✓ Prompt : {prompt[:80]}...")

    # 3. Un seul appel inpaint (pas de boucle, pas d'overlay)
    from garden_ai.image_generation.scene_generator import inpaint
    inpaint(
        image_path=image_path,
        mask_path=mask_path,
        prompt=prompt,
        out_path=out_path,
        seed=42,
        steps=30,
        guidance=50,
    )

    print(f"\n✓ Sortie : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
