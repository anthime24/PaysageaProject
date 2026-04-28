#!/usr/bin/env python3
"""
Prétraitement d'une image (EXIF, redimensionnement, métadonnées JSON).

Usage (depuis la racine Sam_and_Depth) :
  python preprocess_cli.py --input photo.jpg --out-dir work
  python preprocess_cli.py -i photo.jpg --out work    # alias de --out-dir

Sortie sur stdout : JSON avec chemins absolus (pour un frontend / API).

Dépendances : voir SAM/segment-anything/preprocess/requirements.txt (Pillow).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREPROCESS_DIR = ROOT / "SAM" / "segment-anything" / "preprocess"
sys.path.insert(0, str(PREPROCESS_DIR))

from preprocess_image import preprocess_image, save_metadata  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="Prétraitement image → *_preprocessed.jpg + *_preprocessed.json",
    )
    p.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Image source (jpg, png, webp, …)",
    )
    p.add_argument(
        "--out-dir",
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Dossier de sortie (défaut: <racine du projet>/work). --out est un alias de --out-dir.",
    )
    p.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="Taille max du plus long côté en pixels (défaut: 1024)",
    )
    args = p.parse_args()
    out_dir = args.out_dir if args.out_dir is not None else (ROOT / "work")

    inp = args.input.resolve()
    if not inp.is_file():
        print(f"Fichier introuvable : {inp}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    out_jpg = out_dir / f"{inp.stem}_preprocessed.jpg"
    out_json = out_jpg.with_suffix(".json")

    meta = preprocess_image(str(inp), str(out_jpg), max_side=args.max_side)
    save_metadata(meta, str(out_json))

    summary = {
        "preprocessed_image": str(out_jpg.resolve()),
        "preprocessed_json": str(out_json.resolve()),
        "image_id": meta.get("image_id"),
        "resized_size": meta.get("preprocess", {}).get("resized_size"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
