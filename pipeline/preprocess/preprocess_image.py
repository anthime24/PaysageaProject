#!/usr/bin/env python3
"""
preprocess_image.py
===================
Une seule vérité pour l'image d'entrée.

Ce script garantit que :
- la même image est utilisée
- avec les mêmes dimensions
- la même orientation
- les mêmes coordonnées

Responsabilités :
1. Charger l'image proprement (EXIF orientation, RGB forcé)
2. Redimensionner (garder ratio, max_side, pas d'upscale)
3. Normaliser l'espace de coordonnées
4. Sauvegarder l'image preprocessée
5. Retourner les métadonnées JSON
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

from PIL import Image, ImageOps


def compute_image_hash(image_path: str) -> str:
    """
    Calcule un hash SHA-256 du contenu de l'image.
    
    Ce hash est stable et unique pour une image donnée, ce qui permet de :
    - Identifier de manière unique une image preprocessée
    - Détecter si une image a changé
    - Fusionner des résultats de plusieurs pipelines
    
    Args:
        image_path: Chemin vers l'image
    
    Returns:
        Hash SHA-256 au format "sha256:..." (premiers 16 caractères)
    """
    sha256_hash = hashlib.sha256()
    
    with open(image_path, "rb") as f:
        # Lire par blocs pour gérer les grandes images
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    # Retourner les 16 premiers caractères pour la lisibilité
    full_hash = sha256_hash.hexdigest()
    return f"sha256:{full_hash[:16]}"


def preprocess_image(
    input_path: str,
    output_path: str,
    max_side: int = 1024,
    keep_ratio: bool = True
) -> Dict:
    """
    Prétraite une image selon les règles strictes définies.
    
    Args:
        input_path: Chemin vers l'image source
        output_path: Chemin vers l'image preprocessée
        max_side: Taille maximale du côté le plus long
        keep_ratio: Garder le ratio d'aspect (toujours True)
    
    Returns:
        Dict contenant les métadonnées de prétraitement
    """
    
    # 1️⃣ CHARGER L'IMAGE PROPREMENT
    print(f"📂 Chargement de l'image : {input_path}")
    
    # Extraire les noms de fichiers pour tracking
    source_path = Path(input_path)
    output_path_obj = Path(output_path)
    
    source_filename = source_path.name
    preprocessed_filename = output_path_obj.name
    
    print(f"   Source : {source_filename}")
    
    image = Image.open(input_path)
    
    # Détecter l'orientation EXIF AVANT correction
    exif_present = False
    exif_orientation = None
    applied_rotation_deg = 0
    
    try:
        exif = image.getexif()
        if exif:
            # Tag 0x0112 = Orientation
            exif_orientation = exif.get(0x0112, None)
            if exif_orientation is not None:
                exif_present = True
                
                # Mapper l'orientation EXIF vers la rotation en degrés
                # Référence: https://exif.org/Exif2-2.PDF page 18
                orientation_to_rotation = {
                    1: 0,    # Normal
                    2: 0,    # Mirrored
                    3: 180,  # Rotated 180
                    4: 180,  # Mirrored and rotated 180
                    5: 270,  # Mirrored and rotated 270 CW
                    6: 270,  # Rotated 270 CW
                    7: 90,   # Mirrored and rotated 90 CW
                    8: 90,   # Rotated 90 CW
                }
                applied_rotation_deg = orientation_to_rotation.get(exif_orientation, 0)
                
                if applied_rotation_deg != 0:
                    print(f"   🔄 EXIF orientation détectée : {exif_orientation} → rotation de {applied_rotation_deg}°")
    except Exception as e:
        print(f"   ⚠️  Impossible de lire EXIF : {e}")
    
    # Corriger l'orientation EXIF
    image_before = image
    image = ImageOps.exif_transpose(image)
    
    if not exif_present:
        print(f"   ℹ️  Pas de tag EXIF orientation, image inchangée")
    
    # Forcer RGB (convertir RGBA, L, etc.)
    if image.mode != 'RGB':
        print(f"   Conversion {image.mode} → RGB")
        image = image.convert('RGB')
    
    # Stocker la taille originale
    original_width, original_height = image.size
    original_size = [original_width, original_height]
    
    print(f"   Taille originale : {original_width}x{original_height}")
    
    # 2️⃣ REDIMENSIONNER
    # Garder le ratio, max_side, pas d'upscale
    max_original = max(original_width, original_height)
    
    if max_original <= max_side:
        # Pas d'upscale nécessaire
        resized_width = original_width
        resized_height = original_height
        scale_factor = 1.0
        print(f"   ✓ Image déjà ≤ {max_side}px, pas de redimensionnement")
    else:
        # Calculer les nouvelles dimensions en gardant le ratio
        scale_factor = max_side / max_original
        resized_width = int(original_width * scale_factor)
        resized_height = int(original_height * scale_factor)
        
        print(f"   🔄 Redimensionnement : {resized_width}x{resized_height}")
        print(f"   📏 Facteur d'échelle : {scale_factor:.4f}")
        
        # Redimensionner avec LANCZOS pour la meilleure qualité
        image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    
    resized_size = [resized_width, resized_height]
    
    # 4️⃣ SAUVEGARDER L'IMAGE PREPROCESSÉE
    print(f"💾 Sauvegarde : {output_path}")
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    image.save(output_path, quality=95, optimize=True)
    
    # Calculer l'image_id (hash de l'image preprocessée)
    print(f"🔑 Calcul de l'identifiant stable...")
    image_id = compute_image_hash(output_path)
    print(f"   Image ID : {image_id}")
    
    # 3️⃣ & 5️⃣ NORMALISER L'ESPACE DE COORDONNÉES ET RETOURNER MÉTADONNÉES
    metadata = {
        "image_id": image_id,
        "source_filename": source_filename,
        "preprocessed_filename": preprocessed_filename,
        "preprocess": {
            "original_size": original_size,
            "resized_size": resized_size,
            "scale_factor": round(scale_factor, 4),
            "max_side": max_side,
            "keep_ratio": keep_ratio,
            "orientation": {
                "exif_present": exif_present,
                "exif_orientation": exif_orientation,
                "applied_rotation_deg": applied_rotation_deg
            }
        }
    }
    
    print("\n✅ Prétraitement terminé")
    print(f"📊 Métadonnées :")
    print(json.dumps(metadata, indent=2))
    
    return metadata


def save_metadata(metadata: Dict, metadata_path: str) -> None:
    """Sauvegarde les métadonnées dans un fichier JSON."""
    metadata_path_obj = Path(metadata_path)
    metadata_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Métadonnées sauvegardées : {metadata_path}")


def load_metadata(metadata_path: str) -> Dict:
    """Charge les métadonnées depuis un fichier JSON."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_coordinates_to_original(
    x: float,
    y: float,
    metadata: Dict
) -> Tuple[float, float]:
    """
    Convertit des coordonnées depuis l'image redimensionnée vers l'image originale.
    
    Args:
        x, y: Coordonnées dans l'image redimensionnée
        metadata: Métadonnées de prétraitement
    
    Returns:
        (x_orig, y_orig): Coordonnées dans l'image originale
    """
    scale_factor = metadata["preprocess"]["scale_factor"]
    x_orig = x / scale_factor
    y_orig = y / scale_factor
    return x_orig, y_orig


def convert_coordinates_to_resized(
    x: float,
    y: float,
    metadata: Dict
) -> Tuple[float, float]:
    """
    Convertit des coordonnées depuis l'image originale vers l'image redimensionnée.
    
    Args:
        x, y: Coordonnées dans l'image originale
        metadata: Métadonnées de prétraitement
    
    Returns:
        (x_resized, y_resized): Coordonnées dans l'image redimensionnée
    """
    scale_factor = metadata["preprocess"]["scale_factor"]
    x_resized = x * scale_factor
    y_resized = y * scale_factor
    return x_resized, y_resized


def main():
    """Point d'entrée CLI."""
    if len(sys.argv) < 3:
        print("Usage: python preprocess_image.py <input_image> <output_image> [max_side]")
        print("\nExemple:")
        print("  python preprocess_image.py photo.jpg photo_preprocessed.jpg 1024")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    max_side = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    
    # Vérifier que le fichier d'entrée existe
    if not Path(input_path).exists():
        print(f"❌ Erreur : Le fichier '{input_path}' n'existe pas")
        sys.exit(1)
    
    # Prétraiter l'image
    metadata = preprocess_image(input_path, output_path, max_side=max_side)
    
    # Sauvegarder les métadonnées
    metadata_path = str(Path(output_path).with_suffix('.json'))
    save_metadata(metadata, metadata_path)
    
    print(f"\n🎉 Terminé !")
    print(f"   Image : {output_path}")
    print(f"   Metadata : {metadata_path}")


if __name__ == "__main__":
    main()
