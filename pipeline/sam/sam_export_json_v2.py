#!/usr/bin/env python3
"""
Script pour exporter les résultats de SAM au format JSON
MODIFIÉ : Lit le preprocess.json pour garantir la cohérence avec le pipeline
Supporte le format RLE (recommandé) ou masque binaire
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from pycocotools import mask as mask_utils
import json
import sys
import os
import hashlib
from pathlib import Path


def resolve_preprocessed_image_path(preprocess_json_path, preprocess_data):
    """
    Résout le chemin absolu de l'image prétraitée.
    Le JSON enregistre souvent seulement le nom de fichier : on cherche d'abord
    à côté du .json, puis le chemin tel quel (cwd).
    """
    name = preprocess_data.get("preprocessed_filename") or ""
    p = Path(name)
    if p.is_absolute() and p.exists():
        return str(p.resolve())
    json_dir = Path(preprocess_json_path).resolve().parent
    cand = json_dir / Path(name).name
    if cand.exists():
        return str(cand.resolve())
    if p.exists():
        return str(p.resolve())
    raise FileNotFoundError(
        f"Image preprocessed introuvable (près de {preprocess_json_path}) : {name}"
    )

def load_preprocess_json(preprocess_json_path):
    """
    Charge le fichier preprocess.json
    C'est la SOURCE DE VÉRITÉ pour les dimensions et métadonnées
    """
    if not os.path.exists(preprocess_json_path):
        raise FileNotFoundError(
            f"❌ Fichier preprocess non trouvé : {preprocess_json_path}\n"
            f"   Vous devez d'abord lancer preprocess_image.py"
        )
    
    with open(preprocess_json_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Preprocess chargé : {preprocess_json_path}")
    print(f"  Image ID : {data['image_id']}")
    print(f"  Taille : {data['preprocess']['resized_size']}")
    print(f"  Orientation : {data['preprocess']['orientation']}")
    
    return data

def verify_image_matches_preprocess(image, preprocess_data):
    """
    VÉRIFICATION CRITIQUE : L'image chargée correspond-elle au preprocess ?
    Si ça plante ici, c'est qu'il y a une incohérence dangereuse
    """
    h, w = image.shape[:2]
    expected_w, expected_h = preprocess_data['preprocess']['resized_size']
    
    if (w, h) != (expected_w, expected_h):
        raise ValueError(
            f"❌ ERREUR CRITIQUE : Dimensions de l'image ne correspondent pas !\n"
            f"   Image chargée : {w}x{h}\n"
            f"   Attendu (preprocess) : {expected_w}x{expected_h}\n"
            f"   → Vérifiez que vous utilisez la bonne image preprocessed"
        )
    
    print(f"✓ Vérification dimensions : {w}x{h} ✓")

def calculate_image_hash(image_path):
    """Calcule le hash SHA256 d'une image pour vérification"""
    with open(image_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]

def download_model():
    """Télécharge le modèle vit_b si nécessaire"""
    checkpoint_path = "sam_vit_b_01ec64.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"✓ Modèle trouvé : {checkpoint_path}")
        return checkpoint_path
    
    print("⏳ Téléchargement du modèle vit_b (375 MB)...")
    import urllib.request
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    
    try:
        urllib.request.urlretrieve(url, checkpoint_path)
        print(f"✓ Modèle téléchargé : {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        return None

def mask_to_rle(mask):
    """Convertit un masque binaire en format RLE (COCO format)"""
    # Convertir en format COCO (Fortran order)
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    # Décoder les bytes en string pour JSON
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def calculate_bbox_normalized(mask, image_width, image_height):
    """Calcule la bounding box normalisée [x, y, w, h] entre 0 et 1"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Normaliser entre 0 et 1
    x = x_min / image_width
    y = y_min / image_height
    w = (x_max - x_min + 1) / image_width
    h = (y_max - y_min + 1) / image_height
    
    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]

def calculate_centroid_normalized(mask, image_width, image_height):
    """Calcule le centroïde normalisé [x, y] entre 0 et 1"""
    rows, cols = np.where(mask)
    
    if len(rows) == 0 or len(cols) == 0:
        return [0, 0]
    
    centroid_y = np.mean(rows) / image_height
    centroid_x = np.mean(cols) / image_width
    
    return [round(centroid_x, 4), round(centroid_y, 4)]

def segment_to_json(mask, segment_id, image_width, image_height, format="rle"):
    """Convertit un segment en format JSON"""
    
    # Calculer l'aire (ratio par rapport à l'image totale)
    area_pixels = np.sum(mask)
    total_pixels = image_width * image_height
    area_ratio = round(area_pixels / total_pixels, 4)
    
    # Calculer bbox normalisée
    bbox = calculate_bbox_normalized(mask, image_width, image_height)
    
    # Calculer centroïde normalisé
    centroid = calculate_centroid_normalized(mask, image_width, image_height)
    
    segment_data = {
        "segment_id": segment_id,
        "area_ratio": area_ratio,
        "bbox": bbox,
        "centroid": centroid
    }
    
    # Ajouter le masque selon le format
    if format == "rle":
        rle = mask_to_rle(mask)
        segment_data["mask_rle"] = rle
    elif format == "binary":
        # Convertir en liste de listes pour JSON
        segment_data["mask_binary"] = mask.tolist()
    
    return segment_data

def masks_to_json(masks, image_shape, preprocess_data, format="rle", output_file="sam_output.json"):
    """
    Convertit une liste de masques en format JSON
    MODIFIÉ : Inclut les métadonnées du preprocess
    
    Args:
        masks: Liste de dictionnaires de masques SAM
        image_shape: Tuple (height, width, channels) de l'image
        preprocess_data: Données du preprocess.json
        format: "rle" (recommandé) ou "binary"
        output_file: Nom du fichier de sortie
    """
    
    image_height, image_width = image_shape[:2]
    
    segments = []
    for idx, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        segment = segment_to_json(mask, idx, image_width, image_height, format)
        segments.append(segment)
    
    # Trier par aire décroissante
    segments.sort(key=lambda x: x['area_ratio'], reverse=True)
    
    # Réassigner les IDs après tri
    for idx, segment in enumerate(segments):
        segment['segment_id'] = idx
    
    # Structure JSON avec métadonnées du preprocess
    output = {
        "version": "sam_output_v1",
        "image_id": preprocess_data['image_id'],
        "preprocess": preprocess_data['preprocess'],  # Copie complète du preprocess
        "preprocessed_filename": preprocess_data['preprocessed_filename'],
        "sam_output": {
            "image_size": [image_width, image_height],
            "num_segments": len(segments),
            "format": format,
            "segments": segments
        }
    }
    
    # Sauvegarder en JSON
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Résultats sauvegardés : {output_file}")
    print(f"  - Nombre de segments : {len(segments)}")
    print(f"  - Format : {format}")
    print(f"  - Taille fichier : {os.path.getsize(output_file) / 1024:.2f} KB")
    print(f"  - Image ID : {preprocess_data['image_id']}")
    
    return output

def segment_automatic_with_export(
    preprocess_json_path, checkpoint_path, format="rle", device="cuda", out_dir=None
):
    """
    Segmentation automatique avec export JSON
    MODIFIÉ : Utilise le preprocess.json comme source de vérité
    """
    print("\n=== SEGMENTATION AUTOMATIQUE AVEC EXPORT JSON ===")
    
    preprocess_json_path = os.path.abspath(preprocess_json_path)
    # 1. Charger le preprocess.json (SOURCE DE VÉRITÉ)
    preprocess_data = load_preprocess_json(preprocess_json_path)
    
    # 2. Charger l'image preprocessed
    image_path = resolve_preprocessed_image_path(preprocess_json_path, preprocess_data)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image preprocessed non trouvée : {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Erreur lors du chargement de l'image : {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✓ Image chargée : {image_path}")
    
    # 3. VÉRIFICATION CRITIQUE : dimensions correspondent-elles ?
    verify_image_matches_preprocess(image, preprocess_data)
    
    # 4. Charger le modèle SAM
    print("Chargement du modèle SAM...")
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA non disponible, utilisation du CPU")
            device = "cpu"
    except:
        device = "cpu"
    
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=device)
    print(f"✓ Modèle chargé sur {device}")
    
    # 5. Générer les masques
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=100,
    )
    
    print("Génération des masques...")
    masks = mask_generator.generate(image)
    print(f"✓ {len(masks)} objets détectés")
    
    # 6. Exporter en JSON (avec métadonnées du preprocess)
    if out_dir is None:
        out_dir = os.path.dirname(preprocess_json_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = os.path.join(out_dir, f"{base_name}_sam_output.json")
    
    result = masks_to_json(masks, image.shape, preprocess_data, format=format, output_file=output_file)
    
    # 7. Visualiser
    visualize_with_ids(
        image, masks, output_file=os.path.join(out_dir, f"{base_name}_sam_visualization.png")
    )
    
    return result

def segment_interactive_with_export(preprocess_json_path, checkpoint_path, format="rle", device="cuda"):
    """
    Segmentation interactive avec export JSON
    MODIFIÉ : Utilise le preprocess.json comme source de vérité
    """
    print("\n=== SEGMENTATION INTERACTIVE AVEC EXPORT JSON ===")
    
    preprocess_json_path = os.path.abspath(preprocess_json_path)
    # 1. Charger le preprocess.json (SOURCE DE VÉRITÉ)
    preprocess_data = load_preprocess_json(preprocess_json_path)
    
    # 2. Charger l'image preprocessed
    image_path = resolve_preprocessed_image_path(preprocess_json_path, preprocess_data)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image preprocessed non trouvée : {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Erreur lors du chargement de l'image : {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = image.shape[:2]
    print(f"✓ Image chargée : {image_path}")
    
    # 3. VÉRIFICATION CRITIQUE
    verify_image_matches_preprocess(image, preprocess_data)
    
    # 4. Charger le modèle
    print("Chargement du modèle SAM...")
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
    except:
        device = "cpu"
    
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    print(f"✓ Modèle chargé sur {device}")
    
    # 5. Interface de clic
    print("\n📍 Cliquez sur l'objet à segmenter")
    print("   - Clic gauche : point positif (sur l'objet)")
    print("   - Clic droit : point négatif (hors de l'objet)")
    print("   - Touche 'q' : terminer et exporter")
    
    points = []
    labels = []
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title("Cliquez sur l'image (q pour terminer)")
    ax.axis('off')
    
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            
            if event.button == 1:  # Clic gauche
                points.append([x, y])
                labels.append(1)
                color = 'green'
                marker = '*'
                print(f"  ✓ Point positif : ({x}, {y})")
            elif event.button == 3:  # Clic droit
                points.append([x, y])
                labels.append(0)
                color = 'red'
                marker = 'x'
                print(f"  ✗ Point négatif : ({x}, {y})")
            
            ax.plot(x, y, color=color, marker=marker, markersize=15, markeredgewidth=3)
            plt.draw()
    
    def onkey(event):
        if event.key == 'q' and len(points) > 0:
            plt.close()
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()
    
    if len(points) == 0:
        print("❌ Aucun point sélectionné")
        return None
    
    # 6. Prédire
    print(f"\nGénération du masque avec {len(points)} point(s)...")
    input_points = np.array(points)
    input_labels = np.array(labels)
    
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )
    
    # 7. Prendre le meilleur masque
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]
    
    print(f"✓ Meilleur masque : score = {best_score:.3f}")
    
    # 8. Convertir en format SAM standard
    mask_data = [{
        'segmentation': best_mask,
        'area': np.sum(best_mask),
        'bbox': calculate_bbox_normalized(best_mask, image_width, image_height),
        'predicted_iou': best_score,
        'stability_score': best_score
    }]
    
    # 9. Exporter en JSON (avec métadonnées du preprocess)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = f"{base_name}_sam_interactive_output.json"
    
    result = masks_to_json(mask_data, image.shape, preprocess_data, format=format, output_file=output_file)
    
    # 10. Visualiser
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.imshow(best_mask, alpha=0.5, cmap='jet')
    
    for point, label in zip(input_points, input_labels):
        color = 'green' if label == 1 else 'red'
        marker = '*' if label == 1 else 'x'
        ax.plot(point[0], point[1], color=color, marker=marker, 
               markersize=15, markeredgewidth=3)
    
    ax.set_title(f"Score: {best_score:.3f}")
    ax.axis('off')
    plt.savefig(f"{base_name}_sam_interactive_visualization.png", dpi=150, bbox_inches='tight')
    print(f"✓ Visualisation sauvegardée : {base_name}_sam_interactive_visualization.png")
    plt.show()
    
    return result

def visualize_with_ids(image, masks, output_file="visualization.png"):
    """Visualise les masques avec leurs IDs"""
    if len(masks) == 0:
        return
    
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(image)
    ax.set_autoscale_on(False)
    
    for idx, ann in enumerate(sorted_masks):
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
        
        # Ajouter l'ID au centre du masque
        rows, cols = np.where(m)
        if len(rows) > 0:
            center_y = int(np.mean(rows))
            center_x = int(np.mean(cols))
            ax.text(center_x, center_y, str(idx), 
                   color='white', fontsize=12, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.axis('off')
    ax.set_title(f"{len(masks)} segments détectés", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visualisation sauvegardée : {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Segmentation SAM avec export JSON (pipeline preprocess).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python sam_export_json_v2.py path/to/preprocessed.json\n"
            "  python sam_export_json_v2.py preprocessed.json --format rle --mode automatic\n"
            "  python sam_export_json_v2.py preprocessed.json --mode interactive\n"
            "\n"
            "Par défaut : mode automatique (sans menu), pour API / frontend."
        ),
    )
    parser.add_argument(
        "preprocess_json",
        help="Chemin vers *_preprocessed.json (sortie de preprocess_image.py)",
    )
    parser.add_argument(
        "--format",
        choices=("rle", "binary"),
        default="rle",
        help="Format des masques (défaut: rle)",
    )
    parser.add_argument(
        "--mode",
        choices=("automatic", "interactive"),
        default="automatic",
        help="Segmentation automatique (défaut) ou interactive (clics)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Dossier pour *_sam_output.json et la visualisation (défaut: même dossier que le .json)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SEGMENTATION SAM AVEC EXPORT JSON")
    print("  (Version intégrée au pipeline preprocess)")
    print("=" * 60)

    preprocess_json_path = args.preprocess_json
    if not os.path.exists(preprocess_json_path):
        print(f"❌ Fichier preprocess non trouvé : {preprocess_json_path}")
        print("   Lance d'abord : python preprocess_image.py <image> <sortie>")
        return 1

    checkpoint_path = download_model()
    if checkpoint_path is None:
        return 1

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    out_dir = args.out_dir
    if args.mode == "automatic":
        result = segment_automatic_with_export(
            preprocess_json_path,
            checkpoint_path,
            format=args.format,
            device=device,
            out_dir=out_dir,
        )
    else:
        result = segment_interactive_with_export(
            preprocess_json_path, checkpoint_path, format=args.format, device=device
        )

    if result:
        print("\n✅ Terminé !")
        print("\n📊 Résumé :")
        print(f"  - Image ID : {result['image_id']}")
        print(f"  - Dimensions : {result['preprocess']['resized_size']}")
        print(f"  - Segments détectés : {result['sam_output']['num_segments']}")
        print(f"  - Format : {result['sam_output']['format']}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main() or 0)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption par l'utilisateur")
        raise SystemExit(130)
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
