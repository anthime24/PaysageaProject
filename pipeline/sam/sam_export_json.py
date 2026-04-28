#!/usr/bin/env python3
"""
Script pour exporter les résultats de SAM au format JSON
Supporte le format RLE (recommandé) ou masque binaire
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from pycocotools import mask as mask_utils
import json
import sys
import os

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

def masks_to_json(masks, image_shape, format="rle", output_file="sam_output.json"):
    """
    Convertit une liste de masques en format JSON
    
    Args:
        masks: Liste de dictionnaires de masques SAM
        image_shape: Tuple (height, width, channels) de l'image
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
    
    output = {
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
    
    return output

def segment_automatic_with_export(image_path, checkpoint_path, format="rle", device="cuda"):
    """Segmentation automatique avec export JSON"""
    print("\n=== SEGMENTATION AUTOMATIQUE AVEC EXPORT JSON ===")
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Erreur : Impossible de lire '{image_path}'")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✓ Image chargée : {image.shape[1]}x{image.shape[0]} pixels")
    
    # Charger le modèle
    print("Chargement du modèle...")
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
    
    # Générer les masques
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=100,
    )
    
    print("Génération des masques...")
    masks = mask_generator.generate(image)
    print(f"✓ {len(masks)} objets détectés")
    
    # Exporter en JSON
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = f"{base_name}_sam_output.json"
    
    result = masks_to_json(masks, image.shape, format=format, output_file=output_file)
    
    # Visualiser
    visualize_with_ids(image, masks, output_file=f"{base_name}_visualization.png")
    
    return result

def segment_interactive_with_export(image_path, checkpoint_path, format="rle", device="cuda"):
    """Segmentation interactive avec export JSON"""
    print("\n=== SEGMENTATION INTERACTIVE AVEC EXPORT JSON ===")
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Erreur : Impossible de lire '{image_path}'")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = image.shape[:2]
    print(f"✓ Image chargée : {image_width}x{image_height} pixels")
    
    # Charger le modèle
    print("Chargement du modèle...")
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
    
    # Interface de clic
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
    
    # Prédire
    print(f"\nGénération du masque avec {len(points)} point(s)...")
    input_points = np.array(points)
    input_labels = np.array(labels)
    
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )
    
    # Prendre le meilleur masque
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]
    
    print(f"✓ Meilleur masque : score = {best_score:.3f}")
    
    # Convertir en format SAM standard
    mask_data = [{
        'segmentation': best_mask,
        'area': np.sum(best_mask),
        'bbox': calculate_bbox_normalized(best_mask, image_width, image_height),
        'predicted_iou': best_score,
        'stability_score': best_score
    }]
    
    # Exporter en JSON
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = f"{base_name}_interactive_sam_output.json"
    
    result = masks_to_json(mask_data, image.shape, format=format, output_file=output_file)
    
    # Visualiser
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
    plt.savefig(f"{base_name}_interactive_visualization.png", dpi=150, bbox_inches='tight')
    print(f"✓ Visualisation sauvegardée : {base_name}_interactive_visualization.png")
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

def load_and_visualize_json(json_file, image_path):
    """Charge un JSON et visualise les segments"""
    print(f"\n=== CHARGEMENT DE {json_file} ===")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    sam_output = data['sam_output']
    print(f"✓ Nombre de segments : {sam_output['num_segments']}")
    print(f"✓ Format : {sam_output['format']}")
    print(f"✓ Taille image : {sam_output['image_size']}")
    
    # Afficher quelques statistiques
    print("\nTop 5 segments par taille :")
    for seg in sam_output['segments'][:5]:
        print(f"  - ID {seg['segment_id']}: {seg['area_ratio']*100:.2f}% de l'image")
    
    return data

def main():
    print("=" * 60)
    print("  SEGMENTATION SAM AVEC EXPORT JSON")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\n❌ Usage : python sam_export_json.py <image_path> [format]")
        print("Exemples :")
        print("  python sam_export_json.py photo.jpg")
        print("  python sam_export_json.py photo.jpg rle")
        print("  python sam_export_json.py photo.jpg binary")
        return
    
    image_path = sys.argv[1]
    format_type = sys.argv[2] if len(sys.argv) > 2 else "rle"
    
    if format_type not in ["rle", "binary"]:
        print(f"❌ Format invalide : {format_type}")
        print("Formats acceptés : rle, binary")
        return
    
    # Télécharger le modèle si nécessaire
    checkpoint_path = download_model()
    if checkpoint_path is None:
        return
    
    # Vérifier PyTorch
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except:
        device = "cpu"
    
    # Menu
    print("\n" + "=" * 60)
    print("  MODE DE SEGMENTATION")
    print("=" * 60)
    print("1. Segmentation automatique (tous les objets)")
    print("2. Segmentation interactive (cliquer sur l'objet)")
    
    choice = input("\nVotre choix (1/2) : ").strip()
    
    if choice == '1':
        result = segment_automatic_with_export(image_path, checkpoint_path, format_type, device)
    elif choice == '2':
        result = segment_interactive_with_export(image_path, checkpoint_path, format_type, device)
    else:
        print("❌ Choix invalide")
        return
    
    if result:
        print("\n✅ Terminé !")
        print("\nFichiers générés :")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if choice == '1':
            print(f"  - {base_name}_sam_output.json")
            print(f"  - {base_name}_visualization.png")
        else:
            print(f"  - {base_name}_interactive_sam_output.json")
            print(f"  - {base_name}_interactive_visualization.png")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
