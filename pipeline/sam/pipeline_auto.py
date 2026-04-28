#!/usr/bin/env python3
"""
Pipeline automatique : Preprocess → SAM → Analyse
Automatise toutes les étapes avec visualisations et résumé final
Réutilise le code de sam_export_json_v2.py pour la cohérence
"""

import os
import sys
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Imports SAM
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pycocotools import mask as mask_utils

# ============================================================================
# FONCTIONS RÉUTILISÉES DE sam_export_json_v2.py
# ============================================================================

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

def mask_to_rle(mask):
    """Convertit un masque binaire en format RLE (COCO format)"""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
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
    area_pixels = np.sum(mask)
    total_pixels = image_width * image_height
    area_ratio = round(area_pixels / total_pixels, 4)
    
    bbox = calculate_bbox_normalized(mask, image_width, image_height)
    centroid = calculate_centroid_normalized(mask, image_width, image_height)
    
    segment_data = {
        "segment_id": segment_id,
        "area_ratio": area_ratio,
        "bbox": bbox,
        "centroid": centroid
    }
    
    if format == "rle":
        segment_data["mask_rle"] = mask_to_rle(mask)
    elif format == "binary":
        segment_data["mask_binary"] = mask.tolist()
    
    return segment_data

def masks_to_json(masks, image_shape, preprocess_data, format="rle", output_file="sam_output.json"):
    """
    Convertit une liste de masques en format JSON
    MODIFIÉ : Inclut les métadonnées du preprocess
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
        "preprocess": preprocess_data['preprocess'],
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
    
    return output

# ============================================================================
# CLASSE PIPELINE
# ============================================================================

class PipelineAutomation:
    """Classe pour automatiser le pipeline complet"""
    
    def __init__(self, output_dir="pipeline_results"):
        self.output_dir = output_dir
        self.stats = {
            "start_time": None,
            "end_time": None,
            "steps": []
        }
        os.makedirs(output_dir, exist_ok=True)
        
    def log_step(self, step_name, success=True, details=None):
        """Enregistre une étape du pipeline"""
        step = {
            "name": step_name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.stats["steps"].append(step)
        
        status = "✓" if success else "✗"
        print(f"{status} {step_name}")
        if details:
            for key, value in details.items():
                print(f"  - {key}: {value}")
    
    def calculate_image_hash(self, image_path):
        """Calcule le hash SHA256 d'une image"""
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    
    def preprocess_image(self, image_path, target_size=None):
        """ÉTAPE 1 : Preprocess de l'image"""
        print("\n" + "="*60)
        print("ÉTAPE 1/4 : PREPROCESSING")
        print("="*60)
        
        try:
            # Charger l'image
            img = Image.open(image_path)
            original_size = img.size
            self.log_step("Image chargée", details={
                "path": image_path,
                "size": f"{original_size[0]}x{original_size[1]}"
            })
            
            # Corriger l'orientation EXIF
            exif_orientation = 1
            try:
                exif = img._getexif()
                if exif:
                    exif_orientation = exif.get(274, 1)
                    rotation_map = {3: 180, 6: 270, 8: 90}
                    if exif_orientation in rotation_map:
                        img = img.rotate(rotation_map[exif_orientation], expand=True)
                        self.log_step("Orientation corrigée", details={
                            "exif_orientation": exif_orientation
                        })
            except:
                pass
            
            # Redimensionner
            if target_size is None:
                target_size = (int(img.size[0] * 0.25), int(img.size[1] * 0.25))
            
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            orientation = "landscape" if target_size[0] > target_size[1] else "portrait"
            
            self.log_step("Image redimensionnée", details={
                "original": f"{original_size[0]}x{original_size[1]}",
                "resized": f"{target_size[0]}x{target_size[1]}",
                "scale": round(target_size[0] / original_size[0], 2)
            })
            
            # Sauvegarder
            base_name = Path(image_path).stem
            preprocessed_filename = f"{base_name}-{target_size[0]}x{target_size[1]}.jpg"
            preprocessed_path = os.path.join(self.output_dir, preprocessed_filename)
            img_resized.save(preprocessed_path, 'JPEG', quality=95)
            
            # Calculer l'image_id
            image_id = f"sha256:{self.calculate_image_hash(preprocessed_path)}"
            
            # Créer le JSON preprocess
            preprocess_data = {
                "version": "preprocess_v1",
                "image_id": image_id,
                "preprocessed_filename": preprocessed_filename,
                "preprocess": {
                    "original_filename": os.path.basename(image_path),
                    "original_size": list(original_size),
                    "resized_size": list(target_size),
                    "orientation": orientation,
                    "scale": round(target_size[0] / original_size[0], 4),
                    "exif_orientation": exif_orientation
                }
            }
            
            preprocess_json_path = os.path.join(
                self.output_dir,
                f"{base_name}-{target_size[0]}x{target_size[1]}_preprocessed.json"
            )
            
            with open(preprocess_json_path, 'w') as f:
                json.dump(preprocess_data, f, indent=2)
            
            self.log_step("Métadonnées sauvegardées", details={
                "image_id": image_id
            })
            
            return preprocess_json_path, preprocessed_path, preprocess_data
            
        except Exception as e:
            self.log_step("Erreur preprocessing", success=False, details={"error": str(e)})
            raise
    
    def run_sam(self, preprocess_json_path, checkpoint_path="../sam_vit_b_01ec64.pth"):
        """ÉTAPE 2 : Segmentation SAM (réutilise le code de sam_export_json_v2.py)"""
        print("\n" + "="*60)
        print("ÉTAPE 2/4 : SEGMENTATION SAM")
        print("="*60)
        
        try:
            # Charger le preprocess (SOURCE DE VÉRITÉ)
            preprocess_data = load_preprocess_json(preprocess_json_path)
            self.log_step("Preprocess chargé", details={
                "image_id": preprocess_data['image_id']
            })
            
            # Charger l'image preprocessed
            image_path = os.path.join(self.output_dir, preprocess_data['preprocessed_filename'])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # VÉRIFICATION CRITIQUE
            verify_image_matches_preprocess(image, preprocess_data)
            h, w = image.shape[:2]
            self.log_step("Dimensions vérifiées", details={"size": f"{w}x{h}"})
            
            # Charger SAM
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except:
                device = "cpu"
            
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            sam.to(device=device)
            self.log_step("Modèle SAM chargé", details={"device": device})
            
            # Générer les masques
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=0,
                min_mask_region_area=100,
            )
            
            start_time = time.time()
            masks = mask_generator.generate(image)
            inference_time = time.time() - start_time
            
            self.log_step("Segmentation terminée", details={
                "num_segments": len(masks),
                "inference_time": f"{inference_time:.2f}s"
            })
            
            # Créer l'output JSON (réutilise masks_to_json)
            base_name = Path(preprocess_data['preprocessed_filename']).stem
            sam_json_path = os.path.join(self.output_dir, f"{base_name}_sam_output.json")
            
            output = masks_to_json(masks, image.shape, preprocess_data, format="rle", output_file=sam_json_path)
            
            self.log_step("Résultats SAM sauvegardés", details={
                "size": f"{os.path.getsize(sam_json_path) / 1024:.2f} KB"
            })
            
            return sam_json_path, output, image
            
        except Exception as e:
            self.log_step("Erreur SAM", success=False, details={"error": str(e)})
            raise
    
    def create_visualizations(self, image, sam_data, preprocess_data):
        """ÉTAPE 3 : Créer les visualisations"""
        print("\n" + "="*60)
        print("ÉTAPE 3/4 : VISUALISATIONS")
        print("="*60)
        
        base_name = Path(preprocess_data['preprocessed_filename']).stem
        viz_paths = []
        
        try:
            segments = sam_data['sam_output']['segments']
            
            # 1. Image preprocessed
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(image)
            ax.set_title("Image Preprocessed", fontsize=16, fontweight='bold')
            ax.axis('off')
            viz1_path = os.path.join(self.output_dir, f"{base_name}_01_preprocessed.png")
            plt.tight_layout()
            plt.savefig(viz1_path, dpi=150, bbox_inches='tight')
            plt.close()
            viz_paths.append(viz1_path)
            self.log_step("Visualisation 1/5 créée")
            
            # 2. Tous les segments
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            ax.imshow(image)
            ax.set_autoscale_on(False)
            
            for seg in segments:
                mask = mask_utils.decode(seg['mask_rle'])
                color = np.random.random((1, 3)).tolist()[0]
                colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
                for i in range(3):
                    colored_mask[:, :, i] = color[i]
                ax.imshow(np.dstack((colored_mask, mask * 0.4)))
            
            ax.axis('off')
            ax.set_title(f"Tous les segments ({len(segments)} objets)", fontsize=16, fontweight='bold')
            viz2_path = os.path.join(self.output_dir, f"{base_name}_02_all_segments.png")
            plt.tight_layout()
            plt.savefig(viz2_path, dpi=150, bbox_inches='tight')
            plt.close()
            viz_paths.append(viz2_path)
            self.log_step("Visualisation 2/5 créée")
            
            # 3. Segments avec IDs
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            ax.imshow(image)
            ax.set_autoscale_on(False)
            
            for seg in segments:
                mask = mask_utils.decode(seg['mask_rle'])
                color = np.random.random((1, 3)).tolist()[0]
                colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
                for i in range(3):
                    colored_mask[:, :, i] = color[i]
                ax.imshow(np.dstack((colored_mask, mask * 0.35)))
                
                # Ajouter l'ID
                rows, cols = np.where(mask)
                if len(rows) > 0:
                    center_y = int(np.mean(rows))
                    center_x = int(np.mean(cols))
                    ax.text(center_x, center_y, str(seg['segment_id']),
                           color='white', fontsize=12, fontweight='bold',
                           ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            ax.axis('off')
            ax.set_title("Segments avec IDs", fontsize=16, fontweight='bold')
            viz3_path = os.path.join(self.output_dir, f"{base_name}_03_segments_with_ids.png")
            plt.tight_layout()
            plt.savefig(viz3_path, dpi=150, bbox_inches='tight')
            plt.close()
            viz_paths.append(viz3_path)
            self.log_step("Visualisation 3/5 créée")
            
            # 4. Bounding boxes
            image_bbox = image.copy()
            h, w = image.shape[:2]
            
            for seg in segments[:10]:
                x, y, bw, bh = seg['bbox']
                x_px, y_px = int(x * w), int(y * h)
                w_px, h_px = int(bw * w), int(bh * h)
                
                image_bbox = cv2.rectangle(image_bbox, (x_px, y_px), (x_px + w_px, y_px + h_px), (0, 255, 0), 2)
                label = f"#{seg['segment_id']} ({seg['area_ratio']*100:.1f}%)"
                image_bbox = cv2.putText(image_bbox, label, (x_px, y_px - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            ax.imshow(image_bbox)
            ax.set_title("Bounding Boxes (Top 10)", fontsize=16, fontweight='bold')
            ax.axis('off')
            viz4_path = os.path.join(self.output_dir, f"{base_name}_04_bounding_boxes.png")
            plt.tight_layout()
            plt.savefig(viz4_path, dpi=150, bbox_inches='tight')
            plt.close()
            viz_paths.append(viz4_path)
            self.log_step("Visualisation 4/5 créée")
            
            # 5. Comparaison
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            axes[0].imshow(image)
            axes[0].set_title("Image Originale", fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(image)
            for seg in segments:
                mask = mask_utils.decode(seg['mask_rle'])
                color = np.random.random((1, 3)).tolist()[0]
                colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
                for i in range(3):
                    colored_mask[:, :, i] = color[i]
                axes[1].imshow(np.dstack((colored_mask, mask * 0.4)))
            axes[1].set_title(f"Segmentation ({len(segments)} objets)", fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            viz5_path = os.path.join(self.output_dir, f"{base_name}_05_comparison.png")
            plt.tight_layout()
            plt.savefig(viz5_path, dpi=150, bbox_inches='tight')
            plt.close()
            viz_paths.append(viz5_path)
            self.log_step("Visualisation 5/5 créée")
            
            return viz_paths
            
        except Exception as e:
            self.log_step("Erreur visualisations", success=False, details={"error": str(e)})
            raise
    
    def generate_summary(self, sam_data, preprocess_data, viz_paths):
        """ÉTAPE 4 : Générer le résumé"""
        print("\n" + "="*60)
        print("ÉTAPE 4/4 : RÉSUMÉ FINAL")
        print("="*60)
        
        try:
            base_name = Path(preprocess_data['preprocessed_filename']).stem
            segments = sam_data['sam_output']['segments']
            areas = [seg['area_ratio'] for seg in segments]
            
            summary = {
                "pipeline_info": {
                    "date": datetime.now().isoformat(),
                    "total_time": f"{time.time() - self.stats['start_time']:.2f}s"
                },
                "input": {
                    "original_file": preprocess_data['preprocess']['original_filename'],
                    "original_size": preprocess_data['preprocess']['original_size'],
                },
                "preprocessing": {
                    "resized_size": preprocess_data['preprocess']['resized_size'],
                    "scale": preprocess_data['preprocess']['scale'],
                    "orientation": preprocess_data['preprocess']['orientation']
                },
                "segmentation": {
                    "num_segments": len(segments),
                    "format": sam_data['sam_output']['format']
                },
                "statistics": {
                    "area_mean": round(np.mean(areas) * 100, 2),
                    "area_median": round(np.median(areas) * 100, 2),
                    "area_min": round(np.min(areas) * 100, 2),
                    "area_max": round(np.max(areas) * 100, 2),
                    "distribution": {
                        "tiny (<1%)": sum(1 for a in areas if a < 0.01),
                        "small (1-5%)": sum(1 for a in areas if 0.01 <= a < 0.05),
                        "medium (5-20%)": sum(1 for a in areas if 0.05 <= a < 0.2),
                        "large (>20%)": sum(1 for a in areas if a >= 0.2)
                    }
                },
                "top_segments": [
                    {
                        "id": seg['segment_id'],
                        "area_percent": round(seg['area_ratio'] * 100, 2),
                        "bbox": seg['bbox'],
                        "centroid": seg['centroid']
                    }
                    for seg in segments[:5]
                ],
                "outputs": {
                    "preprocess_json": os.path.basename(preprocess_data['preprocessed_filename'].replace('.jpg', '_preprocessed.json')),
                    "sam_json": f"{base_name}_sam_output.json",
                    "visualizations": [os.path.basename(p) for p in viz_paths]
                }
            }
            
            # JSON
            summary_json = os.path.join(self.output_dir, f"{base_name}_SUMMARY.json")
            with open(summary_json, 'w') as f:
                json.dump(summary, f, indent=2)
            self.log_step("Résumé JSON créé")
            
            # TXT
            summary_txt = os.path.join(self.output_dir, f"{base_name}_SUMMARY.txt")
            with open(summary_txt, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("RÉSUMÉ DU PIPELINE - PREPROCESS → SAM\n")
                f.write("="*60 + "\n\n")
                f.write(f"📅 Date : {summary['pipeline_info']['date']}\n")
                f.write(f"⏱️  Temps total : {summary['pipeline_info']['total_time']}\n\n")
                f.write("📸 IMAGE ORIGINALE\n")
                f.write(f"  • Fichier : {summary['input']['original_file']}\n")
                f.write(f"  • Taille : {summary['input']['original_size'][0]}x{summary['input']['original_size'][1]}\n\n")
                f.write("🔄 PREPROCESSING\n")
                f.write(f"  • Taille finale : {summary['preprocessing']['resized_size'][0]}x{summary['preprocessing']['resized_size'][1]}\n")
                f.write(f"  • Scale : {summary['preprocessing']['scale']}\n\n")
                f.write("🎯 SEGMENTATION SAM\n")
                f.write(f"  • Segments : {summary['segmentation']['num_segments']}\n\n")
                f.write("📊 STATISTIQUES\n")
                f.write(f"  • Aire moyenne : {summary['statistics']['area_mean']}%\n")
                f.write(f"  • Plus grand : {summary['statistics']['area_max']}%\n\n")
                f.write("🏆 TOP 5 SEGMENTS\n")
                for i, seg in enumerate(summary['top_segments'], 1):
                    f.write(f"  {i}. Segment #{seg['id']} : {seg['area_percent']}%\n")
            
            self.log_step("Résumé TXT créé")
            
            return summary, summary_json, summary_txt
            
        except Exception as e:
            self.log_step("Erreur résumé", success=False, details={"error": str(e)})
            raise
    
    def run(self, image_path, checkpoint_path="../sam_vit_b_01ec64.pth"):
        """Exécute le pipeline complet"""
        self.stats["start_time"] = time.time()
        
        print("\n" + "="*60)
        print("🚀 PIPELINE AUTOMATIQUE : PREPROCESS → SAM")
        print("="*60)
        print(f"Image : {image_path}")
        print(f"Output : {self.output_dir}\n")
        
        try:
            # Étape 1 : Preprocess
            preprocess_json, preprocessed_path, preprocess_data = self.preprocess_image(image_path)
            
            # Étape 2 : SAM
            sam_json, sam_data, image = self.run_sam(preprocess_json, checkpoint_path)
            
            # Étape 3 : Visualisations
            viz_paths = self.create_visualizations(image, sam_data, preprocess_data)
            
            # Étape 4 : Résumé
            summary, summary_json, summary_txt = self.generate_summary(sam_data, preprocess_data, viz_paths)
            
            self.stats["end_time"] = time.time()
            
            # Afficher résumé
            self.print_final_summary(summary)
            
            return {
                "success": True,
                "summary": summary,
                "files": {
                    "preprocess_json": preprocess_json,
                    "sam_json": sam_json,
                    "summary_json": summary_json,
                    "summary_txt": summary_txt,
                    "visualizations": viz_paths
                }
            }
            
        except Exception as e:
            self.stats["end_time"] = time.time()
            print(f"\n❌ ERREUR : {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def print_final_summary(self, summary):
        """Affiche le résumé final"""
        print("\n" + "="*60)
        print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        print("="*60)
        print(f"\n⏱️  Temps total : {summary['pipeline_info']['total_time']}")
        print(f"\n📊 Résultats :")
        print(f"  • {summary['segmentation']['num_segments']} objets détectés")
        print(f"  • {len(summary['outputs']['visualizations'])} visualisations créées")
        print(f"\n🏆 Top 3 objets :")
        for i, seg in enumerate(summary['top_segments'][:3], 1):
            print(f"  {i}. Segment #{seg['id']} : {seg['area_percent']}%")
        print(f"\n📁 Tous les fichiers dans : {self.output_dir}/")
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline automatique : Preprocess → SAM → Analyse")
    parser.add_argument("image", help="Chemin de l'image à traiter")
    parser.add_argument("--output", default="pipeline_results", help="Dossier de sortie")
    parser.add_argument("--checkpoint", default="../sam_vit_b_01ec64.pth", help="Checkpoint SAM")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Image non trouvée : {args.image}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint SAM non trouvé : {args.checkpoint}")
        print(f"Assurez-vous que le fichier existe dans le dossier actuel.")
        sys.exit(1)
    
    pipeline = PipelineAutomation(output_dir=args.output)
    result = pipeline.run(args.image, args.checkpoint)
    
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
