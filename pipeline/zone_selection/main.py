import argparse
import cv2
import numpy as np
from pathlib import Path

from utils import (
    CoordinateMapper,
    compute_area_pixels,
    compute_bbox_from_mask,
    compute_centroid_from_mask,
    mask_to_coco_rle,
)
from polygon_tool import PolygonTool
from brush_tool import BrushTool
from export_json import build_zone_dict, export_user_zones, save_mask_png, save_overlay_png


# racine du projet Paysagea
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZONE_SELECTION_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REF_IMAGE_PATH = PROJECT_ROOT / "Inputs" / "menphis_depay-400x225_01_preprocessed.png"
DEFAULT_OUTPUTS_DIR = ZONE_SELECTION_ROOT / "outputs"
DEFAULT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _choose_mode() -> str:
    """
    Petite fenêtre de choix UX : pinceau ou polygone.
    Retourne 'brush' ou 'polygon'.
    """
    win_name = "Choix du mode"
    canvas = 255 * np.ones((200, 500, 3), dtype="uint8")
    cv2.putText(canvas, "Appuie sur 'b' pour Pinceau", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "ou sur 'p' pour Polygone", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, canvas)

    mode = None
    while mode is None:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("b"), ord("B")):
            mode = "brush"
        elif key in (ord("p"), ord("P")):
            mode = "polygon"
        elif key == 27:  # ESC -> on annule
            mode = "cancel"

    cv2.destroyWindow(win_name)
    return mode


def _choose_next_action(current_mode: str, zone_count: int) -> str:
    """
    Fenêtre UX après validation d'une zone.

    Raccourcis :
    - n : nouvelle zone (même mode)
    - p : passer en polygone et créer une nouvelle zone
    - b : passer en pinceau et créer une nouvelle zone
    - s : sauvegarder (export) et quitter
    - esc : quitter sans sauvegarder
    """
    win_name = "Action suivante"
    canvas = 255 * np.ones((260, 700, 3), dtype="uint8")

    cv2.putText(
        canvas,
        f"Zones creees: {zone_count} | Mode actuel: {current_mode}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    cv2.putText(canvas, "n : nouvelle zone (meme mode)", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "p : nouvelle zone en polygone", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "b : nouvelle zone au pinceau", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "s : sauvegarder (export) et quitter", (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "ESC : quitter sans sauvegarder", (420, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, canvas)

    action = None
    while action is None:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("n"), ord("N")):
            action = "new_same"
        elif key in (ord("p"), ord("P")):
            action = "new_polygon"
        elif key in (ord("b"), ord("B")):
            action = "new_brush"
        elif key in (ord("s"), ord("S")):
            action = "save_quit"
        elif key == 27:
            action = "quit_no_save"

    cv2.destroyWindow(win_name)
    return action


def main():
    parser = argparse.ArgumentParser(
        description="Outil zone-selection (polygon/brush) exportant user_zone.json aligné sur une image de référence."
    )
    parser.add_argument(
        "--ref-image",
        type=str,
        default=str(DEFAULT_REF_IMAGE_PATH),
        help="Chemin vers l'image de référence (préprocessée, même taille que SAM/Depth).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUTS_DIR),
        help="Dossier de sortie (user_zone.json + mask/overlay PNG).",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=3.0,
        help="Facteur d'agrandissement pour l'affichage UI (coordonnées exportées restent celles de la ref).",
    )

    args = parser.parse_args()

    ref_image_path = Path(args.ref_image).expanduser().resolve()
    outputs_dir = Path(args.output_dir).expanduser().resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tentative de chargement de l'image de référence : {ref_image_path}")

    if not ref_image_path.exists():
        print("Erreur : le fichier n'existe pas à cet emplacement.")
        return

    ref_img = cv2.imread(str(ref_image_path))
    if ref_img is None:
        print("Erreur : impossible de charger l'image (cv2.imread a échoué).")
        return

    ref_h, ref_w = ref_img.shape[:2]
    print(f"Image de référence chargée avec succès : {ref_w}x{ref_h}")

    # image d'affichage : upscale pour le confort
    scale_factor = float(args.display_scale)
    disp_w = int(ref_w * scale_factor)
    disp_h = int(ref_h * scale_factor)
    disp_img = cv2.resize(ref_img, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

    mapper = CoordinateMapper.from_shapes(ref_shape=(ref_h, ref_w), disp_shape=(disp_h, disp_w))

    # --- boucle multi-zones ---
    zones = []
    combined_mask_ref = np.zeros((ref_h, ref_w), dtype="uint8")
    zone_idx = 0

    # mode initial (une seule fois)
    mode = _choose_mode()
    if mode == "cancel":
        print("Choix annulé par l'utilisateur, arrêt.")
        return

    should_export = False
    quit_without_saving = False

    while True:

        if mode == "polygon":
            window_name = f"Zone {zone_idx + 1} (polygon)"
            tool = PolygonTool(disp_img, window_name=window_name)
        else:
            window_name = f"Zone {zone_idx + 1} (brush)"
            tool = BrushTool(disp_img, window_name=window_name)

        mask_disp, points_disp = tool.run()
        if mask_disp is None:
            print("Aucune zone validée, arrêt de la création de zones.")
            break

        # Conversion du masque en espace de référence
        mask_ref = mapper.mask_display_to_ref(mask_disp)
        area_pixels = compute_area_pixels(mask_ref)
        if area_pixels == 0:
            print("Zone vide (aucun pixel sélectionné), ignorée.")
            continue

        # Calcul des métriques en espace de référence
        bbox = compute_bbox_from_mask(mask_ref)  # [x, y, w, h]
        centroid = compute_centroid_from_mask(mask_ref)
        rle = mask_to_coco_rle(mask_ref)

        label = f"plantable_zone_{zone_idx + 1}"
        zone = build_zone_dict(
            zone_id=zone_idx,
            mode=mode,
            label=label,
            bbox=bbox,
            centroid=centroid,
            area_pixels=area_pixels,
            polygon_points=points_disp,
            mask_rle=rle,
        )
        zones.append(zone)
        combined_mask_ref = np.maximum(combined_mask_ref, mask_ref)
        zone_idx += 1

        print(f"Zone {zone_idx} ajoutée ({mode}), aire = {area_pixels} pixels.")

        # après validation : action suivante
        action = _choose_next_action(current_mode=mode, zone_count=zone_idx)
        if action == "save_quit":
            should_export = True
            break
        if action == "quit_no_save":
            quit_without_saving = True
            break
        if action == "new_polygon":
            mode = "polygon"
        elif action == "new_brush":
            mode = "brush"
        else:
            # new_same
            pass

    if quit_without_saving:
        print("Sortie demandée sans sauvegarde. Aucun fichier n'a été exporté.")
        return

    if not zones:
        print("Aucune zone créée, arrêt sans export.")
        return

    if not should_export:
        # Par sécurité : on n'exporte que sur demande explicite (touche 's')
        print("Aucune sauvegarde demandée (touche 's'). Aucun fichier n'a été exporté.")
        return

    # Préparation des chemins de sortie (export final avec toutes les zones)
    image_filename = ref_image_path.name
    image_id = ref_image_path.stem

    json_path = outputs_dir / "user_zone.json"
    mask_png_path = outputs_dir / "user_zone_mask.png"
    overlay_png_path = outputs_dir / "user_zone_overlay.png"

    export_user_zones(
        zones=zones,
        output_json_path=json_path,
        image_id=image_id,
        image_filename=image_filename,
        image_size=(ref_w, ref_h),
    )

    save_mask_png(combined_mask_ref, mask_png_path)
    save_overlay_png(ref_img, combined_mask_ref, overlay_png_path)

    print(f"{len(zones)} zone(s) utilisateur sauvegardée(s) dans : {json_path}")
    print(f"Masque combiné des zones : {mask_png_path}")
    print(f"Overlay combiné des zones : {overlay_png_path}")


if __name__ == "__main__":
    main()