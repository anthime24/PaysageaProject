"""
fuse_sam_depth.py - Fusionne les masques SAM avec la carte de profondeur

Prend :
  - depth.npy + depth.json  (sortie de run_depth_paysagea.py)
  - *_sam_output.json       (sortie de sam_export_json_v2.py)
  - *_preprocessed.json     (optionnel, sortie de preprocess_image.py)

Produit :
  - VisionOutput.json (ou --out) avec pour CHAQUE segment :
      mean_depth, depth_std, depth_band (front/mid/back), min/max_depth, num_pixels
  - Un JSON par masque dans Outputs/masks/ (ou --out-masks-dir)

--- USAGE ---

  Mode 1 – Base name (chemin automatique) :
    python fuse_sam_depth.py --base "mon_image-400x225"

  Mode 2 – Chemins explicites (n'importe quelle image) :
    python fuse_sam_depth.py \\
        --depth-npy  Outputs/foo_depth.npy \\
        --depth-json Outputs/foo_depth.json \\
        --sam-json   ../Inputs/foo_sam_output.json

  Options supplémentaires :
    --preprocess-json  ../Inputs/foo_preprocessed.json   (optionnel)
    --out              VisionOutput.json                  (défaut)
    --out-masks-dir    Outputs/masks                      (défaut)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


# ─────────────────────── helpers ──────────────────────────────────────────────

def depth_band(x: float) -> str:
    """front = proche, mid = milieu, back = loin (convention near_is_one=True)."""
    if x >= 0.66:
        return "front"
    if x >= 0.33:
        return "mid"
    return "back"


def resolve_paths(args: argparse.Namespace):
    """
    Déduit les chemins de fichiers selon le mode utilisé :
      - --base  →  convention <Outputs/BASE_depth.npy>, <../Inputs/BASE_sam_output.json>
      - chemins explicites
    """
    if args.base:
        base = args.base
        depth_npy = Path(args.depth_npy) if args.depth_npy else Path(f"Outputs/{base}_depth.npy")
        depth_json = Path(args.depth_json) if args.depth_json else Path(f"Outputs/{base}_depth.json")
        sam_json = Path(args.sam_json) if args.sam_json else Path(f"../Inputs/{base}_sam_output.json")
        pre_json = Path(args.preprocess_json) if args.preprocess_json else Path(f"../Inputs/{base}_preprocessed.json")
    else:
        if not args.depth_npy or not args.depth_json or not args.sam_json:
            print("❌ Spécifie --base OU (--depth-npy + --depth-json + --sam-json)")
            sys.exit(1)
        depth_npy = Path(args.depth_npy)
        depth_json = Path(args.depth_json)
        sam_json = Path(args.sam_json)
        pre_json = Path(args.preprocess_json) if args.preprocess_json else None

    return depth_npy, depth_json, sam_json, pre_json


def load_files(depth_npy: Path, depth_json: Path, sam_json: Path, pre_json):
    """Charge les fichiers nécessaires avec des messages clairs."""
    for p, label in [(depth_npy, "depth .npy"), (depth_json, "depth .json"), (sam_json, "SAM JSON")]:
        if not p.exists():
            print(f"❌ Fichier introuvable ({label}) : {p}")
            sys.exit(1)

    print(f"  Depth map  : {depth_npy}")
    depth = np.load(depth_npy)

    print(f"  Depth meta : {depth_json}")
    with open(depth_json, "r", encoding="utf-8") as f:
        depth_meta = json.load(f)

    print(f"  SAM JSON   : {sam_json}")
    with open(sam_json, "r", encoding="utf-8") as f:
        sam_data = json.load(f)

    preprocess_meta = {}
    if pre_json and pre_json.exists():
        print(f"  Preprocess : {pre_json}")
        with open(pre_json, "r", encoding="utf-8") as f:
            preprocess_meta = json.load(f)
    elif pre_json:
        print(f"  ⚠️  Preprocess non trouvé (optionnel) : {pre_json}")

    return depth, depth_meta, sam_data, preprocess_meta


def decode_mask(rle, H: int, W: int) -> np.ndarray:
    """Décode un masque RLE (format COCO) en array bool (H, W)."""
    if isinstance(rle, dict) and "size" in rle:
        mask = mask_utils.decode(rle).astype(bool)
    else:
        mask = mask_utils.decode({"size": [H, W], "counts": rle}).astype(bool)

    if mask.shape == (W, H):
        mask = np.ascontiguousarray(mask.T)
    return mask


# ─────────────────────── main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fusionne masques SAM + carte de profondeur → VisionOutput.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode 1 : base name
    parser.add_argument(
        "--base", "-b",
        help='Base du nom de fichier (ex: "mon_image-400x225"). '
             'Déduit automatiquement les chemins depth / SAM / preprocess.',
    )

    # Mode 2 : chemins explicites
    parser.add_argument("--depth-npy",       help="Chemin vers le fichier depth.npy")
    parser.add_argument("--depth-json",      help="Chemin vers le fichier depth.json")
    parser.add_argument("--sam-json",        help="Chemin vers le fichier *_sam_output.json")
    parser.add_argument("--preprocess-json", help="Chemin vers le fichier *_preprocessed.json (optionnel)")

    # Sorties
    parser.add_argument("--out",          default="VisionOutput.json", help="Fichier JSON de sortie (défaut: VisionOutput.json)")
    parser.add_argument("--out-masks-dir", default="Outputs/masks",   help="Dossier pour les JSON par masque (défaut: Outputs/masks)")

    args = parser.parse_args()

    # ── Résolution des chemins ──────────────────────────────────────────────
    depth_npy, depth_json, sam_json, pre_json = resolve_paths(args)

    print("\n=== fuse_sam_depth.py ===")
    print("Chargement des fichiers…")
    depth, depth_meta, sam_data, preprocess_meta = load_files(depth_npy, depth_json, sam_json, pre_json)

    H, W = depth.shape
    print(f"\nDimensions depth  : {W}×{H}")

    segments_raw = sam_data["sam_output"]["segments"]
    print(f"Segments SAM      : {len(segments_raw)}")

    # Vérification alignement
    sam_size = sam_data["sam_output"].get("image_size")
    if sam_size and sam_size != [W, H]:
        raise ValueError(f"SAM image_size {sam_size} ≠ depth {[W, H]}")

    near_is_one = bool(depth_meta.get("near_is_one", True))

    # ── Calcul de la profondeur par masque ─────────────────────────────────
    print("\nCalcul de la profondeur par masque…")
    segments_out = []

    for seg in segments_raw:
        seg_id = seg["segment_id"]
        mask = decode_mask(seg["mask_rle"], H, W)

        if mask.shape != (H, W):
            raise ValueError(f"Masque {seg_id} : shape {mask.shape} ≠ depth {(H, W)}")

        vals = depth[mask]
        if vals.size == 0:
            mean_d = depth_std = min_d = max_d = None
            band = None
        else:
            mean_d = float(vals.mean())
            depth_std = float(vals.std())
            band = depth_band(mean_d)
            min_d = float(vals.min())
            max_d = float(vals.max())

        seg_enriched = dict(seg)
        seg_enriched.update({
            "mean_depth": mean_d,
            "depth_std":  depth_std,
            "depth_band": band,
            "min_depth":  min_d,
            "max_depth":  max_d,
            "num_pixels": int(mask.sum()),
        })
        segments_out.append(seg_enriched)

    # ── Construction de VisionOutput ───────────────────────────────────────
    vision_output = {
        "version": "vision_segments_v1",
        "image_id": (
            preprocess_meta.get("image_id")
            or sam_data.get("image_id")
            or depth_meta.get("image_id")
        ),
        "image_size": [W, H],
        "preprocess": preprocess_meta,
        "depth_meta": {
            "model":       depth_meta.get("model", "LiheYoung/depth_anything_vitl14"),
            "near_is_one": near_is_one,
            "depth_range": depth_meta.get("depth_range", [0.0, 1.0]),
            "normalized":  depth_meta.get("normalized", True),
            "depth_file":  str(depth_npy),
        },
        "sam_meta": {
            "sam_file":       str(sam_json),
            "segments_count": len(segments_out),
        },
        "segments": segments_out,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vision_output, f, indent=2)

    # ── Un JSON par masque ─────────────────────────────────────────────────
    masks_dir = Path(args.out_masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for seg in segments_out:
        sid = seg["segment_id"]
        mask_data = {k: seg.get(k) for k in [
            "segment_id", "mean_depth", "depth_std", "depth_band",
            "min_depth", "max_depth", "num_pixels", "area_ratio",
            "centroid", "bbox",
        ]}
        with open(masks_dir / f"mask_{sid}.json", "w", encoding="utf-8") as f:
            json.dump(mask_data, f, indent=2)

    n = len(segments_out)
    print(f"\n✅  {out_path}  ({n} segments)")
    print(f"✅  {masks_dir}/mask_0.json … mask_{n-1}.json")

    if n > 0:
        s0 = vision_output["segments"][0]
        print("\nExemple segment 0 :", {
            k: s0.get(k)
            for k in ["segment_id", "mean_depth", "depth_std", "depth_band"]
        })


if __name__ == "__main__":
    main()
