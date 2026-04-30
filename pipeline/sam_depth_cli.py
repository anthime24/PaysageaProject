#!/usr/bin/env python3
"""
Après preprocess : SAM (masques) + Depth Anything (carte de profondeur).
Optionnel : fusion SAM + depth → VisionOutput.json.

Par défaut, toutes les sorties vont dans le même dossier que le preprocess
(ex. <projet>/work/). Utilise --work-dir pour forcer un dossier (ex. work/).

À la fin, écrit un JSON fusionné (*_pipeline_result.json) : preprocess + depth
+ résumé SAM + (optionnel) vision complète si --fuse.

Usage (depuis la racine Sam_and_Depth) :
  python sam_depth_cli.py --preprocess-json work/photo_preprocessed.json
  python sam_depth_cli.py --preprocess-json work/photo_preprocessed.json --fuse

Prérequis :
  - pip install -e SAM/segment-anything
  - dépendances Depth-Anything (torch, opencv, huggingface_hub, …)
  - checkpoint SAM (téléchargé auto par sam_export_json_v2.py si absent)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
SAM_ROOT = ROOT / "sam"
DEPTH_ROOT = ROOT / "depth"


def resolve_preprocessed_image_path(preprocess_json_path: Path, preprocess_data: dict) -> Path:
    """Même logique que sam_export_json_v2.resolve_preprocessed_image_path."""
    name = preprocess_data.get("preprocessed_filename") or ""
    p = Path(name)
    if p.is_absolute() and p.is_file():
        return p.resolve()
    cand = preprocess_json_path.parent / Path(name).name
    if cand.is_file():
        return cand.resolve()
    if p.is_file():
        return p.resolve()
    raise FileNotFoundError(
        f"Image preprocessed introuvable (attendu près de {preprocess_json_path}) : {name}"
    )


def depth_base_from_image_stem(stem: str) -> str:
    """Aligné sur run_depth_paysagea.py."""
    return stem.replace("_preprocessed", "").replace("_01", "")


def run_step(cmd: List[str], cwd: Path) -> int:
    print(f"\n→ {' '.join(cmd)}", file=sys.stderr)
    print(f"  cwd={cwd}", file=sys.stderr)
    r = subprocess.run(cmd, cwd=str(cwd))
    return r.returncode


def rel_path_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def write_merged_pipeline(
    work_dir: Path,
    preprocess_meta: Dict[str, Any],
    depth_data: Dict[str, Any],
    sam_data: Dict[str, Any],
    file_paths: Dict[str, Path],
    vision_data: Optional[Dict[str, Any]],
    merged_out: Path,
) -> None:
    """Écrit un seul JSON avec métadonnées et chemins relatifs à work_dir."""
    files_rel = {k: rel_path_or_abs(v, work_dir) for k, v in file_paths.items()}

    merged: Dict[str, Any] = {
        "version": "pipeline_merged_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_id": preprocess_meta.get("image_id"),
        "work_dir": str(work_dir.resolve()),
        "files": files_rel,
        "preprocess": preprocess_meta,
        "depth": depth_data,
        "sam": {
            "version": sam_data.get("version"),
            "num_segments": sam_data["sam_output"]["num_segments"],
            "format": sam_data["sam_output"]["format"],
            "image_size": sam_data["sam_output"]["image_size"],
        },
    }
    if vision_data is not None:
        merged["vision"] = vision_data

    merged_out.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SAM + Depth Anything + JSON fusionné (work/)",
    )
    ap.add_argument(
        "--preprocess-json",
        required=True,
        type=Path,
        help="Fichier *_preprocessed.json (sortie de preprocess_cli.py)",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Dossier unique pour SAM, depth, fusion et JSON fusionné (défaut: dossier du preprocess, ex. work/)",
    )
    ap.add_argument(
        "--sam-out-dir",
        type=Path,
        default=None,
        help="Dossier SAM (défaut: --work-dir)",
    )
    ap.add_argument(
        "--depth-outdir",
        type=Path,
        default=None,
        help="Dossier depth (défaut: --work-dir)",
    )
    ap.add_argument(
        "--fuse",
        action="store_true",
        help="Lancer fuse_sam_depth.py (VisionOutput.json + masques)",
    )
    ap.add_argument(
        "--vision-out",
        type=Path,
        default=None,
        help="VisionOutput.json si --fuse (défaut: <work-dir>/VisionOutput.json)",
    )
    ap.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help="Dossier masques si --fuse (défaut: <work-dir>/masks)",
    )
    ap.add_argument(
        "--merged-out",
        type=Path,
        default=None,
        help="JSON fusionné (défaut: <work-dir>/<base>_pipeline_result.json)",
    )
    ap.add_argument(
        "--no-merge",
        action="store_true",
        help="Ne pas écrire le JSON fusionné",
    )
    args = ap.parse_args()

    preprocess_json = args.preprocess_json.resolve()
    if not preprocess_json.is_file():
        print(f"Fichier introuvable : {preprocess_json}", file=sys.stderr)
        return 1

    with open(preprocess_json, encoding="utf-8") as f:
        meta = json.load(f)

    img_path = resolve_preprocessed_image_path(preprocess_json, meta)

    work_dir = (args.work_dir.resolve() if args.work_dir is not None else preprocess_json.parent.resolve())
    sam_out_dir = (args.sam_out_dir or work_dir).resolve()
    depth_outdir = (args.depth_outdir or work_dir).resolve()
    sam_out_dir.mkdir(parents=True, exist_ok=True)
    depth_outdir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # 1) SAM
    rc = run_step(
        [
            py,
            "-X", "utf8",
            str(SAM_ROOT / "sam_export_json_v2.py"),
            str(preprocess_json),
            "--mode",
            "automatic",
            "--format",
            "rle",
            "--out-dir",
            str(sam_out_dir),
        ],
        cwd=SAM_ROOT,
    )
    if rc != 0:
        return rc

    stem = img_path.stem
    sam_json = sam_out_dir / f"{stem}_sam_output.json"
    if not sam_json.is_file():
        print(f"Sortie SAM attendue introuvable : {sam_json}", file=sys.stderr)
        return 1

    with open(sam_json, encoding="utf-8") as f:
        sam_data = json.load(f)

    # 2) Depth
    rc = run_step(
        [
            py,
            "-X", "utf8",
            str(DEPTH_ROOT / "run_depth_paysagea.py"),
            "--img",
            str(img_path),
            "--meta",
            str(preprocess_json),
            "--outdir",
            str(depth_outdir),
            "--near-is-one",
        ],
        cwd=DEPTH_ROOT,
    )
    if rc != 0:
        return rc

    base = depth_base_from_image_stem(stem)
    depth_npy = depth_outdir / f"{base}_depth.npy"
    depth_json_path = depth_outdir / f"{base}_depth.json"
    depth_preview = depth_outdir / f"{base}_depth_preview.png"
    for p in (depth_npy, depth_json_path):
        if not p.is_file():
            print(f"Sortie depth attendue introuvable : {p}", file=sys.stderr)
            return 1

    with open(depth_json_path, encoding="utf-8") as f:
        depth_data = json.load(f)

    sam_viz = sam_out_dir / f"{stem}_sam_visualization.png"

    out: Dict[str, Any] = {
        "work_dir": str(work_dir.resolve()),
        "preprocess_json": str(preprocess_json),
        "preprocessed_image": str(img_path),
        "sam_output_json": str(sam_json.resolve()),
        "depth_npy": str(depth_npy.resolve()),
        "depth_json": str(depth_json_path.resolve()),
        "depth_preview": str(depth_preview.resolve()),
    }

    vision_data: Optional[Dict[str, Any]] = None
    vision_out: Optional[Path] = None
    masks_dir_resolved: Optional[Path] = None

    if args.fuse:
        vision_out = (args.vision_out or (work_dir / "VisionOutput.json")).resolve()
        masks_dir_resolved = (args.masks_dir or (work_dir / "masks")).resolve()
        rc = run_step(
            [
                py,
                str(DEPTH_ROOT / "fuse_sam_depth.py"),
                "--depth-npy",
                str(depth_npy),
                "--depth-json",
                str(depth_json_path),
                "--sam-json",
                str(sam_json),
                "--preprocess-json",
                str(preprocess_json),
                "--out",
                str(vision_out),
                "--out-masks-dir",
                str(masks_dir_resolved),
            ],
            cwd=DEPTH_ROOT,
        )
        if rc != 0:
            return rc
        out["vision_output_json"] = str(vision_out)
        out["masks_dir"] = str(masks_dir_resolved)
        with open(vision_out, encoding="utf-8") as f:
            vision_data = json.load(f)

    # Fichiers pour le JSON fusionné (chemins relatifs à work_dir)
    file_paths: Dict[str, Path] = {
        "preprocessed_image": img_path,
        "preprocessed_json": preprocess_json,
        "sam_output": sam_json,
        "depth_npy": depth_npy,
        "depth_json": depth_json_path,
        "depth_preview": depth_preview,
    }
    if sam_viz.is_file():
        file_paths["sam_visualization"] = sam_viz
    if vision_out is not None and vision_out.is_file():
        file_paths["vision_output"] = vision_out
    if masks_dir_resolved is not None and masks_dir_resolved.is_dir():
        file_paths["masks_dir"] = masks_dir_resolved

    if not args.no_merge:
        merged_path = (args.merged_out or (work_dir / f"{base}_pipeline_result.json")).resolve()
        write_merged_pipeline(
            work_dir=work_dir,
            preprocess_meta=meta,
            depth_data=depth_data,
            sam_data=sam_data,
            file_paths=file_paths,
            vision_data=vision_data,
            merged_out=merged_path,
        )
        out["merged_pipeline_json"] = str(merged_path)
        print(f"\n✅ JSON fusionné : {merged_path}", file=sys.stderr)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
