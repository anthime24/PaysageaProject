"""
Génération de scène jardin : inpainting global avec masque plantable intelligent.

Améliorations v2 :
- Utilise plantable_zone_generator.py pour des masques fiables
- Overlay debug automatique (mask_debug.png)
- Préservation garantie de l'image originale hors masque
- Hook pour zones externes du collègue (external_zones)
- Post-fusion originale + inpaint pixel-perfect
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

from PIL import Image
import numpy as np

from .mask_manager import MaskManager
from .prompt_builder import build_prompt
from .utils_rag import ALLOWED_INPAINT_KWARGS, load_rag


def inpaint(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    prompt: str,
    out_path: Union[str, Path],
    **kwargs,
) -> None:
    """
    Switch provider : MOCK si BFL_API_KEY absent ou MOCK_BFL=true ou 402.
    Ne passe que seed, steps, guidance à l'API BFL.
    """
    allowed = {k: kwargs[k] for k in ALLOWED_INPAINT_KWARGS if k in kwargs}

    use_mock = (
        os.environ.get("MOCK_BFL", "").lower() == "true"
        or not os.environ.get("BFL_API_KEY", "").strip()
    )

    if use_mock:
        print("   [MOCK MODE: skipping image generation API]")
        from .mock_provider import inpaint_mock
        inpaint_mock(image_path, mask_path, prompt, out_path, seed=allowed.get("seed"))
        return

    try:
        from .bfl_provider import inpaint as bfl_inpaint
        bfl_inpaint(image_path, mask_path, prompt, out_path, **allowed)
    except Exception as e:
        if "402" in str(e) or "Insufficient credits" in str(e):
            print("   [BFL 402 Insufficient credits → fallback MOCK]")
            from .mock_provider import inpaint_mock
            inpaint_mock(image_path, mask_path, prompt, out_path, **kwargs)
        else:
            raise


def _preserve_original_outside_mask(
    original_path: Union[str, Path],
    generated_path: Union[str, Path],
    mask_path: Union[str, Path],
    out_path: Union[str, Path],
) -> None:
    """
    Fusionne l'original (hors masque) avec le généré (dans masque).
    Garantit que BFL ne modifie rien en dehors des zones plantables.
    """
    try:
        orig = np.array(Image.open(original_path).convert("RGB"))
        gen_img = Image.open(generated_path).convert("RGB")
        if gen_img.size != (orig.shape[1], orig.shape[0]):
            gen_img = gen_img.resize((orig.shape[1], orig.shape[0]), Image.LANCZOS)
        gen = np.array(gen_img)
        mask_img = Image.open(mask_path).convert("L")
        if mask_img.size != (orig.shape[1], orig.shape[0]):
            mask_img = mask_img.resize((orig.shape[1], orig.shape[0]), Image.NEAREST)
        mask_arr = np.array(mask_img) >= 128  # True = zone generee

        combined = np.where(mask_arr[..., None], gen, orig).astype(np.uint8)
        Image.fromarray(combined).save(out_path)
        print(f"   [POST-FUSION] Originale preservee hors masque -> {out_path}")
    except Exception as e:
        print(f"   [POST-FUSION] Echec : {e}")
        # Fallback : copier le généré brut
        import shutil
        shutil.copy2(generated_path, out_path)


def generate_scene(
    image_path: Union[str, Path],
    rag_json_path: Union[str, Path],
    outputs_dir: Union[str, Path] = "outputs",
    global_style: str | None = None,
    time_of_day: str = "day",
    night_light_intensity: float = 0.5,
    external_zones: list[dict] | None = None,
    external_plantable_mask_path: Union[str, Path] | None = None,
    debug: bool = True,
    mode: str = "sequential",
    **kwargs,
) -> dict:
    """
    Génère le jardin idéal.

    Args:
        image_path: Photo du jardin source
        rag_json_path: JSON de sortie du RAG (plantes sélectionnées)
        outputs_dir: Dossier de sortie
        global_style: Style global (optionnel, priorité au RAG)
        time_of_day: "day" ou "night"
        night_light_intensity: Intensité éclairage nuit (0-1)
        external_zones: Zones plantables fournies par le code du collègue.
        external_plantable_mask_path: Chemin d'un masque PNG externe (du collègue)
        debug: Si True, génère mask_debug.png (overlay vert sur image)
        mode: "sequential" (plante par plante) ou "global" (un seul appel BFL)

    Returns:
        dict scene avec clés: input_image, final_image, metadata, plants, mask_info / steps
    """
    mode = (mode or "sequential").lower().strip()

    if mode == "sequential":
        from .plant_by_plant_generator import generate_garden_plant_by_plant

        scene = generate_garden_plant_by_plant(
            image_path=image_path,
            rag_json_path=rag_json_path,
            outputs_dir=outputs_dir,
            external_plantable_zones=external_zones,
            external_plantable_mask_path=external_plantable_mask_path,
            debug=debug,
            max_plants=kwargs.get("max_plants", 6),
        )
        # Relight nuit éventuel
        final_path = Path(scene["final_image"])
        if time_of_day == "night" and final_path.exists():
            try:
                from ..utils.relight import relight_to_night
                relight_to_night(
                    final_path,
                    Path(outputs_dir) / "final_garden_night.png",
                    light_intensity=night_light_intensity,
                    plants=scene.get("plants", []),
                )
            except Exception as e:
                print(f"[RELIGHT] Erreur : {e}")
        return scene

    # --- Mode historique : global (un seul appel BFL) ---
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = outputs_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image non trouvée : {image_path}")

    metadata, plants_data = load_rag(rag_json_path)
    plants_data = plants_data[:8]
    print(f"🌱 {len(plants_data)} plantes sélectionnées pour l'inpainting global")

    from .prompt_builder import build_full_garden_prompt_from_rag, build_global_context
    from .config import BFL_STEPS, BFL_GUIDANCE, BFL_STRENGTH

    # =========================================================
    # 1. Création du masque plantable intelligent
    # =========================================================
    combined_mask_path = masks_dir / "combined_plantable_mask.png"
    mask_bin_path = masks_dir / "combined_plantable_mask_bin.png"

    from .plantable_zone_generator import (
        create_combined_plantable_mask,
        debug_overlay,
    )

    print("🗺️  Génération du masque plantable...")

    if external_zones or external_plantable_mask_path:
        print("   [MASK] Utilisation des zones externes (code collègue)")
        mask_method = "external"
    else:
        print("   [MASK] Génération intelligente automatique")
        mask_method = "smart_auto"

    mask_pil, plants_with_bboxes = create_combined_plantable_mask(
        image_path=image_path,
        plants=plants_data,
        output_path=combined_mask_path,
        external_plantable_mask=external_plantable_mask_path,
        external_zones=external_zones,
    )

    # Binarisation stricte
    mask_arr = np.array(mask_pil)
    bin_arr = np.where(mask_arr >= 128, 255, 0).astype(np.uint8)
    mask_bin_pil = Image.fromarray(bin_arr, mode="L")
    mask_bin_pil.save(mask_bin_path)

    white_pct = 100.0 * np.sum(bin_arr == 255) / bin_arr.size
    print(f"   [MASK] Zones plantables : {white_pct:.1f}% de l'image")

    if debug:
        debug_overlay(
            image_path=image_path,
            mask=mask_bin_pil,
            output_path=outputs_dir / "mask_debug.png",
        )
        print(f"   [DEBUG] Overlay -> {outputs_dir / 'mask_debug.png'}")

    # =========================================================
    # 2. Construction du prompt consolidé
    # =========================================================
    global_context = build_global_context(metadata) if metadata else (global_style or "")
    prompt = build_full_garden_prompt_from_rag(
        metadata=metadata,
        plants=plants_with_bboxes,
        preserve_base=True,
    )
    print(f"   [PROMPT] {prompt[:120]}...")

    # =========================================================
    # 3. Appel API BFL Fill
    # =========================================================
    final_path = outputs_dir / "final_garden.png"
    raw_gen_path = outputs_dir / "final_garden_raw.png"

    print("🚀 Appel API BFL (Inpainting Global)...")
    inpaint(
        image_path=image_path,
        mask_path=mask_bin_path,
        prompt=prompt,
        out_path=raw_gen_path,
        seed=42,
        steps=BFL_STEPS,
        guidance=BFL_GUIDANCE,
        strength=BFL_STRENGTH,
    )

    # =========================================================
    # 4. Post-fusion : préservation garantie hors masque
    # =========================================================
    _preserve_original_outside_mask(
        original_path=image_path,
        generated_path=raw_gen_path,
        mask_path=mask_bin_path,
        out_path=final_path,
    )

    # =========================================================
    # 5. Données de sortie
    # =========================================================
    plants_out = []
    for i, p in enumerate(plants_with_bboxes):
        bbox = p.get("bbox", [0, 0, 100, 100])
        plants_out.append({
            "plant_id": p.get("plant_id", f"plant_{i:02d}"),
            "name": p.get("name", "plant"),
            "type": p.get("type", ""),
            "bbox": bbox,
            "centroid": [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2],
            "mask_path": str(combined_mask_path),
            "zone_hint": p.get("zone_hint", "midground_center"),
            "prompt_used": prompt,
            "editable": True,
            "layer_order": i,
            "status": "placed",
        })

    # =========================================================
    # 6. Relight nuit
    # =========================================================
    if time_of_day == "night" and final_path.exists():
        try:
            from ..utils.relight import relight_to_night
            relight_to_night(
                final_path,
                outputs_dir / "final_garden_night.png",
                light_intensity=night_light_intensity,
                plants=plants_out,
            )
        except Exception as e:
            print(f"[RELIGHT] Erreur : {e}")

    # =========================================================
    # 7. scene.json
    # =========================================================
    scene_dict = {
        "input_image": str(image_path),
        "final_image": str(final_path),
        "raw_generated": str(raw_gen_path),
        "metadata": metadata,
        "global_context": global_context,
        "mask_info": {
            "path": str(combined_mask_path),
            "bin_path": str(mask_bin_path),
            "white_pct": round(white_pct, 1),
            "method": mask_method,
            "debug_overlay": str(outputs_dir / "mask_debug.png") if debug else None,
            "external_zones_used": bool(external_zones or external_plantable_mask_path),
        },
        "plants": plants_out,
    }
    with open(outputs_dir / "scene.json", "w", encoding="utf-8") as f:
        json.dump(scene_dict, f, indent=2, ensure_ascii=False)

    from .mock_provider import create_preview_boxes
    create_preview_boxes(final_path, plants_out, outputs_dir / "preview_boxes.png")

    print(f"✅ Scène sauvegardée : {final_path}")
    print(f"📊 Masque : {white_pct:.1f}% zones plantables | méthode : {mask_method}")
    return scene_dict
