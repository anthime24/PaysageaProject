"""
Design jardin global : un seul appel FLUX Fill avec masque plantable.

Mode "add plants only" par défaut : préserver l'image, ajouter des plantations dans les zones masquées.
BFL : blanc = zone à modifier, noir = zone à conserver.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
from PIL import Image

from .config import BFL_GUIDANCE as CONFIG_GUIDANCE, BFL_STEPS as CONFIG_STEPS, BFL_STRENGTH as CONFIG_STRENGTH
from .bfl_provider import has_bfl_key
from .plantable_mask import (
    compute_mask_white_percent,
    create_border_mask,
    create_fallback_mask_exclude_sky,
    create_fallback_mask_full,
    generate_plantable_mask,
    reduce_mask_to_borders,
)
from .prompt_builder import (
    RELIGHT_NIGHT_PROMPT,
    build_full_garden_prompt,
    build_full_garden_prompt_from_rag,
)

# Add-only : utiliser config (guidance/steps modérés)
GUIDANCE_ADD_PLANTS = CONFIG_GUIDANCE
GUIDANCE_REDESIGN = CONFIG_GUIDANCE
STEPS_ADD_PLANTS = CONFIG_STEPS
STEPS_REDESIGN = CONFIG_STEPS
MASK_TOO_LARGE_THRESHOLD = 60.0  # % blanc au-delà = risque redesign


def _load_rag_data(raw: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extrait metadata et liste plantes depuis JSON RAG."""
    if isinstance(raw, list):
        return {}, raw
    metadata = {}
    if isinstance(raw, dict):
        metadata = raw.get("metadata", raw)
        if not isinstance(metadata, dict):
            metadata = {}
        plants = raw.get("garden") or raw.get("plants") or raw.get("recommendations") or raw.get("plants_data") or []
    else:
        plants = []
    return metadata, plants


def _extract_plant_list(plants: list[dict[str, Any]], max_items: int = 15) -> list[str]:
    """Extrait la liste des noms de plantes (name/species) depuis le RAG."""
    names: list[str] = []
    seen: set[str] = set()
    for p in plants[:max_items]:
        n = p.get("name") or p.get("species")
        if n and isinstance(n, str) and n.strip() and n.strip().lower() not in seen:
            names.append(n.strip())
            seen.add(n.strip().lower())
    return names


def _log(msg: str, log_fn: Callable[[str], None] | None = None) -> None:
    print(msg)
    if log_fn:
        try:
            log_fn(msg)
        except Exception:
            pass


def generate_full_garden(
    image_path: Union[str, Path],
    outputs_dir: Union[str, Path] = "outputs",
    exclude_lawn: bool = True,
    plant_density: str = "medium",
    use_mask: bool = True,
    force_full_mask: bool = False,
    preserve_base: bool = True,
    return_debug: bool = True,
    time_of_day: str = "day",
    night_light_intensity: float = 0.5,
    rag_path: Path | str | None = None,
    log_fn: Callable[[str], None] | None = None,
    seed: int | None = None,
) -> Path | tuple[Path, dict[str, Any]]:
    """
    Un seul appel FLUX Fill avec masque plantable.

    preserve_base=True : mode "add plants only", conserver l'image.
    force_full_mask : DANGER — redesign complet (debug uniquement).
    return_debug=True : retourne (path, debug_info), sinon path seul.
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = outputs_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image non trouvée : {image_path}")
    with Image.open(image_path) as img_check:
        w0, h0 = img_check.size
    _log(f"   SAFE: image size={w0}x{h0}", log_fn)

    mask_path = masks_dir / "plantable_mask.png"
    final_path = outputs_dir / "final_garden.png"
    final_night_path = outputs_dir / "final_garden_night.png"

    force_full_redesign = force_full_mask or not use_mask
    mode = "Redesign" if force_full_redesign else "Add plants only"

    used_mask_path = mask_path
    debug_info: dict[str, Any] = {
        "use_mock": False,
        "white_pct": 0.0,
        "used_fallback": False,
        "used_mask_path": "",
        "mask_path": str(mask_path),
        "prompt": "",
        "prompt_preview": "",
        "steps": STEPS_ADD_PLANTS,
        "guidance": GUIDANCE_ADD_PLANTS,
        "seed": 42,
        "mode": mode,
        "preserve_base": preserve_base,
        "force_full_mask": force_full_mask,
        "mask_too_large": False,
        "rag_plants_used": [],
        "white_pct_bin": None,
        "mask_bin_path": None,
    }

    # ⚠️ WARNING masque plein (redesign complet)
    if force_full_mask or (not use_mask):
        _log("⚠️ DANGEROUS: Full mask — REDESIGN RISK. Reserved for debug only.", log_fn)
        _log("   En prod, utilisez masque plantable (use_mask=True, force_full_mask=False).", log_fn)
        debug_info["mode"] = "Redesign"

    # 1. Mode MOCK / BFL
    bfl_key = os.environ.get("BFL_API_KEY", "").strip()
    use_mock = os.environ.get("MOCK_BFL", "").lower() == "true" or not bfl_key
    debug_info["use_mock"] = use_mock

    if use_mock:
        _log("⚠️ [MOCK MODE] Pas d'appel API BFL", log_fn)
    else:
        _log(f"[GARDEN] Mode: {mode}", log_fn)

    # 2. Masque
    white_pct = 100.0
    if force_full_mask:
        img = Image.open(image_path).convert("RGB")
        mask = Image.new("L", img.size, 255)
        mask.save(mask_path)
        used_mask_path = mask_path
        white_pct = 100.0
        debug_info["white_pct"] = 100.0
        debug_info["used_fallback"] = False
        debug_info["mask_too_large"] = True
        debug_info["used_mask_path"] = str(mask_path)
        _log("   [MASK] FORCE FULL MASK (debug)", log_fn)
    elif use_mask:
        mask_pil, white_pct, used_fallback = generate_plantable_mask(
            image_path=image_path,
            exclude_lawn=exclude_lawn,
            output_path=mask_path,
            min_white_percent=5.0,
        )
        debug_info["white_pct"] = white_pct
        debug_info["used_fallback"] = used_fallback

        if white_pct < 5.0:
            mask_pil = create_fallback_mask_exclude_sky(image_path)
            mask_pil.save(mask_path)
            white_pct = compute_mask_white_percent(mask_pil)
            debug_info["white_pct"] = white_pct

        # Sanity check : masque trop large => border mask (contour uniquement)
        if white_pct > MASK_TOO_LARGE_THRESHOLD and preserve_base:
            _log(f"   [SANITY] Masque {white_pct:.1f}% > {MASK_TOO_LARGE_THRESHOLD}% — passage au masque bordures", log_fn)
            _log("   SAFE: fallback border mask — no global redesign", log_fn)
            debug_info["mask_too_large"] = True
            border_path = masks_dir / "plantable_mask_border.png"
            mask_pil, white_pct = create_border_mask(
                mask_path, erosion_pixels=15, output_path=border_path
            )
            used_mask_path = border_path
            mask_path = border_path  # utilisé pour inpaint
            debug_info["white_pct"] = white_pct
            debug_info["used_mask_path"] = str(border_path)
            _log(f"   [MASK] Bordures seules → {white_pct:.1f}% blanc", log_fn)
        else:
            used_mask_path = mask_path
            debug_info["used_mask_path"] = str(mask_path)
            _log(f"   [MASK] plantable | blanc={white_pct:.1f}%", log_fn)
    else:
        img = Image.open(image_path).convert("RGB")
        mask = Image.new("L", img.size, 255)
        mask.save(mask_path)
        used_mask_path = mask_path
        white_pct = 100.0
        debug_info["white_pct"] = 100.0
        debug_info["mask_too_large"] = True
        debug_info["used_mask_path"] = str(mask_path)
        _log("   [MASK] masque plein", log_fn)

    # 3. Prompt (mode additive ou redesign) — dépend du RAG si fourni
    actual_preserve = preserve_base and not force_full_redesign
    plant_list: list[str] = []
    if rag_path and Path(rag_path).exists():
        with open(rag_path, encoding="utf-8") as f:
            raw = json.load(f)
        metadata, plants = _load_rag_data(raw)
        plant_list = _extract_plant_list(plants, max_items=15)
        debug_info["rag_plants_used"] = plant_list
        rag_data = {"metadata": metadata, "garden": plants}
        prompt = build_full_garden_prompt_from_rag(
            metadata, plants, plant_density, preserve_base=actual_preserve, plant_list=plant_list
        )
        _log(f"   [PROMPT] RAG ({len(plant_list)} plantes)" + (" add plants" if actual_preserve else " redesign"), log_fn)
        (outputs_dir / "last_rag_used.json").write_text(
            json.dumps(rag_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        prompt = build_full_garden_prompt(
            plant_density=plant_density,
            preserve_base=actual_preserve,
            force_full_redesign=force_full_redesign,
            plant_list=None,
        )
        _log(f"   [PROMPT] défaut (add plants)" if actual_preserve else "   [PROMPT] défaut (redesign)", log_fn)

    debug_info["prompt"] = prompt
    debug_info["prompt_preview"] = prompt[:200] + ("..." if len(prompt) > 200 else "")
    debug_info["used_mask_path"] = str(used_mask_path)
    (outputs_dir / "last_prompt.txt").write_text(prompt, encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    (outputs_dir / "debug_prompt_hash.txt").write_text(prompt_hash, encoding="utf-8")

    guidance = GUIDANCE_REDESIGN if force_full_redesign else GUIDANCE_ADD_PLANTS
    steps = STEPS_ADD_PLANTS if actual_preserve else STEPS_REDESIGN
    strength = CONFIG_STRENGTH
    if seed is None:
        try:
            seed = int(os.environ.get("DEBUG_SEED", "42"))
        except (TypeError, ValueError):
            seed = 42
    seed = int(seed)
    # Strength dynamique selon taille du masque (éviter texture/halo si masque grand)
    white_pct_ratio = white_pct / 100.0
    mask_mode = "border" if debug_info.get("mask_too_large") else "plantable"
    if white_pct_ratio > 0.60:
        strength = min(strength, 0.65)
    elif white_pct_ratio > 0.45:
        strength = min(strength, 0.70)
    strength_final = strength
    debug_info["guidance"] = guidance
    debug_info["steps"] = steps
    debug_info["strength"] = strength_final
    debug_info["seed"] = seed
    debug_info["mask_mode"] = mask_mode
    _log(f"   SAFE: white_pct={white_pct:.1f} mask_mode={mask_mode} strength_final={strength_final}", log_fn)
    _log(f"   SAFE: params guidance={guidance} steps={steps} strength_final={strength_final} seed={seed}", log_fn)
    if plant_list:
        _log(f"   SAFE: rag_plants_used={plant_list}", log_fn)

    # 4. Appel FLUX Fill
    if use_mock:
        _log("   [MOCK] Copie image", log_fn)
        Image.open(image_path).convert("RGB").save(final_path)
    else:
        # Masque 8-bit binarisé (0/255) avant inpaint — évite niveaux de gris → flou
        mask_img = Image.open(used_mask_path).convert("L")
        arr = np.array(mask_img)
        bin_arr = np.where(arr > 127, 255, 0).astype(np.uint8)
        mask_bin = Image.fromarray(bin_arr, mode="L")
        bin_path = masks_dir / "plantable_mask_bin.png"
        mask_bin.save(bin_path)
        white_pct_bin = 100.0 * np.sum(bin_arr == 255) / arr.size
        debug_info["white_pct_bin"] = round(white_pct_bin, 1)
        debug_info["mask_bin_path"] = str(bin_path)
        _log(f"   [MASK] binarisé 8-bit → {white_pct_bin:.1f}% blanc → {bin_path.name}", log_fn)
        _log(f"   SAFE: white_pct={white_pct_bin:.1f}", log_fn)

        from .bfl_provider import inpaint

        _log("   [BFL] Appel inpaint()...", log_fn)
        inpaint(
            image_path=image_path,
            mask_path=str(bin_path),
            prompt=prompt,
            out_path=final_path,
            seed=seed,
            steps=steps,
            guidance=guidance,
            strength=strength_final,
        )
        _log("   [BFL] Terminé.", log_fn)

        # --- Post-processing: overlay original hors masque (anti-halo avec bord feathered) ---
        # Masque binaire envoyé à BFL inchangé. Pour le compositing final uniquement, on utilise
        # un alpha lissé (blur 3–7 px) pour fondre la jonction et éviter un halo.
        try:
            if preserve_base and os.path.exists(final_path):
                original_img = Image.open(image_path).convert("RGB")
                generated_img = Image.open(final_path).convert("RGB")
                mask_for_overlay = Image.open(bin_path).convert("L")
                orig_arr = np.array(original_img, dtype=np.float32)
                gen_arr = np.array(generated_img, dtype=np.float32)
                # Alpha pour overlay : 0 = original, 1 = généré. On floute le bord du masque.
                mask_float = (np.array(mask_for_overlay, dtype=np.float32) / 255.0)
                try:
                    from scipy.ndimage import gaussian_filter
                    alpha = gaussian_filter(mask_float, sigma=4.0)
                except ImportError:
                    alpha = mask_float
                alpha = np.clip(alpha, 0.0, 1.0)
                alpha_3 = alpha[..., np.newaxis]
                combined_arr = (orig_arr * (1.0 - alpha_3) + gen_arr * alpha_3).astype(np.float32)
                combined_arr = np.clip(combined_arr, 0, 255).astype(np.uint8)
                Image.fromarray(combined_arr).save(final_path)
                _log("   [POST] Fusion originale + inpainting (bord feathered), output PNG", log_fn)
                _log("   SAFE: overlay applied — base preserved outside mask", log_fn)
        except Exception as e:
            _log(f"   [POST] Fusion échouée: {e}", log_fn)

    # Log SAFE consolidé par génération
    _log(
        f"   SAFE: gen_done image_size={w0}x{h0} guidance={debug_info.get('guidance')} steps={debug_info.get('steps')} "
        f"strength_final={debug_info.get('strength')} seed={debug_info.get('seed')} "
        f"white_pct={debug_info.get('white_pct_bin') or debug_info.get('white_pct')} mask_mode={debug_info.get('mask_mode', '')} "
        f"rag_plants={debug_info.get('rag_plants_used', [])} output={final_path}",
        log_fn,
    )
    _log(f"   [OUT] {final_path}", log_fn)

    # 5. Relight nuit
    if time_of_day == "night" and final_path.exists():
        if use_mock:
            import sys
            root = Path(__file__).resolve().parent.parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from utils.relight import relight_to_night
            relight_to_night(
                final_path, final_night_path,
                strength=0.85, light_intensity=night_light_intensity, plants=None,
            )
        else:
            from .bfl_provider import inpaint
            full_mask_path = masks_dir / "_relight_full_mask.png"
            img = Image.open(final_path).convert("RGB")
            Image.new("L", img.size, 255).save(full_mask_path)
            inpaint(
                image_path=final_path,
                mask_path=full_mask_path,
                prompt=RELIGHT_NIGHT_PROMPT,
                out_path=final_night_path,
                seed=43, steps=30, guidance=25,
            )
        _log(f"   [OUT] {final_night_path}", log_fn)
        out_path = final_night_path
    else:
        out_path = final_path

    return (out_path, debug_info) if return_debug else out_path
