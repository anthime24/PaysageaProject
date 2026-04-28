"""
Provider BFL FLUX.1 Fill PRO - inpainting via API.

POST image + mask (base64) -> poll -> télécharge result.sample.
"""
from __future__ import annotations

import base64
import math
import os
import time
from pathlib import Path
from typing import Union

from .config import BFL_GUIDANCE, BFL_STEPS

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

BFL_API_URL = "https://api.bfl.ai/v1/flux-pro-1.0-fill"
POLL_INTERVAL = 2.0
POLL_TIMEOUT = 300
# L'API BFL exige largeur ET hauteur >= 256 px (sinon HTTP 422).
BFL_MIN_SIDE = 256


def has_bfl_key() -> bool:
    """Retourne True si BFL_API_KEY est définie et non vide."""
    return bool(os.getenv("BFL_API_KEY", "").strip())


def _encode_image(path: Union[str, Path]) -> str:
    """Encode une image en base64."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image non trouvée : {path}")
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("ascii")


def _post_inpaint(
    image_b64: str,
    mask_b64: str,
    prompt: str,
    api_key: str,
    steps: int = BFL_STEPS,
    guidance: float = BFL_GUIDANCE,
    seed: int | None = None,
    strength: float = 0.85,
) -> str:
    """
    Envoie la requête d'inpainting, retourne polling_url.
    """
    if not HAS_REQUESTS:
        raise ImportError("pip install requests")

    # Validation payload BFL
    steps = max(15, min(50, int(steps)))
    guidance = max(1.5, min(100.0, float(guidance)))
    strength = max(0.0, min(1.0, float(strength)))
    if seed is not None:
        seed = int(seed)

    body = {
        "image": image_b64,
        "mask": mask_b64,
        "prompt": prompt,
        "steps": steps,
        "guidance": guidance,
        "strength": strength,
        "output_format": "png",
    }
    if seed is not None:
        body["seed"] = seed

    print("BFL API CALL START")
    print(f"  BFL_API_KEY loaded: {'yes' if (api_key and api_key.strip()) else 'no'}")
    print(f"  Endpoint: {BFL_API_URL}")
    print(f"  Steps: {steps} | Guidance: {guidance} | Seed: {seed}")
    print(f"  Prompt: {(prompt[:80] + '...' if len(prompt) > 80 else prompt)}")

    headers = {"x-key": api_key, "Content-Type": "application/json"}
    resp = requests.post(
        BFL_API_URL,
        headers=headers,
        json=body,
        timeout=60,
    )

    print(f"  Status code: {resp.status_code}")

    if resp.status_code >= 400:
        try:
            err_json = resp.json()
            err_msg = err_json.get("message", err_json.get("error", str(err_json)))
        except Exception:
            err_msg = resp.text[:500]

        print("----- BFL ERROR (4xx/5xx) -----")
        print(f"Status: {resp.status_code}")
        print("Response:", err_msg)
        print("--------------------------------")

        if resp.status_code in (401, 403):
            raise RuntimeError("BFL: clé API invalide ou non autorisée (401/403)")
        if resp.status_code == 402:
            raise RuntimeError("BFL: crédits insuffisants (402)")
        raise RuntimeError(f"BFL erreur HTTP {resp.status_code}: {err_msg}")
    data = resp.json()
    polling_url = data.get("polling_url")
    if not polling_url:
        raise RuntimeError(f"Pas de polling_url dans la réponse : {data}")
    print(f"  Polling URL: {polling_url[:60]}...")
    return polling_url


def _poll_and_download(
    polling_url: str,
    api_key: str,
    out_path: Union[str, Path],
) -> None:
    """
    Poll jusqu'à status Ready, télécharge result.sample.
    """
    if not HAS_REQUESTS:
        raise ImportError("pip install requests")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    elapsed = 0.0

    while elapsed < POLL_TIMEOUT:
        resp = requests.get(
            polling_url,
            headers={"x-key": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "").lower()
        if status == "ready":
            result = data.get("result") or data
            sample_url = result.get("sample") or result.get("output") or result.get("url")
            if not sample_url:
                raise RuntimeError(f"Pas de sample dans la réponse : {data}")

            # Télécharger immédiatement (URL expire vite)
            img_resp = requests.get(sample_url, timeout=60)
            img_resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(img_resp.content)
            return

        if status in ("failed", "error"):
            msg = data.get("message", data.get("error", str(data)))
            raise RuntimeError(f"BFL Inpainting échoué : {msg}")

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    raise TimeoutError(f"Timeout après {POLL_TIMEOUT}s")


def inpaint(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    prompt: str,
    out_path: Union[str, Path],
    seed: int | None = None,
    steps: int = BFL_STEPS,
    guidance: float = BFL_GUIDANCE,
    strength: float = 0.85,
) -> None:
    """
    Inpaint une zone de l'image via BFL FLUX.1 Fill PRO.

    Args:
        image_path: Chemin image source
        mask_path: Chemin masque (noir=conserver, blanc=modifier)
        prompt: Description du contenu à générer
        out_path: Chemin de sortie (PNG)
        seed: Seed optionnel
        steps: Nombre de steps (défaut 30)
        guidance: Guidance (défaut 50)
    """
    api_key = os.environ.get("BFL_API_KEY", "")
    print("BFL_PROVIDER CALLED")
    print(f"  BFL_API_KEY loaded: {'yes' if api_key else 'no'}")
    if not api_key:
        raise RuntimeError(
            "BFL_API_KEY non définie. Définir : export BFL_API_KEY='votre_clé'"
        )
    api_key = api_key.strip()

    # Image RGB et masque L en PNG lossless ; un seul resize si masque ≠ image (ratio conservé)
    import io
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    mask_img = Image.open(mask_path).convert("L")
    if mask_img.size != img.size:
        mask_img = mask_img.resize(img.size, Image.Resampling.NEAREST)

    w, h = img.size
    scale_up = 1.0
    if w < BFL_MIN_SIDE or h < BFL_MIN_SIDE:
        scale_up = max(BFL_MIN_SIDE / w, BFL_MIN_SIDE / h)
        nw = max(BFL_MIN_SIDE, int(math.ceil(w * scale_up)))
        nh = max(BFL_MIN_SIDE, int(math.ceil(h * scale_up)))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        mask_img = mask_img.resize((nw, nh), Image.Resampling.NEAREST)
        print(
            f"  BFL min {BFL_MIN_SIDE}px: upscale {orig_w}x{orig_h} -> {nw}x{nh} "
            f"(API BFL refuse les images < {BFL_MIN_SIDE}px sur un côté)"
        )

    buf_img = io.BytesIO()
    img.save(buf_img, format="PNG")
    image_b64 = base64.standard_b64encode(buf_img.getvalue()).decode("ascii")

    buf_mask = io.BytesIO()
    mask_img.save(buf_mask, format="PNG")
    mask_b64 = base64.standard_b64encode(buf_mask.getvalue()).decode("ascii")

    print(f"  SAFE: image size sent to BFL: {img.size[0]}x{img.size[1]} (PNG lossless)")
    polling_url = _post_inpaint(
        image_b64, mask_b64, prompt, api_key, steps, guidance, seed, strength
    )
    print(f"   Polling...")
    _poll_and_download(polling_url, api_key, out_path)
    # Remettre aux dimensions d'entrée pour composite avec le masque à taille native
    if scale_up > 1.0:
        out_img = Image.open(out_path).convert("RGB")
        if out_img.size != (orig_w, orig_h):
            out_img = out_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            out_img.save(out_path)
            print(f"  Downscale résultat BFL -> {orig_w}x{orig_h} pour fusion locale")
    print(f"   OK : {out_path}")
