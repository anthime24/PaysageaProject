from __future__ import annotations
from typing import Any
import unicodedata

ADDITIVE_BASE = ("Use the input photo as the base. Keep camera angle, perspective, lighting, shadows, and all existing elements exactly the same. Preserve the original photo completely. Same composition and color grading. ")
ADDITIVE_VISIBLE = ("Add clearly noticeable plants ONLY in the masked areas. The added plants must be clearly visible and recognizable (not subtle). Keep image sharp, no blur, no haze. Do not change lawn, do not change trees, do not change sky. Photorealistic, seamless blend. No text, no labels, no overlays. ")
ADDITIVE_NEGATIVE = ("DO NOT add any new objects or landscaping elements. Only add the requested plants. DO NOT create new flower beds or change terrain. No new soil beds, no edging, no mulch, no rocks, no gravel. No paths, no benches, no ponds, no lights, no fences, no walls. DO NOT add people or animals. DO NOT change season or time of day. DO NOT repaint the whole image.")
MASK_CONSTRAINT = ("ONLY paint inside the masked area. Outside the mask must remain pixel-identical to the original.")
PLACEMENT_REALISTIC = ("Plants must be grounded in soil, with realistic scale and consistent shadows.")
PLACEMENT_RULES = ("Only inside masked area. Keep lighting and shadows consistent with the original photo.")
REDESIGN_BASE = ("Transform this garden into a well-designed landscaped garden. Multiple flowerbeds and planting borders. Photorealistic. No text, no labels, no overlays. ")
RELIGHT_NIGHT_PROMPT = ("Nighttime garden scene, realistic landscape lighting, warm ground spotlights illuminating plants, deep blue night sky, subtle ambient moonlight, keep composition identical, no new objects, photorealistic, no text.")
RAG_MUST_BE_VISIBLE = ("The following plants must appear clearly in the image, recognizable and well visible.")

_PLANT_VISUAL_DB = {
    "haie fleurie semi persistante tons rouges blanc": "flowering hedge shrub with dense branches, vivid red and white blooms, small oval green leaves, compact bushy shape, 200cm tall",
    "rosa banksiae lutea": "Rosa banksiae Lutea climbing rose, cascading clusters of tiny pale yellow double flowers, glossy dark green leaves, thornless arching stems, 200cm tall",
    "copie de rosa banksiae": "Rosa banksiae Lutea climbing rose, cascading clusters of tiny pale yellow double flowers, glossy dark green leaves, thornless arching stems, 200cm tall",
    "haie mediterraneenne mixte fleurie": "mixed Mediterranean flowering hedge with colorful blooms of pink, white and purple, dense evergreen foliage, varied shrub textures, 90cm tall",
    "haie fleurie caduque": "deciduous flowering hedge shrub, bright red blossoms, medium density branching, green leaves, 150cm tall",
    "jardin sec mediterraneen": "drought-tolerant Mediterranean planting, silver-grey foliage mix of lavender, rosemary and santolina, low spreading mound, 60cm tall",
    "rosmarinus officinalis": "Rosmarinus officinalis rosemary shrub, dense needle-like silver-green aromatic leaves, small violet-blue flowers along woody stems, rounded mounding habit, 80cm tall",
    "haie de photinia": "Photinia fraseri Red Robin hedge, vivid bright-red new leaf growth contrasting with dark glossy green mature leaves, dense upright shrub, 150cm tall",
    "photinia": "Photinia fraseri Red Robin, striking bright-red young shoots and dark glossy green mature leaves, dense upright evergreen shrub, 150cm tall",
    "prunus laurocerasus": "Prunus laurocerasus cherry laurel, large glossy dark-green oval leaves, dense evergreen canopy, white flower spikes in spring, 300cm tall",
}

_TYPE_VISUAL_DB = {
    "arbuste": "ornamental shrub with dense natural foliage, well-rooted in soil, rounded form",
    "graminee": "ornamental grass with long arching silver-green blades and feathery plumes",
    "fleur": "flowering plant with colorful blooms, petals clearly visible, lush foliage",
    "arbre": "small garden tree with structured trunk and seasonal canopy",
    "vivace": "perennial plant with lush foliage and seasonal flowers",
    "rosier": "rose bush with glossy leaves, large blooms in clusters, arching canes",
    "haie": "garden hedge with dense tightly-packed foliage, uniform upright height",
}

def _norm(s):
    s = unicodedata.normalize("NFD", (s or "").lower().strip())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

def _get_visual(plant):
    name_n = _norm(plant.get("name", ""))
    for key, desc in _PLANT_VISUAL_DB.items():
        if key in name_n or name_n in key:
            return desc
    for word in [w for w in name_n.split() if len(w) > 4]:
        for key, desc in _PLANT_VISUAL_DB.items():
            if word in key:
                return desc
    return _TYPE_VISUAL_DB.get((plant.get("type") or "").lower().strip(), "ornamental garden plant with natural foliage")

def build_single_plant_inpaint_prompt(plant, metadata, surrounding_context="", iteration=0):
    name = plant.get("name") or plant.get("species") or "garden plant"
    color = (plant.get("color") or "").replace("_", " ")
    visual = _get_visual(plant)
    color_hint = f"Dominant color: {color}. " if color and color.lower() not in visual.lower() else ""
    style = metadata.get("style") or metadata.get("climat") or metadata.get("climate") or ""
    style_desc = f"Maintain {style} garden aesthetic. " if style else ""
    surrounding = f"Harmonize with existing plants: {surrounding_context}. " if surrounding_context and iteration > 0 else ""
    prompt = (f"Add a single {name} planted naturally in the ground, ONLY inside the white masked area. {visual}. {color_hint}Photorealistic, match existing garden lighting, perspective and shadows. Roots firmly in the soil, no floating effect, no dark halo at base. {style_desc}{surrounding}CRITICAL: Do NOT modify ANYTHING outside the masked area. No text, no labels, no people.")
    return " ".join(prompt.split())

def build_global_context(metadata):
    parts = []
    if metadata.get("style"): parts.append(f"{metadata['style']} garden")
    if metadata.get("season") and metadata["season"] != "toutes_saisons": parts.append(metadata["season"])
    if metadata.get("climate"): parts.append(metadata["climate"])
    if metadata.get("sun_exposure"): parts.append(metadata["sun_exposure"].replace("_", " "))
    return ", ".join(parts)

def build_inpaint_prompt(plant_name):
    return (f"Add a highly visible and realistic {plant_name} planted in the ground inside the masked area. Vibrant colors, clear details. Photorealistic. Match lighting and perspective. PRESERVE THE ORIGINAL PHOTO COMPLETELY. ONLY modify the masked area. No text, no labels, no overlays.")

def build_full_garden_prompt(plant_density="medium", preserve_base=True, force_full_redesign=False, plant_list=None):
    if force_full_redesign or not preserve_base:
        return f"{REDESIGN_BASE} Dense flowerbeds, numerous plants."
    plant_block = f" Use these plants: {', '.join(plant_list[:15])}." if plant_list else ""
    return f"{ADDITIVE_BASE} {ADDITIVE_VISIBLE} Several clearly visible plants.{plant_block} {MASK_CONSTRAINT} {PLACEMENT_REALISTIC} {ADDITIVE_NEGATIVE}"

def build_full_garden_prompt_from_rag(metadata, plants, plant_density="medium", preserve_base=True, plant_list=None, debug=False):
    base = f"{ADDITIVE_BASE} {ADDITIVE_VISIBLE}" if preserve_base else REDESIGN_BASE
    names = plant_list or [p.get("name") or p.get("species") for p in plants if p.get("name") or p.get("species")]
    names = [n for n in names if n and n != "plant"][:15]
    parts = []
    if metadata.get("style"): parts.append(f"Style: {metadata['style']}.")
    if metadata.get("description"): parts.append(f"Ambiance: {metadata['description']}.")
    if names: parts.append(f"Use these plants: {', '.join(names)}.")
    return f"{base} {MASK_CONSTRAINT} {PLACEMENT_REALISTIC} {ADDITIVE_NEGATIVE} RAG: {' '.join(parts)}"

def build_plant_prompt(plant, metadata=None):
    parts = [f"realistic {plant.get('name') or 'garden plant'}"]
    if plant.get("type"): parts.append(plant["type"])
    parts.append("photorealistic, match lighting. No text, no labels.")
    if metadata:
        ctx = build_global_context(metadata)
        if ctx: parts.insert(1, ctx)
    return ", ".join(parts)

def build_prompt(plant, global_style=None):
    return build_plant_prompt(plant, {"style": global_style} if global_style else None)

def build_negative_prompt():
    return "cartoon, CGI, 3d render, text, watermark, logo, distorted, blurry, low quality, oversaturated, artificial, fake plant, plastic, deformed"
