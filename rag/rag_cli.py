"""
rag_cli.py — Interface CLI pour le RAG Paysagea
================================================
Lit les préférences utilisateur (JSON via stdin ou --prefs fichier),
filtre et score les plantes de plantes_data.json,
retourne un JSON au format rag_output.json attendu par generate_garden_cli.py.

Usage:
    echo '{"style":"japonais","exposition":"plein_soleil","n_plants":6}' | python rag_cli.py
    python rag_cli.py --prefs prefs.json --n-plants 6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ── Mapping sous_type_excel → type interne ────────────────────────────────────
_TYPE_MAP: dict[str, str] = {
    "arbuste":              "arbuste",
    "arbuste nain":         "arbuste",
    "buisson":              "arbuste",
    "arbre":                "arbre",
    "arbre fruitier":       "arbre",
    "palmier":              "arbre",
    "vivace":               "vivace",
    "aromate":              "vivace",
    "bulbe":                "fleur",
    "annuelle":             "fleur",
    "grimpante":            "arbuste",
    "graminée":             "graminee",
    "graminee":             "graminee",
    "rosier":               "rosier",
    "haie":                 "haie",
    "vivace / couvre-sol":  "couvre_sol",
    "vivace / succulente":  "vivace",
}
_EXCLUDE_TYPES = {
    "mobilier de jardin", "décoration", "decoration",
    "contenant / poterie", "circulation",
    "loisir aquatique", "plante en pot",
}

# Mots couleur (FR) → termes à chercher dans le champ couleur de la plante
_COLOR_KEYWORDS: dict[str, list[str]] = {
    "bleu":    ["bleu", "blue", "azur", "indigo"],
    "bleue":   ["bleu", "blue", "azur", "indigo"],
    "bleues":  ["bleu", "blue", "azur", "indigo"],
    "violet":  ["violet", "mauve", "pourpre", "lilas"],
    "violette":["violet", "mauve", "pourpre", "lilas"],
    "rose":    ["rose", "pink"],
    "blanc":   ["blanc", "white", "crème", "creme"],
    "blanche": ["blanc", "white", "crème", "creme"],
    "jaune":   ["jaune", "yellow", "or", "doré"],
    "rouge":   ["rouge", "red", "écarlate", "cramoisi"],
    "orange":  ["orange"],
    "pourpre": ["pourpre", "bordeaux", "burgund"],
}

# Mots-clés style → score bonus
_STYLE_KEYWORDS: dict[str, list[str]] = {
    "japonais":       ["japon", "acer", "maple", "érable", "bambou", "mousse", "zen", "cerisier"],
    "mediterraneen":  ["olivier", "lavande", "romarin", "thym", "agave", "citrus", "cactus", "aloe"],
    "moderne":        ["graphique", "architectural", "graminée", "if", "buis", "miscanthus"],
    "naturel":        ["prairie", "sauvage", "graminée", "vivace", "rudbeckia", "echinacea"],
    "potager":        ["tomate", "fraise", "thym", "basilic", "aromate", "potagère"],
    "champetre":      ["rosier", "lilas", "pivoine", "géranium", "digitale", "campanule"],
}


def _map_type(sous_type: str) -> str:
    low = (sous_type or "").lower().strip()
    if low in _TYPE_MAP:
        return _TYPE_MAP[low]
    for key, val in _TYPE_MAP.items():
        if key in low:
            return val
    return "vivace"


def _map_exposition(ensoleillement: str) -> str:
    low = (ensoleillement or "").lower()
    if "mi" in low or "semi" in low:
        return "mi_ombre"
    if "ombre" in low and "mi" not in low:
        return "ombre"
    return "plein_soleil"


def _map_color(couleur: str) -> str:
    if not couleur:
        return ""
    return re.split(r"[,/]", couleur)[0].strip().lower()


def _normalize_expo_pref(exposition: str) -> str:
    """Normalise l'exposition envoyée depuis le frontend."""
    mapping = {
        "soleil":       "plein_soleil",
        "plein_soleil": "plein_soleil",
        "sun":          "plein_soleil",
        "mi-ombre":     "mi_ombre",
        "mi_ombre":     "mi_ombre",
        "partial":      "mi_ombre",
        "ombre":        "ombre",
        "shade":        "ombre",
    }
    return mapping.get((exposition or "").lower().strip(), "plein_soleil")


def _normalize_style(style: str) -> str:
    """Normalise le style frontend (peut contenir des tirets, suffixes, etc.)."""
    low = (style or "naturel").lower()
    for key in _STYLE_KEYWORDS:
        if key in low:
            return key
    return "naturel"


def _load_plants(data_file: Path) -> list[dict]:
    """Charge plantes_data.json (format multi-tableaux [] [] concatenés)."""
    content = data_file.read_text(encoding="utf-8")
    # Le fichier contient plusieurs tableaux JSON consécutifs : ][\n[
    parts = re.split(r"\]\s*\[", content)
    plants: list[dict] = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part.startswith("["):
            part = "[" + part
        if not part.endswith("]"):
            part = part + "]"
        try:
            plants.extend(json.loads(part))
        except json.JSONDecodeError:
            pass
    return plants


def _filter_and_score(plants: list[dict], prefs: dict) -> list[dict]:
    style = _normalize_style(prefs.get("style", "naturel"))
    expo_pref = _normalize_expo_pref(prefs.get("exposition", ""))
    description = (prefs.get("description") or "").lower()
    entretien = (prefs.get("entretien") or "moyen").lower()
    usda_zone = prefs.get("usda_zone")   # ex: 9 → méditerranéen
    temp_min = prefs.get("temp_min")     # ex: -5.5°C → filtre rusticité
    keywords = _STYLE_KEYWORDS.get(style, [])

    # Extraire les mots-couleur demandés dans la description
    wanted_color_terms: list[str] = []
    for word in description.split():
        if word in _COLOR_KEYWORDS:
            wanted_color_terms.extend(_COLOR_KEYWORDS[word])

    result = []
    for p in plants:
        sous_type = (p.get("sous_type_excel") or "").lower().strip()

        # Exclure le mobilier / décorations et les doublons "Copie de"
        if sous_type in _EXCLUDE_TYPES:
            continue
        nom = p.get("nom", "")
        if nom.lower().startswith("copie de") or nom.lower().startswith("copy of"):
            continue

        type_mapped = _map_type(p.get("sous_type_excel", ""))
        expo_plant = _map_exposition(p.get("ensoleillement", ""))

        score = 0

        # Bonus style
        nom_low = p.get("nom", "").lower()
        if any(kw in nom_low for kw in keywords):
            score += 3

        # Bonus exposition compatible
        if expo_pref == expo_plant:
            score += 2
        elif expo_pref == "mi_ombre" and expo_plant in ("plein_soleil", "ombre"):
            score += 0
        elif expo_pref == "plein_soleil" and expo_plant == "mi_ombre":
            score += 1  # toléré

        # Bonus description textuelle (mots > 4 chars dans le nom)
        if description:
            desc_words = [w for w in description.split() if len(w) > 4]
            if any(w in nom_low for w in desc_words):
                score += 2

        # Bonus/malus couleur : forte priorité si l'utilisateur demande une couleur
        if wanted_color_terms:
            couleur_low = (p.get("couleur") or "").lower()
            if any(ct in couleur_low for ct in wanted_color_terms):
                score += 5  # forte récompense = remonte en tête de liste
            else:
                score -= 2  # pénalité légère pour les autres couleurs

        # Bonus entretien faible
        if entretien == "faible":
            eau_str = str(p.get("besoin_eau", ""))
            if "0" in eau_str or "1" in eau_str:
                score += 1

        # Rusticité : filtrage strict si temp_min connu, sinon bonus
        rusticite = (p.get("rusticite_valeur") or "").lower()
        if temp_min is not None:
            # Extraire la température min de la rusticité (ex: "-5°C" → -5)
            import re as _re
            m = _re.search(r"(-?\d+)", rusticite)
            plant_temp_min = int(m.group(1)) if m else 0
            if plant_temp_min > temp_min + 2:
                # Plante non rustique pour ce climat : pénalité forte
                score -= 5
            elif plant_temp_min <= temp_min:
                score += 2  # bonne rusticité
        else:
            if "très rustique" in rusticite or "-20" in rusticite or "-15" in rusticite:
                score += 1

        # Bonus zone USDA : méditerranéen (zone 9-10) → favoriser les plantes méditerranéennes
        if usda_zone is not None and usda_zone >= 9:
            mots_med = ["lavande", "romarin", "olivier", "thym", "agave", "ciste", "cytise",
                        "oleander", "laurier", "mimosa", "pittosporum", "euphorbe"]
            if any(m in nom_low for m in mots_med):
                score += 3

        result.append({**p, "_type": type_mapped, "_score": score})

    result.sort(key=lambda x: x["_score"], reverse=True)
    return result


def _convert(p: dict, index: int, style: str) -> dict:
    """Convertit une plante vers le format rag_output.json."""
    return {
        "plant_id":        f"plant_{index + 1:02d}",
        "name":            p.get("nom", f"plant_{index}"),
        "type":            p["_type"],
        "height_cm":       100,
        "width_cm":        80,
        "density":         "medium",
        "color":           _map_color(p.get("couleur", "")),
        "climate":         "tempere",
        "sun_exposure":    _map_exposition(p.get("ensoleillement", "")),
        "season":          "printemps",
        "water_needs":     "faible" if "0" in str(p.get("besoin_eau", "")) else "moyen",
        "soil_preference": p.get("type_sol", ""),
        "maintenance_level": (p.get("entretien") or "").split("(")[0].strip(),
        "zone_hint":       "midground_center",
        "style_tags":      [style],
        "reason":          (
            f"{p.get('sous_type_excel', '')} · {p.get('couleur', '')} · "
            f"exposition: {p.get('ensoleillement', '')} · {p.get('rusticite_valeur', '')}"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG CLI Paysagea")
    parser.add_argument("--prefs", default="", help="Fichier JSON des préférences (sinon stdin)")
    parser.add_argument("--n-plants", type=int, default=6, help="Nombre de plantes à retourner")
    parser.add_argument(
        "--data-dir", default="",
        help="Chemin vers le dossier data/ contenant plantes_data.json",
    )
    args = parser.parse_args()

    # ── Charger les préférences ────────────────────────────────────────────────
    if args.prefs:
        prefs = json.loads(Path(args.prefs).read_text(encoding="utf-8"))
    else:
        prefs = json.loads(sys.stdin.read())

    # ── Localiser plantes_data.json ────────────────────────────────────────────
    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir / "data"
    data_file = data_dir / "plantes_data.json"

    if not data_file.exists():
        sys.stdout.write(json.dumps({"error": f"plantes_data.json introuvable: {data_file}"}))
        sys.exit(1)

    # ── Traitement ────────────────────────────────────────────────────────────
    all_plants = _load_plants(data_file)
    scored = _filter_and_score(all_plants, prefs)
    top = scored[: args.n_plants]

    style = _normalize_style(prefs.get("style", "naturel"))
    jardin = [_convert(p, i, style) for i, p in enumerate(top)]

    result = {
        "succes": True,
        "jardin": jardin,
        "infos": {
            "description": prefs.get("description", ""),
            "style":       style,
            "taille":      prefs.get("taille", "moyen"),
            "plant_count": len(jardin),
            "climat":      prefs.get("climat", "tempere"),
            "exposition":  prefs.get("exposition", "plein_soleil"),
            "budget":      prefs.get("budget", "moyen"),
            "entretien":   prefs.get("entretien", "moyen"),
            "region":      prefs.get("region", "France"),
        },
    }

    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
