# Paysagea — paquet « final » (scripts à intégrer)



---

## Contenu (structure minimale métier)

| Élément | Rôle |
|--------|------|
| `run_full_phase1_b_sam_depth_zone.py` | Orchestrateur Phase 1-B : SAM → préprocess → profondeur → fusion VisionOutput → export `main.json` → UI zone (polygone) → `integration/final_scene_input.json` |
| `SAM_for_paysagea/` | Pipeline auto SAM + préprocess (`auto_pipeline/pipeline_auto.py`, `preprocess/`, lib `segment_anything/`) |
| `Depth-Anything/` | Profondeur, fusion SAM+depth, export vision (`run_depth_paysagea.py`, `fuse_sam_depth.py`, `export_vision_v0.py`) + dépendances locales (ex. `torchhub/`) |
| `zone-selection/app/` | Outil polygone / brush + export `user_zone.json` |
| `integration/merge_user_zone_with_vision.py` | Fusion `main.json` (vision) + zone utilisateur → `final_scene_input.json` |
| `garden_ia_3/` | **Garden AI** : génération BFL (zone plantable + masques), UI Streamlit (`ui/app.py`), prompts, RAG, etc. |
| `Inputs/` | Dossier vide (référence) : l’orchestrateur y copie l’image préprocessée pour l’alignement UI |

---

## Flux fonctionnel cible (aligné avec le front)

1. **Entrée données** — JSON Vision + sorties SAM / préprocess (produits par la Phase 1-B ou équivalent).
2. **Zone plantable** — polygone / masque utilisateur (`zone-selection` → `zone-selection/outputs/user_zone.json` + PNG).
3. **Génération d’images** — `garden_ia_3` + clé **BFL** + masque `user_zone_mask.png` (voir Streamlit et `dispatch_generation`).
4. **Édition des masques** — onglet « Éditer masques » dans `garden_ia_3/ui/app.py` → PNG + `scene_masks_edited.json`.

---

## Prérequis

- **Python 3.10+** recommandé (3.13 utilisé côté projet).
- Environnement virtuel : à la racine du clone (parent de `final/`), par ex. `.venv`.
- Dépendances : `pip install -r garden_ia_3/requirements.txt` (et prérequis **SAM** / **Depth-Anything** selon tes `requirements` dans `SAM_for_paysagea/preprocess/` ou doc Depth).
- **`BFL_API_KEY`** pour la génération réelle (sinon mode mock si prévu).
- **Checkpoint SAM** (`.pth`) : à placer où tu veux et passer `--sam-checkpoint` à l’orchestrateur (ex. `SAM_for_paysagea/sam_vit_b_01ec64.pth` — non inclus ici s’il est trop lourd pour Git).

---

## Commandes utiles

À lancer depuis la **racine du dépôt** (là où se trouvent `final/`, `Inputs/`, etc.) — adapter si tu n’utilises que le sous-dossier `final/` seul :

```bash
# Phase 1-B complète (SAM + depth + zone + merge)
python run_full_phase1_b_sam_depth_zone.py --image Inputs/ma_photo.jpg

# Garden AI (génération + édition masques)
cd garden_ia_3
streamlit run ui/app.py
```

Sous Windows PowerShell, pense à activer le venv et à exporter `BFL_API_KEY`.

---

## Intégration côté collègue (front)

- **Contrat fichiers** : même noms et rôles que ci-dessus (`main.json`, `integration/final_scene_input.json`, `zone-selection/outputs/user_zone_mask.png`, etc.).
- **Option A** : cloner tout le dépôt et travailler à la racine (recommandé).
- **Option B** : copier uniquement le dossier `final/` dans son repo, en conservant les **chemins relatifs** entre `garden_ia_3`, `SAM_for_paysagea`, `Depth-Anything`, etc.
- **Option C** : importer les modules Python depuis `garden_ia_3` dans son orchestrateur (éviter de dupliquer la logique des boutons Streamlit dans son code ; préférer appeler les mêmes fonctions / sous-processus).

---

## Perf et lenteurs

- SAM, depth et BFL sont **lourds** : prévoir loaders / file d’attente côté UI.
- L’éditeur de masques peut être lent avec beaucoup de calques : le dépôt principal filtre déjà les masques « plante seule » via `scene_sequential.json`.

---

## Mise à jour de ce dossier `final/`

Quand le code évolue dans le dépôt principal, régénérer la copie (PowerShell, depuis la racine Paysagea) :

```powershell
Remove-Item -Recurse -Force final -ErrorAction SilentlyContinue
# puis relancer le script de copie (robocopy) utilisé pour créer `final/`, ou copier à la main les dossiers listés ci-dessus.
```

Pour un suivi simple, **versionner tout le dépôt** et traiter `final/` comme miroir documenté plutôt que comme seule source de vérité.

---

*Généré pour accompagner l’intégration front / back — Paysagea.*
