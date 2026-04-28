# Garden IA 3 - Commandes rapides (Windows / PowerShell)

Ce README contient les commandes prêtes a copier-coller pour relancer le site Streamlit en local et tester le flux complet avec la pipeline Phase1-B (SAM + Depth + Zone).

## 1) Lancer le site Streamlit (commande rapide)

```powershell
cd "c:\Users\sabri\OneDrive\Desktop\PGE2\clinique\s2\Paysagea\garden_ia_3"
streamlit run ui\app.py
```

## 2) Setup complet (si besoin)

```powershell
cd "c:\Users\sabri\OneDrive\Desktop\PGE2\clinique\s2\Paysagea"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r "garden_ia_3\requirements.txt"
pip install -r "garden_ia_3\requirements_rag.txt"
pip install opencv-python numpy torch torchvision matplotlib pycocotools Pillow streamlit
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## 3) Activer la clé API BFL (optionnel, pour génération réelle)

Dans le même terminal que Streamlit :

```powershell
cd "c:\Users\sabri\OneDrive\Desktop\PGE2\clinique\s2\Paysagea"
.\.venv\Scripts\Activate.ps1
$env:BFL_API_KEY = "TA_CLE_API_ICI"
Remove-Item Env:MOCK_BFL -ErrorAction SilentlyContinue
```

Pour vérifier :

```powershell
echo $env:BFL_API_KEY
```

## 4) Relancer Streamlit avec le venv explicite

```powershell
cd "c:\Users\sabri\OneDrive\Desktop\PGE2\clinique\s2\Paysagea\garden_ia_3"
& "..\.venv\Scripts\python.exe" -m streamlit run ui\app.py
```

## 5) Workflow test dans l'UI

1. Uploader l'image du jardin (png/jpg).
2. Cliquer le bouton:
   - `Lancer Phase1-B (SAM+Depth+Zone) => final_scene_input.json`
3. Garder coche:
   - `Utiliser la zone plantable (user_zone_mask.png)`
4. Mettre `Nb max de plantes` a 4 ou 5.
5. Cliquer:
   - `GENERER MON JARDIN`

## 6) Lancer la pipeline Phase1-B en ligne de commande (hors UI)

```powershell
cd "c:\Users\sabri\OneDrive\Desktop\PGE2\clinique\s2\Paysagea"
.\.venv\Scripts\Activate.ps1
python ".\run_full_phase1_b_sam_depth_zone.py" --image ".\Inputs\menphis_depay.jpeg" --sam-checkpoint ".\SAM_for_paysagea\sam_vit_b_01ec64.pth"
```

## 7) Phase1-B sans ouvrir l'UI OpenCV de zone-selection

```powershell
cd "c:\Users\sabri\OneDrive\Desktop\PGE2\clinique\s2\Paysagea"
.\.venv\Scripts\Activate.ps1
python ".\run_full_phase1_b_sam_depth_zone.py" --image ".\Inputs\menphis_depay.jpeg" --sam-checkpoint ".\SAM_for_paysagea\sam_vit_b_01ec64.pth" --skip-zone-selection --user-zone-json ".\zone-selection\outputs\user_zone.json"
```

## 8) Fichiers de sortie a verifier

- `zone-selection\outputs\user_zone_overlay.png`
- `zone-selection\outputs\user_zone_mask.png`
- `main.json`
- `integration\final_scene_input.json`
- `garden_ia_3\outputs\final_garden.png`
- `garden_ia_3\outputs\scene_sequential.json`

## 9) Erreurs frequentes

- `CommandNotFound ..\ .venv\Scripts\python.exe`
  - Corriger en `..\.venv\Scripts\python.exe` (sans espace).
- `UnicodeEncodeError` sur Windows
  - Deja corrige dans l'orchestrateur (prints sans emoji).
- `cannot write mode RGBA as JPEG`
  - Deja corrige (upload conserve l'extension + conversion RGB cote pipeline).


