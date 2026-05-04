# 🌿 Paysagea 

> Plateforme de conception de jardin assistée par IA : analyse de photos de jardin, segmentation (SAM), estimation de profondeur (Depth Anything), RAG plantes, et génération d'image (BFL FLUX Fill Pro).

---

## 📁 Structure du projet

```
paysagea_site_react/          ← Racine du monorepo (repo GitHub)
│
├── src/                      ← Frontend React (Vite + TailwindCSS)
├── public/                   ← Assets statiques
├── index.html
├── package.json              ← Config frontend
├── vite.config.js
│
├── backend/                  ← Serveur Node.js/Express
│   ├── server.js             ← API principale (géocodage, météo, IA bridge)
│   ├── package.json
│   ├── .env                  ← ⚠️ Ne jamais versionner
│   └── .env.example          ← Template des variables d'env
│
├── rag/                      ← Module RAG (recommandation de plantes)
│   ├── src/                  ← Moteur RAG Python
│   ├── data/
│   │   └── plantes_data.json ← Base de données ~2 500 plantes
│   ├── rag_cli.py            ← CLI appelé par le backend Node
│   └── requirements.txt
│
├── pipeline/                 ← Pipeline IA (SAM + Depth + Génération)
│   ├── preprocess_cli.py     ← Prétraitement image (appelé par backend)
│   ├── sam_depth_cli.py      ← SAM + Depth Anything (appelé par backend)
│   ├── preprocess/           ← Module de prétraitement
│   ├── sam/                  ← SAM (Segment Anything Model) — Meta
│   │   └── segment_anything/ ← Code source SAM
│   ├── depth/                ← Depth Anything
│   │   └── depth_anything/   ← Code source Depth Anything
│   ├── generation/           ← Génération jardin (BFL FLUX + masques)
│   │   ├── generate_garden_cli.py
│   │   ├── image_generation/ ← Générateur BFL
│   │   ├── rag/              ← Intégration RAG côté pipeline
│   │   └── segmentation/     ← Gestion des masques
│   ├── zone_selection/       ← Outil Python de sélection de zone (Tkinter)
│   ├── work/                 ← ⚠️ Fichiers runtime (gitignored)
│   └── shared/               ← ⚠️ Fichiers partagés runtime (gitignored)
│
└── docs/                     ← Documentation
    ├── RAPPORT_PROJET.md
    ├── productdesign.md
    └── ...
```

---

## 🚀 Installation & Lancement

### 1. Frontend (React)
```bash
npm install
npm run dev        # http://localhost:5173
```

### 2. Backend (Node.js)
```bash
cd backend
cp .env.example .env   # Remplir les clés API
npm install
node server.js         # http://localhost:3001
```

### 3. RAG (Python)
```bash
cd rag
python -m venv venv
pip install -r requirements.txt
python rag_cli.py --help
```

### 4. Pipeline IA (SAM + Depth Anything)
```bash
cd pipeline
python -m venv venv
pip install -r requirements_sam.txt
pip install -r requirements_depth.txt

# Télécharger le modèle SAM (357 MB)
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# → Placer dans pipeline/sam/

python preprocess_cli.py --help
python sam_depth_cli.py --help
```

---

## 🔑 Variables d'environnement requises

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Clé Google Places API (géocodage + autocomplete) |
| `BFL_API_KEY` | Clé Black Forest Labs (génération image FLUX Fill Pro) |
| `MOCK_BFL` | `true` pour tester sans appeler BFL (mode mock) |

---

## 📦 Données volumineuses (non versionnées)

Les fichiers suivants sont exclus du repo (`.gitignore`) mais nécessaires au fonctionnement :

| Fichier/Dossier | Taille | Où le trouver |
|-----------------|--------|---------------|
| `rag/data/all_photos/` | ~34 GB | Disque externe / Drive partagé |
| `pipeline/sam/sam_vit_b_01ec64.pth` | ~357 MB | [Meta SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |

---

## 🏗️ Architecture

```
Frontend React
    ↓ (upload image + préférences)
Backend Node.js (port 3001)
    ↓                    ↓                    ↓
preprocess_cli.py   rag_cli.py      generate_garden_cli.py
    ↓                    ↓                    ↓
sam_depth_cli.py    RAG (ChromaDB)   BFL FLUX Fill Pro
    ↓
pipeline_result.json → génération image finale
```
