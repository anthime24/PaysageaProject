# 🔄 Comparatif : sam_export_json.py (v1 vs v2)

## 📦 Deux versions disponibles

### Version 1 (Standalone) : `sam_export_json.py`
Script autonome qui fonctionne directement avec des images

### Version 2 (Pipeline) : `sam_export_json_v2.py`
Script intégré au pipeline preprocessing, lit le `preprocess.json`

---

## 🎯 Quand utiliser quelle version ?

| Situation | Version à utiliser |
|-----------|-------------------|
| **Test rapide d'une image** | v1 (standalone) |
| **Pipeline de production** | v2 (pipeline) |
| **Pas de preprocessing nécessaire** | v1 (standalone) |
| **Cohérence avec Depth/autres modules** | v2 (pipeline) |
| **Traçabilité complète requise** | v2 (pipeline) |
| **Prototypage/exploration** | v1 (standalone) |

---

## 📝 Différences de commande

### Version 1 (Standalone)
```bash
# Direct : donne l'image
python sam_export_json.py photo.jpg rle

# Outputs :
#   - photo_sam_output.json
#   - photo_visualization.png
```

### Version 2 (Pipeline)
```bash
# Étape 1 : Preprocess d'abord
python preprocess_image.py photo.jpg

# Étape 2 : SAM avec le JSON preprocess
python sam_export_json_v2.py photo-535x356_preprocessed.json rle

# Outputs :
#   - photo-535x356_sam_output.json
#   - photo-535x356_sam_visualization.png
```

---

## 🔍 Différences de code

### 1. Fonction principale

#### **Version 1 (Standalone)**
```python
def segment_automatic_with_export(image_path, checkpoint_path, format="rle", device="cuda"):
    # Charge l'image directement
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Continue normalement...
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    masks = mask_generator.generate(image)
    
    # Export simple
    output = {
        "sam_output": {
            "image_size": [w, h],
            "segments": [...]
        }
    }
```

#### **Version 2 (Pipeline)**
```python
def segment_automatic_with_export(preprocess_json_path, checkpoint_path, format="rle", device="cuda"):
    # 1. Charge le preprocess.json (SOURCE DE VÉRITÉ)
    preprocess_data = load_preprocess_json(preprocess_json_path)
    
    # 2. Extrait le chemin de l'image preprocessed
    image_path = preprocess_data['preprocessed_filename']
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. VÉRIFICATION CRITIQUE
    verify_image_matches_preprocess(image, preprocess_data)
    
    # 4. Continue normalement...
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    masks = mask_generator.generate(image)
    
    # 5. Export avec métadonnées
    output = {
        "version": "sam_output_v1",
        "image_id": preprocess_data['image_id'],
        "preprocess": preprocess_data['preprocess'],
        "preprocessed_filename": preprocess_data['preprocessed_filename'],
        "sam_output": {
            "image_size": [w, h],
            "segments": [...]
        }
    }
```

---

### 2. Structure du JSON de sortie

#### **Version 1 (Standalone)**
```json
{
  "sam_output": {
    "image_size": [1920, 1080],
    "num_segments": 15,
    "format": "rle",
    "segments": [
      {
        "segment_id": 0,
        "mask_rle": {...},
        "area_ratio": 0.1842,
        "bbox": [0.42, 0.61, 0.31, 0.22],
        "centroid": [0.52, 0.74]
      }
    ]
  }
}
```

#### **Version 2 (Pipeline)**
```json
{
  "version": "sam_output_v1",
  "image_id": "sha256:a319a26478988116",
  "preprocess": {
    "original_filename": "IMG_5177.HEIC",
    "resized_size": [535, 356],
    "orientation": "landscape",
    "scale": 0.25,
    "exif_orientation": 1
  },
  "preprocessed_filename": "IMG_5177-535x356.jpg",
  "sam_output": {
    "image_size": [535, 356],
    "num_segments": 15,
    "format": "rle",
    "segments": [
      {
        "segment_id": 0,
        "mask_rle": {...},
        "area_ratio": 0.1842,
        "bbox": [0.42, 0.61, 0.31, 0.22],
        "centroid": [0.52, 0.74]
      }
    ]
  }
}
```

**Différences clés :**
- ✅ `image_id` : traçabilité unique
- ✅ `preprocess` : toutes les métadonnées du preprocessing
- ✅ `preprocessed_filename` : quelle image a été utilisée
- ✅ `version` : versioning du format

---

### 3. Sécurités

#### **Version 1 (Standalone)**
```python
# Vérifie juste que l'image existe
if not os.path.exists(image_path):
    print(f"❌ Erreur : L'image '{image_path}' n'existe pas")
    return None

# Vérifie qu'elle se charge
image = cv2.imread(image_path)
if image is None:
    print(f"❌ Erreur : Impossible de lire '{image_path}'")
    return None
```

#### **Version 2 (Pipeline)**
```python
# Vérifie que le preprocess.json existe
if not os.path.exists(preprocess_json_path):
    raise FileNotFoundError(
        f"❌ Fichier preprocess non trouvé : {preprocess_json_path}\n"
        f"   Vous devez d'abord lancer preprocess_image.py"
    )

# Charge le preprocess
preprocess_data = load_preprocess_json(preprocess_json_path)

# Vérifie que l'image preprocessed existe
image_path = preprocess_data['preprocessed_filename']
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image preprocessed non trouvée : {image_path}")

# Charge l'image
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"❌ Erreur lors du chargement de l'image : {image_path}")

# VÉRIFICATION CRITIQUE : dimensions correspondent ?
h, w = image.shape[:2]
expected_w, expected_h = preprocess_data['preprocess']['resized_size']
if (w, h) != (expected_w, expected_h):
    raise ValueError(
        f"❌ ERREUR CRITIQUE : Dimensions de l'image ne correspondent pas !\n"
        f"   Image chargée : {w}x{h}\n"
        f"   Attendu (preprocess) : {expected_w}x{expected_h}"
    )
```

**Version 2 a 4 niveaux de vérification au lieu de 2 !**

---

## 📊 Tableau comparatif complet

| Caractéristique | Version 1 (Standalone) | Version 2 (Pipeline) |
|----------------|------------------------|----------------------|
| **Input** | Fichier image | Fichier preprocess.json |
| **Dépendance** | Aucune | Nécessite preprocess_image.py |
| **Dimensions** | Telles quelles dans l'image | Depuis preprocess.json |
| **Vérifications** | 2 (existe, lisible) | 4 (preprocess existe, image existe, lisible, dimensions) |
| **Traçabilité** | Limitée | Complète (image_id) |
| **Output JSON** | Simple | Enrichi avec métadonnées |
| **Cohérence pipeline** | Non garantie | Garantie |
| **Use case** | Test/prototype | Production |
| **Rapidité** | Plus rapide (1 commande) | 2 commandes (preprocess + SAM) |
| **Sécurité** | Basique | Renforcée |

---

## 🎯 Exemples d'utilisation

### Scénario 1 : Test rapide d'une nouvelle image

```bash
# Version 1 : Direct et rapide
python sam_export_json.py test.jpg rle
# ✅ Idéal pour tester rapidement
```

### Scénario 2 : Pipeline de production avec Depth

```bash
# Version 2 : Cohérence garantie
python preprocess_image.py photo.jpg
python sam_export_json_v2.py photo-535x356_preprocessed.json rle
python depth_estimation.py photo-535x356_preprocessed.json
python fusion.py photo-535x356_sam_output.json photo-535x356_depth_output.json

# ✅ Toutes les étapes utilisent le même preprocess
# ✅ image_id identique partout
# ✅ Dimensions garanties cohérentes
```

### Scénario 3 : Batch processing

```bash
# Version 1 : Simple mais sans cohérence
for img in *.jpg; do
    python sam_export_json.py "$img" rle
done
# ⚠️ Pas de garantie sur les dimensions

# Version 2 : Cohérent et tracé
for img in *.jpg; do
    python preprocess_image.py "$img"
    preprocess_json="${img%.jpg}-*_preprocessed.json"
    python sam_export_json_v2.py $preprocess_json rle
done
# ✅ Toutes les images traitées de manière cohérente
```

---

## 🔧 Migration de v1 vers v2

Si vous avez des scripts qui utilisent la version 1 :

### Étape 1 : Ajouter le preprocessing
```bash
# AVANT
python sam_export_json.py image.jpg rle

# APRÈS
python preprocess_image.py image.jpg
python sam_export_json_v2.py image-*_preprocessed.json rle
```

### Étape 2 : Modifier votre code Python

```python
# AVANT
from sam_export_json import segment_automatic_with_export

result = segment_automatic_with_export(
    image_path="photo.jpg",
    checkpoint_path="sam_vit_b_01ec64.pth",
    format="rle"
)

# APRÈS
from sam_export_json_v2 import segment_automatic_with_export

result = segment_automatic_with_export(
    preprocess_json_path="photo-535x356_preprocessed.json",
    checkpoint_path="sam_vit_b_01ec64.pth",
    format="rle"
)
```

### Étape 3 : Adapter le parsing des résultats

```python
# AVANT
num_segments = result['sam_output']['num_segments']

# APRÈS
num_segments = result['sam_output']['num_segments']
image_id = result['image_id']  # Nouveau : traçabilité
original_filename = result['preprocess']['original_filename']  # Nouveau
```

---

## 💡 Recommandations

### Pour débuter / prototyper
→ **Utilisez la Version 1** : plus simple, moins de friction

### Pour la production
→ **Utilisez la Version 2** : cohérence et traçabilité garanties

### Pour un pipeline complet (SAM + Depth + Fusion)
→ **Obligatoire : Version 2** : seule façon de garantir la cohérence

---

## 🚨 Erreurs courantes

### Erreur 1 : Utiliser v2 sans preprocess
```bash
python sam_export_json_v2.py photo.jpg rle
# ❌ FileNotFoundError: Fichier preprocess non trouvé
```

**Solution :**
```bash
python preprocess_image.py photo.jpg
python sam_export_json_v2.py photo-535x356_preprocessed.json rle
```

### Erreur 2 : Dimensions incohérentes
```bash
# Si vous modifiez manuellement l'image après le preprocess
# ❌ ValueError: ERREUR CRITIQUE : Dimensions de l'image ne correspondent pas !
```

**Solution :** Ne jamais modifier l'image après le preprocess

### Erreur 3 : Mauvais fichier preprocess
```bash
python sam_export_json_v2.py wrong_preprocessed.json rle
# ❌ Image preprocessed non trouvée
```

**Solution :** Vérifier le nom du fichier dans le preprocess.json

---

## 📁 Fichiers à conserver

Si vous travaillez en **standalone** (v1) :
- ✅ `sam_export_json.py`
- ✅ Images originales
- ✅ Outputs SAM

Si vous travaillez en **pipeline** (v2) :
- ✅ `sam_export_json_v2.py`
- ✅ `preprocess_image.py`
- ✅ Images originales
- ✅ Images preprocessed (`*-535x356.jpg`)
- ✅ Fichiers preprocess (`*_preprocessed.json`)
- ✅ Outputs SAM (`*_sam_output.json`)

---

## ✅ Checklist de choix

**Utilisez Version 1 si :**
- [ ] Vous testez rapidement une image
- [ ] Vous n'avez pas besoin de cohérence avec d'autres modules
- [ ] Vous faites du prototypage
- [ ] La traçabilité n'est pas critique

**Utilisez Version 2 si :**
- [ ] Vous construisez un pipeline (SAM + autres)
- [ ] La cohérence des dimensions est critique
- [ ] Vous avez besoin de traçabilité (image_id)
- [ ] C'est pour de la production
- [ ] Vous allez fusionner avec Depth ou autre

---

**En résumé : v1 pour la rapidité, v2 pour la robustesse ! 🎯**
