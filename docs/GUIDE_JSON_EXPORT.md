# 📊 Guide d'export JSON pour SAM

## 🎯 Ce que fait ce script

Le script `sam_export_json.py` vous permet de :
1. Segmenter des objets avec SAM
2. **Exporter les résultats au format JSON structuré**
3. Choisir entre format RLE (recommandé) ou binaire

## 📦 Installation supplémentaire

```bash
# Installer pycocotools pour le format RLE
pip install pycocotools
```

## 🚀 Utilisation

### Format RLE (recommandé - fichiers plus légers)
```bash
python sam_export_json.py photo.jpg rle
```

### Format binaire (pour debug ou traitement direct)
```bash
python sam_export_json.py photo.jpg binary
```

### Par défaut (RLE)
```bash
python sam_export_json.py photo.jpg
```

## 📄 Format de sortie JSON

```json
{
  "sam_output": {
    "image_size": [1920, 1080],
    "num_segments": 15,
    "format": "rle",
    "segments": [
      {
        "segment_id": 0,
        "mask_rle": {
          "size": [1080, 1920],
          "counts": "aYn05la..."
        },
        "area_ratio": 0.1842,
        "bbox": [0.4215, 0.6104, 0.3145, 0.2187],
        "centroid": [0.5234, 0.7412]
      },
      {
        "segment_id": 1,
        "mask_rle": {...},
        "area_ratio": 0.0956,
        "bbox": [0.1234, 0.3456, 0.2345, 0.1234],
        "centroid": [0.2456, 0.4123]
      }
    ]
  }
}
```

## 📊 Explication des champs

### `image_size`
- Format : `[width, height]`
- Dimensions de l'image en pixels

### `num_segments`
- Nombre total de segments détectés

### `format`
- `"rle"` : Run-Length Encoding (COCO format)
- `"binary"` : Masque binaire 2D

### Pour chaque segment :

#### `segment_id`
- ID unique du segment (trié par taille décroissante)

#### `mask_rle` (si format = "rle")
- **Format COCO RLE** - Très compact
- `size` : `[height, width]`
- `counts` : String encodé
- Peut être décodé avec `pycocotools.mask`

#### `mask_binary` (si format = "binary")
- Matrice 2D de 0 et 1
- 1 = pixel fait partie du segment
- 0 = pixel n'en fait pas partie
- ⚠️ Fichiers beaucoup plus lourds

#### `area_ratio`
- Proportion de l'image occupée par le segment
- Valeur entre 0 et 1
- Exemple : 0.1842 = 18.42% de l'image

#### `bbox`
- Bounding box normalisée : `[x, y, w, h]`
- Toutes les valeurs entre 0 et 1
- `x, y` : coin supérieur gauche
- `w, h` : largeur et hauteur

#### `centroid`
- Centre de gravité du segment : `[x, y]`
- Valeurs normalisées entre 0 et 1

## 🔄 Décoder un masque RLE

```python
from pycocotools import mask as mask_utils
import json
import numpy as np

# Charger le JSON
with open('photo_sam_output.json', 'r') as f:
    data = json.load(f)

# Récupérer un segment
segment = data['sam_output']['segments'][0]
rle = segment['mask_rle']

# Décoder le RLE en masque binaire
binary_mask = mask_utils.decode(rle)

# binary_mask est maintenant une matrice numpy de 0 et 1
print(binary_mask.shape)  # (height, width)
```

## 💡 Exemples d'utilisation du JSON

### 1. Compter les objets
```python
import json

with open('photo_sam_output.json', 'r') as f:
    data = json.load(f)

num_objects = data['sam_output']['num_segments']
print(f"{num_objects} objets détectés")
```

### 2. Trouver les plus gros objets
```python
segments = data['sam_output']['segments']

# Les 3 plus gros (déjà triés par taille)
top3 = segments[:3]

for seg in top3:
    print(f"Segment {seg['segment_id']}: {seg['area_ratio']*100:.1f}%")
```

### 3. Filtrer par position
```python
# Objets dans la moitié gauche de l'image
left_objects = [
    seg for seg in segments 
    if seg['centroid'][0] < 0.5
]

print(f"{len(left_objects)} objets à gauche")
```

### 4. Extraire un objet spécifique
```python
from pycocotools import mask as mask_utils
import cv2
import numpy as np

# Charger l'image originale
image = cv2.imread('photo.jpg')

# Récupérer un segment
segment = segments[0]  # Le plus gros objet
mask = mask_utils.decode(segment['mask_rle'])

# Extraire l'objet (fond noir)
object_extracted = image * mask[:, :, np.newaxis]

# Sauvegarder
cv2.imwrite('object_extracted.png', object_extracted)
```

### 5. Dessiner les bounding boxes
```python
import cv2

image = cv2.imread('photo.jpg')
height, width = image.shape[:2]

for seg in segments:
    # Convertir bbox normalisée en pixels
    x, y, w, h = seg['bbox']
    x_px = int(x * width)
    y_px = int(y * height)
    w_px = int(w * width)
    h_px = int(h * height)
    
    # Dessiner
    cv2.rectangle(image, (x_px, y_px), (x_px + w_px, y_px + h_px), 
                  (0, 255, 0), 2)
    
    # Ajouter l'ID
    cv2.putText(image, f"#{seg['segment_id']}", 
                (x_px, y_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 2)

cv2.imwrite('with_bboxes.jpg', image)
```

## 📈 Comparaison des formats

### Format RLE (recommandé)
✅ Fichiers 10-100x plus petits
✅ Standard COCO - compatible avec beaucoup d'outils
✅ Rapide à encoder/décoder
❌ Nécessite pycocotools pour décoder

**Exemple** : Image 1920x1080 avec 20 objets → ~50 KB

### Format Binaire
✅ Facile à lire et manipuler
✅ Pas de dépendance supplémentaire
❌ Fichiers très lourds
❌ JSON peut devenir énorme

**Exemple** : Image 1920x1080 avec 20 objets → ~40 MB

## 🎯 Cas d'usage

### 1. Segmentation automatique → JSON
```bash
python sam_export_json.py scene.jpg rle
# Génère : scene_sam_output.json
#          scene_visualization.png
```

### 2. Segmentation interactive → JSON
```bash
python sam_export_json.py object.jpg rle
# Choisir mode 2
# Cliquer sur l'objet
# Génère : object_interactive_sam_output.json
#          object_interactive_visualization.png
```

### 3. Pipeline complet
```python
# 1. Segmenter avec SAM
# 2. Charger le JSON
# 3. Traiter les segments
# 4. Exporter vers votre format

import json
from pycocotools import mask as mask_utils

with open('results.json', 'r') as f:
    data = json.load(f)

# Traiter chaque segment
for seg in data['sam_output']['segments']:
    mask = mask_utils.decode(seg['mask_rle'])
    # Votre traitement ici
    process_segment(mask, seg['bbox'], seg['centroid'])
```

## 🔧 Personnalisation

Vous pouvez modifier le script pour ajouter d'autres informations :
- Couleur moyenne du segment
- Forme (compacité, circularité)
- Textures
- Features custom

## 📚 Ressources

- [COCO RLE Format](https://github.com/cocodataset/cocoapi)
- [pycocotools Documentation](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)
- [SAM GitHub](https://github.com/facebookresearch/segment-anything)

## ❓ FAQ

**Q: Pourquoi RLE est recommandé ?**
R: Fichiers beaucoup plus légers (important pour stocker/transférer) et format standard utilisé dans COCO dataset.

**Q: Comment convertir RLE vers binaire ?**
R: `mask = mask_utils.decode(rle)` avec pycocotools.

**Q: Les coordonnées sont-elles normalisées ?**
R: Oui, bbox et centroid sont entre 0 et 1 pour être indépendants de la résolution.

**Q: Comment filtrer les petits objets ?**
R: Filtrez par `area_ratio` dans le JSON : `seg['area_ratio'] > 0.01` (> 1% de l'image)

**Q: Puis-je combiner plusieurs segments ?**
R: Oui, décodez les RLE et faites un OR logique des masques.
