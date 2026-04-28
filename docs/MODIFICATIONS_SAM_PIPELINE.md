# 🔄 Modifications SAM pour intégration Pipeline Preprocess

## 📋 Résumé des changements

Le script `sam_export_json.py` a été **modifié** pour s'intégrer dans un pipeline avec preprocessing. SAM ne décide plus des dimensions : il **obéit** au `preprocess.json`.

---

## ✅ Ce qui a été modifié

### 1. **Nouvelle signature de fonction**

#### ❌ AVANT :
```python
def segment_automatic_with_export(image_path, checkpoint_path, format="rle", device="cuda"):
    image = cv2.imread(image_path)  # ❌ SAM charge l'image "au hasard"
```

#### ✅ APRÈS :
```python
def segment_automatic_with_export(preprocess_json_path, checkpoint_path, format="rle", device="cuda"):
    # 1. Charger le preprocess.json (SOURCE DE VÉRITÉ)
    preprocess_data = load_preprocess_json(preprocess_json_path)
    
    # 2. Charger l'image preprocessed
    image_path = preprocess_data['preprocessed_filename']
    image = cv2.imread(image_path)
    
    # 3. VÉRIFIER que les dimensions correspondent
    verify_image_matches_preprocess(image, preprocess_data)
```

---

### 2. **Nouvelle fonction : `load_preprocess_json()`**

```python
def load_preprocess_json(preprocess_json_path):
    """
    Charge le fichier preprocess.json
    C'est la SOURCE DE VÉRITÉ pour les dimensions et métadonnées
    """
    if not os.path.exists(preprocess_json_path):
        raise FileNotFoundError(
            f"❌ Fichier preprocess non trouvé : {preprocess_json_path}\n"
            f"   Vous devez d'abord lancer preprocess_image.py"
        )
    
    with open(preprocess_json_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Preprocess chargé : {preprocess_json_path}")
    print(f"  Image ID : {data['image_id']}")
    print(f"  Taille : {data['preprocess']['resized_size']}")
    print(f"  Orientation : {data['preprocess']['orientation']}")
    
    return data
```

**Pourquoi c'est important :**
- Le preprocess.json est la **source unique de vérité**
- SAM ne "devine" plus les dimensions
- Garantit la cohérence avec les autres étapes du pipeline (Depth, etc.)

---

### 3. **Nouvelle fonction : `verify_image_matches_preprocess()`**

```python
def verify_image_matches_preprocess(image, preprocess_data):
    """
    VÉRIFICATION CRITIQUE : L'image chargée correspond-elle au preprocess ?
    Si ça plante ici, c'est qu'il y a une incohérence dangereuse
    """
    h, w = image.shape[:2]
    expected_w, expected_h = preprocess_data['preprocess']['resized_size']
    
    if (w, h) != (expected_w, expected_h):
        raise ValueError(
            f"❌ ERREUR CRITIQUE : Dimensions de l'image ne correspondent pas !\n"
            f"   Image chargée : {w}x{h}\n"
            f"   Attendu (preprocess) : {expected_w}x{expected_h}\n"
            f"   → Vérifiez que vous utilisez la bonne image preprocessed"
        )
    
    print(f"✓ Vérification dimensions : {w}x{h} ✓")
```

**Pourquoi c'est critique :**
- Détecte les incohérences **avant** de faire tourner SAM
- Évite des bugs silencieux où SAM traite la mauvaise image
- Si ça plante ici, c'est **volontaire** et **salvateur** 🛡️

---

### 4. **Output JSON modifié : inclut le preprocess**

#### ❌ AVANT :
```json
{
  "sam_output": {
    "image_size": [1920, 1080],
    "segments": [...]
  }
}
```

#### ✅ APRÈS :
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
    "segments": [...]
  }
}
```

**Avantages :**
- Le JSON SAM est **auto-suffisant** : il contient toutes les métadonnées
- On peut tracer d'où vient chaque résultat
- Compatible avec la fusion ultérieure (SAM + Depth)

---

### 5. **Nouvelle utilisation en ligne de commande**

#### ❌ AVANT :
```bash
python sam_export_json.py photo.jpg rle
```

#### ✅ APRÈS :
```bash
# 1. D'abord : preprocess
python preprocess_image.py IMG_5177.HEIC
# → Génère : IMG_5177-535x356.jpg
#            IMG_5177-535x356_preprocessed.json

# 2. Ensuite : SAM avec le JSON preprocess
python sam_export_json.py IMG_5177-535x356_preprocessed.json rle
# → Génère : IMG_5177-535x356_sam_output.json
#            IMG_5177-535x356_sam_visualization.png
```

---

## 🚫 Ce que SAM ne fait PLUS

SAM ne fait **plus jamais** :
- ❌ Redimensionner l'image
- ❌ Corriger l'orientation
- ❌ Calculer un scale
- ❌ Modifier les coordonnées
- ❌ Deviner quelle image utiliser

**Tout ça est fait par `preprocess_image.py`**

---

## ✅ Ce que SAM fait MAINTENANT

1. **Lit** le `preprocess.json`
2. **Charge** l'image preprocessed indiquée
3. **Vérifie** que les dimensions correspondent (sécurité)
4. **Segmente** l'image (son vrai job)
5. **Exporte** avec les métadonnées du preprocess copiées

---

## 🔍 Exemple de workflow complet

```bash
# Étape 1 : Preprocess
python preprocess_image.py IMG_5177.HEIC

Output:
✓ Image chargée : IMG_5177.HEIC
✓ Orientation corrigée : 1
✓ Image redimensionnée : 535x356
✓ Métadonnées sauvegardées : IMG_5177-535x356_preprocessed.json
✓ Image sauvegardée : IMG_5177-535x356.jpg

# Étape 2 : SAM
python sam_export_json.py IMG_5177-535x356_preprocessed.json rle

Output:
✓ Preprocess chargé : IMG_5177-535x356_preprocessed.json
  Image ID : sha256:a319a26478988116
  Taille : [535, 356]
  Orientation : landscape
✓ Image chargée : IMG_5177-535x356.jpg
✓ Vérification dimensions : 535x356 ✓  # ← CRITIQUE
✓ Modèle chargé sur cuda
Génération des masques...
✓ 15 objets détectés
✓ Résultats sauvegardés : IMG_5177-535x356_sam_output.json
  - Nombre de segments : 15
  - Format : rle
  - Image ID : sha256:a319a26478988116  # ← Même ID que preprocess ✓

# Étape 3 : Depth (futur)
python depth_estimation.py IMG_5177-535x356_preprocessed.json
# → depth utilise le MÊME preprocess.json
# → Garantie de cohérence avec SAM ✓
```

---

## 🛡️ Sécurités ajoutées

### 1. Vérification du fichier preprocess
```python
if not os.path.exists(preprocess_json_path):
    raise FileNotFoundError(
        f"❌ Fichier preprocess non trouvé : {preprocess_json_path}\n"
        f"   Vous devez d'abord lancer preprocess_image.py"
    )
```

### 2. Vérification de l'image preprocessed
```python
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image preprocessed non trouvée : {image_path}")
```

### 3. Vérification des dimensions (LA PLUS IMPORTANTE)
```python
if (w, h) != (expected_w, expected_h):
    raise ValueError(
        f"❌ ERREUR CRITIQUE : Dimensions de l'image ne correspondent pas !\n"
        f"   Image chargée : {w}x{h}\n"
        f"   Attendu (preprocess) : {expected_w}x{expected_h}"
    )
```

**Si une de ces vérifications échoue, le script plante VOLONTAIREMENT.**
C'est une bonne chose : mieux vaut planter tôt que produire des résultats incohérents.

---

## 📊 Comparaison avant/après

| Aspect | AVANT | APRÈS |
|--------|-------|-------|
| **Input** | Chemin image | Chemin preprocess.json |
| **Source dimensions** | SAM devine | preprocess.json |
| **Vérification** | Aucune | Dimensions vérifiées |
| **Output JSON** | Basique | Avec métadonnées preprocess |
| **Traçabilité** | Limitée | Complète (image_id) |
| **Cohérence pipeline** | ❌ Non garantie | ✅ Garantie |

---

## 🎯 Principe de design

> **SAM ne "décide plus" de la taille, il obéit.**

Le `preprocess.json` est le **contrat d'interface** entre les différentes étapes du pipeline :
- Preprocess → SAM : via le JSON
- Preprocess → Depth : via le JSON
- SAM + Depth → Fusion : même image_id garantit la cohérence

---

## 🔧 Modifications minimales nécessaires

Si vous avez d'autres scripts qui utilisent SAM, voici les changements minimaux :

```python
# AVANT
image = cv2.imread(image_path)
sam_result = segment_automatic_with_export(image_path, checkpoint, "rle")

# APRÈS
preprocess_data = load_preprocess_json(preprocess_json_path)
image_path = preprocess_data['preprocessed_filename']
image = cv2.imread(image_path)
verify_image_matches_preprocess(image, preprocess_data)
sam_result = segment_automatic_with_export(preprocess_json_path, checkpoint, "rle")
```

---

## ✅ Checklist de migration

Pour adapter un script existant :

- [ ] Remplacer `image_path` par `preprocess_json_path` en argument
- [ ] Ajouter `load_preprocess_json()` au début
- [ ] Extraire l'image path depuis le preprocess : `preprocess_data['preprocessed_filename']`
- [ ] Ajouter `verify_image_matches_preprocess()` après chargement
- [ ] Modifier l'output JSON pour inclure `preprocess` et `image_id`
- [ ] Tester avec une vraie image preprocessed

---

## 📝 Notes importantes

1. **Le script détectera automatiquement** si vous oubliez de préprocesser
2. **Les anciennes commandes ne marcheront plus** (c'est volontaire)
3. **Tous les outputs incluent maintenant l'image_id** pour la traçabilité
4. **La vérification des dimensions est NON-NÉGOCIABLE**

---

## 🚀 Prochaines étapes

Maintenant que SAM est intégré au pipeline :
1. ✅ SAM lit le preprocess.json
2. ✅ SAM vérifie les dimensions
3. ✅ SAM inclut les métadonnées dans son output
4. ⏳ Depth doit faire pareil (même logique)
5. ⏳ Fusion utilisera les image_id pour matcher SAM + Depth

**Le pipeline est maintenant cohérent et sûr ! 🎉**
