# 🌿 Rapport d'Avancement Technique : Projet Paysagea

Ce rapport détaillé présente l'architecture logicielle, les algorithmes et les intégrations réalisées pour la plateforme **Paysagea**, une solution bout-en-bout de conception de jardins par intelligence artificielle.

---

## 1. Architecture du Système de Données

Le cœur du projet repose sur un **Manifeste de Projet** (`latest_project.json`) qui centralise toute l'intelligence collectée au fil du parcours utilisateur :

*   **Intention Utilisateur** : Description textuelle libre et tags de style (ex: moderne, naturel, méditerranéen).
*   **Contexte Visuel** : Image pré-traitée, carte de profondeur, et polygone de la "Zone Utilisateur" (coordonnées normalisées).
*   **Contexte Environnemental** : Géolocalisation précise couplée à un profil climatique complet.

---

## 2. Intelligence Géographique et Climatique (`backend_geo`)

Le backend Node.js ne se contente pas de relayer des données ; il calcule un véritable **Profil de Rusticité** :

*   **Normales Climatiques Glissantes** : Extraction de 12 mois de données historiques via l'archive OpenMeteo (précipitations cumulées, jours de gel, températures records).
*   **Calcul de la Zone USDA** : Algorithme de classification automatique déterminant la zone de rusticité (ex: Zone 8 pour un minima à -10°C).
*   **Seuils de Plantation** : Conversion des données brutes en catégories botaniques :
    *   *Humidité* : Sec (<50%), Moyen (50-70%), Humide (>70%).
    *   *Ensoleillement* : Ombre (<4h/j), Mi-ombre (4-6h/j), Soleil (>6h/j).

---

## 3. Pipeline de Vision par Ordinateur (`zone/final`)

L'analyse de l'image est orchestrée par un pipeline multi-modèles (`run_full_phase1_b...py`) :

*   **Segmentation (SAM)** : Isolation sémantique des éléments structurels (murs, allées) et végétaux existants.
*   **Estimation de Profondeur (Depth Anything)** : Génération d'un fichier `.npy` (matrix) et d'une visualisation `.png` pour comprendre la perspective et l'échelle.
*   **Fusion Topologique** : Le script `fuse_sam_depth.py` réunit les segments et leurs profondeurs respectives pour créer une scène 3D simplifiée exploitable par l'IA générative.
*   **Outil de Sélection Interactive** : Développement d'une application de détourage permettant de définir la zone de travail. Correction algorithmique du scaling pour assurer la correspondance parfaite entre le tracé utilisateur et les masques IA.

---

## 4. Moteur de Recommandation RAG (`Backend_RAG`)

Le système `rag_cli.py` assure la pertinence botanique des propositions :

*   **Filtrage par Rusticité** : Comparaison stricte entre la température minimale record de la localisation et la valeur de rusticité de la plante dans la base de données.
*   **Scoring de Style** : Algorithme de pondération par mots-clés. Une plante "Acer Palmatum" recevra un bonus de score si le style choisi est "Japonais".
*   **Format d'Interopérabilité** : Génération d'un fichier `rag_output.json` contenant non seulement le nom des plantes, mais aussi leurs caractéristiques physiques (hauteur, largeur, couleur) pour guider le modèle de génération d'image.

---

## 5. Frontend & State Management (`paysagea_site_react`)

L'interface est construite pour masquer la complexité technique sous une expérience "Premium" :

*   **Store Zustand** : Gestion centralisée de l'image (originale vs pré-traitée), des filtres et du manifeste de projet.
*   **Address Search** : Intégration fluide de Google Places avec cache local pour optimiser les performances.
*   **Animations Avancées** : Mise en place de transitions Fluides, de scanners de traitement et d'overlays interactifs pour la sélection de zone.
*   **Composants Réutilisables** : Bibliothèque personnalisée de boutons "Premium", panels vitrés (Glassmorphism) et cartes botaniques.

---

## 📊 État d'Avancement et Roadmap

### ✅ Finalisé
*   Proxy Geo/Climat complet avec calcul de zone USDA.
*   Pipeline SAM + Depth orchestré sur image pré-traitée.
*   Interface de sélection de zone interactive et persistante.
*   Système de scoring RAG basé sur le style et la rusticité.

### 🚧 En cours / À venir
*   **Génération d'Image Finale** : Intégration de l'API BFL (Flux Fill Pro) utilisant les masques de zone et la carte de profondeur pour un rendu photo-réaliste.
*   **Studio d'Édition** : Permettre à l'utilisateur de modifier les plantes suggérées après la génération (re-génération ciblée via masques individuels).
*   **Estimation de Coût** : Liaison entre les plantes recommandées et un catalogue de prix pour générer un devis estimatif automatique.

---
*Ce rapport technique témoigne de la solidité de l'infrastructure mise en place, alliant données environnementales réelles et puissance de l'IA générative.*
