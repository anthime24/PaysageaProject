# Document de Conception Produit : Garden AI POC

## 1. Vision du Produit
**Garden AI** est une application web innovante permettant aux propriétaires de jardins de visualiser instantanément le potentiel de transformation de leur espace extérieur grâce à l'intelligence artificielle. Ce POC vise à démontrer une fluidité d'utilisation et une compréhension intelligente du terrain en moins de 60 secondes.

## 2. Objectifs Stratégiques
- **Démonstration de faisabilité** : Prouver que l'IA peut segmenter et transformer un jardin de manière réaliste.
- **Expérience Utilisateur (UX) Premium** : Offrir un parcours fluide et impressionnant pour convaincre les décideurs (CEO/CTO).
- **Potentiel de Scalabilité** : Établir les bases techniques d'un produit complet (éditeur interactif, catalogue de plantes, estimation budgétaire).

## 3. Parcours Utilisateur (Linear Flow)
L'application suit un flux strictement linéaire pour garantir une démonstration efficace :

### Étape 1 — Upload (`/upload`)
- **Action** : L'utilisateur télécharge une photo (JPG/PNG).
- **Filtres** :
    - Styles : Naturel, Méditerranéen, Contemporain.
    - Équipements : Piscine, Pergola, Gazon, Entretien faible.
- **Conversion** : Bouton "Analyser mon jardin".

### Étape 2 — Analyse (`/processing`)
- **Visuel** : Indicateur d'analyse IA "intelligent" (overlay de segmentation, détection de profondeur).
- **Backend Logique** : Simulation de la segmentation des zones et de la préparation des métadonnées.

### Étape 3 — Résultat (`/result/:projectId`)
- **Transformation** : Affichage de l'image transformée générée par l'IA.
- **Résumé** : Liste des modifications apportées et concept général.
- **Engagement** : Options pour modifier les filtres ou régénérer.

## 4. Architecture Technique
- **Frontend** : React 18, Vite, Tailwind CSS (Design System moderne).
- **Navigation** : React Router (Linear route transitions).
- **État** : Zustand (Gestion légère de l'image, des filtres et du résultat).
- **Backend** : Architecture REST (POST `/analyze`, POST `/generate`).
- **Performance** : Temps de parcours total ciblé < 60 secondes.

## 5. UI/UX Principles
- **Minimalisme** : Focus sur l'image et l'action.
- **Professionnalisme** : Palette de couleurs sobres, typographie moderne (Inter/Roboto), micro-animations.
- **Crédibilité** : Les indicateurs d'analyse doivent suggérer une compréhension réelle du terrain.

## 6. Roadmap Évolutive (Vision Long Terme)
- Éditeur de zones interactif.
- Intégration de catalogues botaniques réels.
- Exportation de devis et plans techniques.
- Analyse par drone et reconstruction 3D.
