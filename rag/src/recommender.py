"""
Ce fichier gère le filtrage et le scoring des plantes
"""
import pandas as pd
from typing import List, Dict, Any

class PlantRecommender:
    def __init__(self, df: pd.DataFrame):
        """
        Initialise le système de recommandation
        
        Args:
            df: DataFrame avec les données des plantes
        """
        self.df = df
        self.user_preferences = {}
        
    def set_preferences(self, preferences: Dict[str, Any]):
        """Enregistre les préférences utilisateur"""
        self.user_preferences = preferences
        print(f"✅ Préférences enregistrées: {preferences}")
        
    def filter_by_criteria(self) -> pd.DataFrame:
        """Filtre les plantes selon les critères de base"""
        filtered_df = self.df.copy()
        
        # Filtre par exposition (si spécifié et pas indifférent)
        exposition = self.user_preferences.get('exposition')
        if exposition and exposition != 'indifférent':
            # Cherche si l'exposition est dans la colonne ensoleillement
            mask = filtered_df['ensoleillement'].str.contains(exposition, case=False, na=False)
            filtered_df = filtered_df[mask]
            print(f"📊 Après filtre exposition: {len(filtered_df)} plantes")
        
        # Filtre par entretien
        entretien = self.user_preferences.get('entretien')
        if entretien == 'faible':
            # Plantes avec besoin eau = 0 ou 1
            filtered_df = filtered_df[filtered_df['besoin_eau'] <= 1]
            print(f"📊 Après filtre entretien faible: {len(filtered_df)} plantes")
        
        # Filtre par style (exclut le mobilier pour un jardin naturel)
        style = self.user_preferences.get('style')
        if style in ['japonais', 'naturel', 'mediterraneen']:
            filtered_df = filtered_df[~filtered_df['type_excel'].str.contains('Mobilier|Décoration', na=False)]
            print(f"📊 Après filtre style jardin: {len(filtered_df)} plantes")
        
        return filtered_df
    
    def score_plants(self, filtered_df: pd.DataFrame) -> pd.DataFrame:
        """Note les plantes selon les préférences"""
        scores = []
        
        for idx, row in filtered_df.iterrows():
            score = 0
            
            # Score basé sur le style
            style = self.user_preferences.get('style')
            if style == 'japonais':
                mots_japonais = ['japon', 'acer', 'maple', 'érable', 'bambou', 'mousse']
                if any(mot in row['nom'].lower() for mot in mots_japonais):
                    score += 3
                    
            elif style == 'mediterraneen':
                mots_med = ['olivier', 'lavande', 'romarin', 'thym', 'citron']
                if any(mot in row['nom'].lower() for mot in mots_med):
                    score += 3
                    
            elif style == 'moderne':
                if 'graphique' in row['nom'].lower() or 'design' in row['nom'].lower():
                    score += 2
            
            # Score basé sur la rusticité
            if row['rusticite_valeur'] == 'Haute':
                score += 1
                
            # Score basé sur la couleur (bonus)
            if self.user_preferences.get('couleur_pref'):
                if self.user_preferences['couleur_pref'].lower() in row['couleur'].lower():
                    score += 2
            
            scores.append(score)
        
        filtered_df['score'] = scores
        return filtered_df.sort_values('score', ascending=False)
    
    def recommend(self, n_plants: int = 8) -> List[Dict[str, Any]]:
        """
        Retourne les meilleures recommandations
        
        Args:
            n_plants: Nombre de plantes à recommander
            
        Returns:
            Liste des plantes recommandées
        """
        # Filtre
        filtered = self.filter_by_criteria()
        
        if len(filtered) == 0:
            print("⚠️ Aucune plante après filtrage, retour de toutes les plantes")
            filtered = self.df.copy()
        
        # Score
        scored = self.score_plants(filtered)
        
        # Prend les meilleures
        recommendations = scored.head(n_plants).to_dict('records')
        print(f"🎯 {len(recommendations)} plantes recommandées")
        
        return recommendations