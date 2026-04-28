"""
Ce fichier charge et prépare les données des plantes
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

class DataLoader:
    def __init__(self, data_path: str):
        """
        Initialise le chargeur de données
        
        Args:
            data_path: Chemin vers le dossier contenant plantes_data.json
        """
        self.data_path = Path(data_path)
        
    def load_plantes_data(self) -> List[Dict[str, Any]]:
        """Charge les données des plantes depuis le fichier JSON"""
        json_file = self.data_path / 'plantes_data.json'
        
        if not json_file.exists():
            raise FileNotFoundError(f"Fichier {json_file} non trouvé!")
            
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ {len(data)} plantes chargées")
        return data
    
    def create_plante_dataframe(self) -> pd.DataFrame:
        """Crée un DataFrame pandas pour faciliter les filtres"""
        data = self.load_plantes_data()
        df = pd.DataFrame(data)
        
        # Nettoie les données
        df['besoin_eau'] = pd.to_numeric(df['besoin_eau'], errors='coerce').fillna(0)
        
        # Crée une colonne texte combinée pour la recherche
        df['combined_text'] = df.apply(self._combine_plant_info, axis=1)
        
        print(f"✅ DataFrame créé avec {len(df)} plantes")
        return df
    
    def _combine_plant_info(self, row: pd.Series) -> str:
        """Combine toutes les informations de la plante pour la recherche"""
        return f"""
        Nom: {row['nom']}
        Type: {row['type_excel']}
        Sous-type: {row['sous_type_excel']}
        Couleur: {row['couleur']}
        Exposition: {row['ensoleillement']}
        Rusticité: {row['rusticite_valeur']}
        Entretien: {row['entretien']}
        """