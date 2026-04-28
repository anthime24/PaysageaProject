"""
Ce fichier gère la recherche vectorielle avec ChromaDB
"""
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import pandas as pd
from typing import List, Tuple
import ollama
import os

class RAGEngine:
    def __init__(self, persist_directory: str = "./vector_store/text_embeddings"):
        """
        Initialise le moteur RAG avec Ollama (Mistral) pour la génération
        
        Args:
            persist_directory: Dossier où stocker les embeddings
        """
        self.persist_directory = persist_directory
        
        # Crée le dossier s'il n'existe pas
        os.makedirs(persist_directory, exist_ok=True)
        
        # Utilise un modèle d'embedding local pour la recherche
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        
    def create_vectorstore(self, df: pd.DataFrame):
        """Crée la base de données vectorielle avec les plantes"""
        print("🔄 Création du vector store...")
        
        documents = []
        for idx, row in df.iterrows():
            # Vérifie que le texte n'est pas vide
            combined_text = row.get('combined_text', '')
            if not combined_text or combined_text == '':
                combined_text = f"Nom: {row.get('nom', '')} Type: {row.get('type_excel', '')}"
            
            doc = Document(
                page_content=combined_text,
                metadata={
                    'id': str(row.get('id', idx)),
                    'nom': row.get('nom', 'Inconnu'),
                    'image_path': row.get('image_path', ''),
                    'type_excel': row.get('type_excel', ''),
                    'couleur': row.get('couleur', ''),
                    'ensoleillement': row.get('ensoleillement', ''),
                    'entretien': row.get('entretien', '')
                }
            )
            documents.append(doc)
        
        # Crée le vector store avec Chroma
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Sauvegarde persistante
            self.vectorstore.persist()
            print(f"✅ Vector store créé avec {len(documents)} documents")
        except Exception as e:
            print(f"❌ Erreur création vector store: {e}")
            raise
    
    def search_similar_plants(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Recherche les plantes les plus similaires à la requête
        
        Args:
            query: Texte de recherche
            k: Nombre de résultats à retourner
            
        Returns:
            Liste de tuples (document, score)
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore non initialisé. Appelle create_vectorstore d'abord.")
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            print(f"🔍 Recherche '{query}' -> {len(results)} résultats")
            return results
        except Exception as e:
            print(f"❌ Erreur recherche: {e}")
            return []
    
    def generate_recommendation(self, query: str, context_plants: List[Dict]) -> str:
        """
        Utilise Mistral (via Ollama) pour générer une recommandation personnalisée
        
        Args:
            query: La requête utilisateur
            context_plants: Liste des plantes à utiliser comme contexte
            
        Returns:
            Recommandation textuelle générée
        """
        # Prépare le contexte des plantes
        context_text = ""
        for i, plant in enumerate(context_plants[:5], 1):  # Limite à 5 plantes
            context_text += f"""
            {i}. {plant.get('nom', 'Inconnu')}
               - Type: {plant.get('type_excel', 'N/A')}
               - Couleur: {plant.get('couleur', 'N/A')}
               - Exposition: {plant.get('ensoleillement', 'N/A')}
            """
        
        # Crée le prompt pour Mistral
        prompt = f"""Tu es un expert jardinier. Voici les préférences d'un client:
{query}

Voici quelques plantes qui pourraient correspondre:
{context_text}

Peux-tu:
1. Expliquer pourquoi ces plantes sont adaptées
2. Donner des conseils d'aménagement
3. Suggérer d'autres idées si nécessaire

Réponds de manière concise et utile en français."""

        # Appelle Ollama avec Mistral
        try:
            response = ollama.chat(
                model='mistral',
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            print(f"❌ Erreur avec Ollama: {e}")
            return "Désolé, je n'ai pas pu générer de recommandation détaillée. Vérifie que Ollama est installé et que Mistral est téléchargé (ollama pull mistral)."