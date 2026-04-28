"""
Application principale Streamlit pour la recommandation de plantes
"""
import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import os

# Ajoute src au path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.recommender import PlantRecommender
from src.rag_engine import RAGEngine
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="Assistant Jardinier IA",
    page_icon="🌿",
    layout="wide"
)

# Titre
st.title("🌿 Assistant Jardinier IA")
st.markdown("Créez votre jardin de rêve avec notre assistant intelligent basé sur Mistral")

# Initialisation des données
@st.cache_resource
def init_data():
    """Charge les données et initialise les composants"""
    data_path = Path("data")
    
    if not data_path.exists():
        st.error(f"❌ Dossier data non trouvé: {data_path.absolute()}")
        return None, None, None
    
    loader = DataLoader(data_path)
    
    try:
        df = loader.create_plante_dataframe()
        recommender = PlantRecommender(df)
        
        # Initialise le RAG
        rag = RAGEngine()
        
        # Crée le vector store si nécessaire
        vector_path = Path("./vector_store/text_embeddings")
        if not vector_path.exists() or not list(vector_path.glob("*.parquet")):
            with st.spinner("🔄 Création de la base de connaissances..."):
                rag.create_vectorstore(df)
        
        return df, recommender, rag
        
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

# Charge les données
df, recommender, rag = init_data()

if df is None:
    st.stop()

# Questionnaire
st.header("📝 Questionnaire personnalisé")

with st.form("jardin_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 🌸 Description de votre jardin rêvé")
        description = st.text_area(
            "Décrivez votre jardin idéal",
            placeholder="Ex: un jardin japonais avec des érables et un bassin...",
            height=100
        )
        
        st.subheader("2. 🏛️ Style principal")
        style = st.selectbox(
            "Style",
            options=["japonais", "mediterraneen", "moderne", "naturel", "potager"],
            index=3
        )
        
        st.subheader("3. 📏 Taille de l'espace")
        taille = st.select_slider(
            "Taille",
            options=["balcon", "petit", "moyen", "grand"],
            value="moyen"
        )
        
        st.subheader("4. 🌱 Nombre de plantes")
        n_plants = st.slider("Combien de plantes ?", 1, 15, 8)
    
    with col2:
        st.subheader("5. 🌤️ Climat")
        climat = st.selectbox(
            "Climat de votre région",
            options=["tempere", "mediterraneen", "continental", "oceanique"],
            index=0
        )
        
        st.subheader("6. ☀️ Exposition")
        exposition = st.selectbox(
            "Exposition au soleil",
            options=["plein_soleil", "mi_ombre", "ombre", "indifférent"],
            index=1
        )
        
        st.subheader("7. 💰 Budget")
        budget = st.select_slider(
            "Budget",
            options=["faible", "moyen", "eleve"],
            value="moyen"
        )
        
        st.subheader("8. ⏱️ Entretien")
        entretien = st.select_slider(
            "Temps disponible",
            options=["faible", "moyen", "eleve"],
            value="moyen"
        )
    
    submitted = st.form_submit_button("🎯 Obtenir mes recommandations", type="primary")

if submitted:
    # Prépare les préférences
    preferences = {
        'description': description,
        'style': style,
        'taille': taille,
        'n_plants': n_plants,
        'climat': climat,
        'exposition': exposition,
        'budget': budget,
        'entretien': entretien
    }
    
    # Récapitulatif
    with st.expander("📋 Récapitulatif de vos préférences", expanded=True):
        for key, value in preferences.items():
            st.write(f"**{key}:** {value}")
    
    # Obtient les recommandations
    with st.spinner("🔍 Recherche des meilleures plantes..."):
        recommender.set_preferences(preferences)
        recommendations = recommender.recommend(n_plants)
    
    # Affiche les résultats
    st.header("🌱 Vos plantes recommandées")
    
    # Recherche sémantique avec le RAG si description non vide
    if description and rag:
        with st.spinner("🤖 Génération de conseils personnalisés avec Mistral..."):
            try:
                rag_results = rag.search_similar_plants(description, k=5)
                
                # Prépare le contexte pour Mistral
                context_plants = []
                for doc, score in rag_results:
                    context_plants.append(doc.metadata)
                
                # Génère la recommandation
                ai_advice = rag.generate_recommendation(description, context_plants)
                st.info(f"💡 **Conseil IA:** {ai_advice}")
            except Exception as e:
                st.warning(f"⚠️ Erreur avec le RAG: {e}")
    
    # Affiche les plantes recommandées
    cols = st.columns(2)
    for idx, plant in enumerate(recommendations):
        with cols[idx % 2]:
            with st.container():
                st.subheader(f"🌿 {plant['nom']}")
                
                # Charge et affiche l'image
                image_path = Path("data/all_photos") / Path(plant['image_path']).name
                if image_path.exists():
                    try:
                        img = Image.open(image_path)
                        st.image(img, use_column_width=True)
                    except Exception as e:
                        st.warning(f"Impossible d'ouvrir l'image: {e}")
                else:
                    st.info(f"🖼️ Image non disponible: {image_path.name}")
                
                # Infos
                st.markdown(f"""
                **Type:** {plant.get('type_excel', 'N/A')}  
                **Couleur:** {plant.get('couleur', 'N/A')}  
                **Exposition:** {plant.get('ensoleillement', 'N/A')}  
                **Entretien:** {plant.get('entretien', 'N/A')}  
                **Rusticité:** {plant.get('rusticite_valeur', 'N/A')}
                """)
                
                st.divider()

# Sidebar avec infos
with st.sidebar:
    st.header("ℹ️ À propos")
    st.markdown("""
    Cette application utilise:
    - **Ollama + Mistral** pour la génération de conseils
    - **ChromaDB** pour la recherche vectorielle
    - **Streamlit** pour l'interface
    
    ### Comment ça marche?
    1. Tu réponds au questionnaire
    2. Le système filtre et score les plantes
    3. Mistral génère des conseils personnalisés
    4. Tu vois les résultats avec images
    
    ### 🔧 Vérification
    """)
    
    if rag:
        st.success("✅ RAG initialisé")
    else:
        st.warning("⚠️ RAG non initialisé")
    
    if recommender:
        st.success(f"✅ {len(df)} plantes disponibles")
    
    # Vérifie Ollama
    try:
        import ollama
        response = ollama.list()
        st.success("✅ Ollama est actif")
    except:
        st.error("❌ Ollama n'est pas actif. Lance 'ollama serve' dans un terminal")