import { create } from 'zustand'
import { buildPlantFilter } from '../utils/buildPlantFilter'

const useStore = create((set) => ({
    image: null,
    previewUrl: null,
    filters: {
        style: 'naturel',
        maintenance: 'modere',
        elements: [],
        atmosphere: 'esthetique',
        description: '',
        appliedSuggestions: [], // Track applied IDs to avoid duplicates
    },
    analysisResult: null,
    projectId: null,
    projectContext: null,
    plantFilter: null,
    preprocessData: null, // Stocke les résultats du preprocess (image_id, web_url, paths)
    userZone: [], // Liste des points [{x, y}] de la zone plantable (normalisés 0-1)
    generatedImageUrl: null, // URL de l'image finale générée par BFL
    isGenerating: false, // true pendant l'appel generate-image


    setImage: (file) => {
        const url = URL.createObjectURL(file)
        set({ image: file, previewUrl: url, preprocessData: null })
    },

    setPreprocessData: (data) => set({ 
        preprocessData: data,
        previewUrl: data.web_url, // On bascule sur l'image pré-traitée
        userZone: [] // Reset de la zone quand on change d'image ou de preprocess
    }),

    setUserZone: (zone) => set({ userZone: zone }),

    setGeneratedImageUrl: (url) => set({ generatedImageUrl: url }),
    setIsGenerating: (val) => set({ isGenerating: val }),

    setFilters: (filters) => set((state) => ({
        filters: { ...state.filters, ...filters }
    })),

    toggleElement: (elementId) => set((state) => {
        const elements = state.filters.elements.includes(elementId)
            ? state.filters.elements.filter(e => e !== elementId)
            : [...state.filters.elements, elementId]
        return { filters: { ...state.filters, elements } }
    }),

    setAnalysisResult: (result) => set({ analysisResult: result }),
    setProjectId: (id) => set({ projectId: id }),
    setProjectContext: (context) => {
        const plantFilter = context ? buildPlantFilter(context) : null
        set({ projectContext: context, plantFilter })
    },

    // HELPER: Construit le manifeste complet pour le RAG / Image Gen
    getProjectManifest: () => {
        const state = useStore.getState()
        return {
            project_id: state.projectId || 'demo-' + Math.random().toString(36).substr(2, 9),
            user_intent: {
                description: state.filters.description,
                style: state.filters.style,
                maintenance: state.filters.maintenance,
                applied_tags: state.filters.appliedSuggestions
            },
            visual_context: {
                image_url: state.preprocessData?.web_url || state.previewUrl,
                user_zone: state.userZone, // Les points dessinés par l'utilisateur
                image_size: state.preprocessData?.image_size // Dimensions [w, h] pour repère
            },
            environmental_context: {
                location: state.projectContext?.location,
                botanical_filter: state.plantFilter,
                climate_summary: state.projectContext?.annual_profile?.summary
            }
        }
    },

    enrichDescription: (phrase, id) => set((state) => {
        const currentDesc = state.filters.description.trim()
        const newDesc = currentDesc ? `${currentDesc}\n${phrase}` : phrase
        const applied = state.filters.appliedSuggestions.includes(id)
            ? state.filters.appliedSuggestions
            : [...state.filters.appliedSuggestions, id]
        
        return {
            filters: {
                ...state.filters,
                description: newDesc,
                appliedSuggestions: applied
            }
        }
    })
}))

export default useStore
