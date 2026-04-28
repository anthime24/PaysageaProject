import { useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
    Upload, Sparkles, Waves, Trees, Shovel, Palette,
    Infinity, Clock, Sprout, Fence, Layout, Sofa,
    Citrus, Footprints, Droplet, Palmtree, Users, Cloud, Bird,
    Wind, Flower2, Minimize2, Sun, MessageSquare, Lightbulb, MousePointer2
} from 'lucide-react'
import useStore from '../store/useStore'
import AddressSearch from '../components/AddressSearch'
import PremiumDropdown from '../components/PremiumDropdown'
import { SUGGESTIONS } from '../data/suggestions'
import SelectionOverlay from '../components/SelectionOverlay'

const UploadPage = () => {
    const navigate = useNavigate()
    const { previewUrl, setImage, setPreprocessData, filters, setFilters, projectContext, setProjectContext, enrichDescription, userZone, preprocessData } = useStore()
    const [isUploading, setIsUploading] = useState(false)
    const [showSelection, setShowSelection] = useState(false)

    const handleFileChange = async (e) => {
        const file = e.target.files[0]
        if (!file) return

        setImage(file)
        setIsUploading(true)

        // IA BRIDGE: Upload immédiat pour preprocess
        try {
            const formData = new FormData()
            formData.append('image', file)

            const API_URL = import.meta.env.VITE_CLIMATE_API_URL || 'http://localhost:3001'
            const response = await fetch(`${API_URL}/api/project/upload`, {
                method: 'POST',
                body: formData
            })

            if (!response.ok) throw new Error('Erreur upload/preprocess')
            
            const result = await response.json()
            console.log('✅ Preprocess terminé:', result)
            setPreprocessData(result)
        } catch (err) {
            console.error('❌ Erreur upload:', err)
            setImage(null) // Reset on error
            alert("Erreur lors du prétraitement de l'image. Est-ce que le backend est lancé ?")
        } finally {
            setIsUploading(false)
        }
    }

    const handleResolve = useCallback((context) => {
        setProjectContext(context)
    }, [setProjectContext])

    return (
        <div className="h-full flex flex-col items-center max-w-6xl mx-auto space-y-12 animate-in fade-in duration-700">
            {/* Address Search Bar - Elegant and Integrated */}
            <div className="w-full max-w-2xl px-4 z-30">
                <div className="bg-white/80 backdrop-blur-md rounded-2xl shadow-lg border border-gray-100 p-2 transform hover:scale-[1.01] transition-transform duration-300">
                    <AddressSearch onResolve={handleResolve} />
                </div>
            </div>

            {/* Main Visual Focus - The Garden Image */}
            <div className="w-full relative px-4">
                <label className={`
                    block w-full cursor-pointer transition-all duration-500
                    ${previewUrl ? 'rounded-none shadow-2xl bg-neutral-900 border border-white/10' : 'h-[60vh] rounded-3xl border-2 border-dashed border-gray-200 bg-white/50 hover:bg-white'}
                    overflow-hidden relative group max-w-5xl mx-auto
                `}>
                    <input type="file" className="hidden" onChange={handleFileChange} accept="image/*" />

                    {previewUrl ? (
                        <div className="relative w-full h-full flex items-center justify-center">
                            <div className="relative w-full shadow-2xl">
                                <img src={previewUrl} className={`w-full h-auto block transition-opacity duration-500 ${isUploading ? 'opacity-30 blur-sm' : 'opacity-100'}`} alt="Votre Jardin" />
                                
                                {/* Visualisation de la zone sélectionnée (grisé hachuré) */}
                                {!isUploading && userZone.length > 0 && (
                                    <svg 
                                        className="absolute inset-0 w-full h-full pointer-events-none" 
                                        viewBox="0 0 100 100" 
                                        preserveAspectRatio="none"
                                    >
                                        <defs>
                                            <pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                                                <line x1="0" y1="0" x2="0" y2="8" stroke="rgba(255,255,255,0.4)" strokeWidth="1" />
                                            </pattern>
                                        </defs>
                                        <polygon
                                            points={userZone.map(p => `${p.x * 100},${p.y * 100}`).join(' ')}
                                            fill="url(#hatch)"
                                            fillOpacity="0.8"
                                            stroke="rgba(255,255,255,0.5)"
                                            strokeWidth="0.2"
                                            className="animate-pulse"
                                        />
                                    </svg>
                                )}
                            </div>
                            
                            <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent pointer-events-none" />
                            
                            {isUploading && (
                                <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4">
                                    <div className="w-12 h-12 border-4 border-[var(--color-nature)] border-t-transparent rounded-full animate-spin" />
                                    <p className="text-[var(--color-nature)] font-black uppercase tracking-widest text-xs">Optimisation de l'image...</p>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="absolute inset-0 flex flex-col items-center justify-center p-12 text-center space-y-6">
                            <div className="w-24 h-24 rounded-3xl bg-[var(--color-nature)]/10 flex items-center justify-center text-[var(--color-nature)] animate-pulse">
                                <Upload size={48} />
                            </div>
                            <div className="space-y-2">
                                <h3 className="text-2xl font-bold tracking-tight text-[var(--color-structure)]">Capturez votre extérieur</h3>
                                <p className="text-gray-400 max-w-sm">Déposez une photo de votre jardin pour commencer la transformation IA.</p>
                            </div>
                            <div className="px-8 py-4 bg-[var(--color-nature)] text-white rounded-2xl font-bold shadow-lg shadow-[#7b9872]/20 hover:scale-105 transition-transform duration-300">
                                Sélectionner une image
                            </div>
                        </div>
                    )}

                    {previewUrl && (
                        <div className="absolute top-6 right-6 flex items-center gap-3 opacity-0 group-hover:opacity-100 transition-opacity">
                            {userZone.length > 0 && (
                                <div className="px-4 py-2 bg-[var(--color-nature)] text-white rounded-full text-[10px] font-black uppercase tracking-widest shadow-xl border border-white/20 flex items-center gap-2">
                                    <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                                    Zone définie ({userZone.length} pts)
                                </div>
                            )}
                            <div className="px-6 py-3 bg-white/90 backdrop-blur text-[var(--color-structure)] rounded-full text-sm font-bold shadow-xl border border-gray-100">
                                Remplacer l'image
                            </div>
                        </div>
                    )}
                </label>

                {/* Selection Tool Trigger */}
                {previewUrl && !isUploading && (
                    <motion.div 
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="absolute bottom-10 left-1/2 -translate-x-1/2 z-20"
                    >
                        <button 
                            onClick={(e) => {
                                e.preventDefault()
                                e.stopPropagation()
                                setShowSelection(true)
                            }}
                            className="group flex items-center gap-3 px-8 py-4 bg-white/90 backdrop-blur-xl text-[var(--color-structure)] rounded-2xl font-black uppercase tracking-widest text-xs shadow-2xl border border-white hover:bg-[var(--color-nature)] hover:text-white transition-all duration-300 transform hover:scale-105"
                        >
                            <MousePointer2 size={18} className="group-hover:scale-110 transition-transform" />
                            {userZone.length > 0 ? 'Modifier la zone plantable' : 'Définir la zone plantable'}
                        </button>
                    </motion.div>
                )}
            </div>

            <SelectionOverlay 
                isOpen={showSelection} 
                onClose={() => setShowSelection(false)} 
                imageUrl={previewUrl}
            />

            {/* Style & Maintenance Dropdowns */}
            <div className="w-full max-w-4xl px-4 animate-in slide-in-from-bottom-4 duration-1000">
                <div className="grid grid-cols-2 gap-4">
                    <PremiumDropdown
                        label="Style de jardin"
                        icon={Palette}
                        value={filters.style}
                        onChange={(v) => setFilters({ style: v })}
                        options={[
                            { id: 'naturel',      label: 'Naturel',        icon: Trees,   desc: 'Vivaces, graminées, prairie' },
                            { id: 'japonais',     label: 'Japonais',       icon: Wind,    desc: 'Érable, bambou, mousse, zen' },
                            { id: 'mediterraneen',label: 'Méditerranéen',  icon: Citrus,  desc: 'Olivier, lavande, romarin' },
                            { id: 'champetre',    label: 'Champêtre',      icon: Flower2, desc: 'Rosier, lilas, pivoine' },
                            { id: 'moderne',      label: 'Moderne',        icon: Layout,  desc: 'Graphique, architectural, if' },
                            { id: 'potager',      label: 'Potager',        icon: Sprout,  desc: 'Aromates, légumes, fraisiers' },
                        ]}
                    />
                    <PremiumDropdown
                        label="Niveau d'entretien"
                        icon={Shovel}
                        value={filters.maintenance}
                        onChange={(v) => setFilters({ maintenance: v })}
                        options={[
                            { id: 'faible',   label: 'Faible',   icon: Droplet,   desc: 'Plantes résistantes, peu d\'arrosage' },
                            { id: 'modere',   label: 'Modéré',   icon: Clock,     desc: 'Taille saisonnière, arrosage régulier' },
                            { id: 'intensif', label: 'Intensif', icon: Infinity,  desc: 'Fleurissement dense, suivi régulier' },
                        ]}
                    />
                </div>
            </div>

            {/* Custom Description Input */}
            <div className="w-full max-w-4xl px-4 animate-in slide-in-from-bottom-4 duration-1000 delay-100">
                <div className="relative group">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-[var(--color-nature)]/20 to-[var(--color-action)]/20 rounded-3xl blur opacity-30 group-focus-within:opacity-100 transition duration-1000"></div>
                    <div className="relative bg-white/70 backdrop-blur-xl border border-gray-100 rounded-3xl p-6 shadow-xl space-y-4">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-[var(--color-nature)]/10 flex items-center justify-center text-[var(--color-nature)]">
                                <MessageSquare size={20} />
                            </div>
                            <div>
                                <h4 className="text-sm font-black text-[var(--color-structure)] uppercase tracking-widest">Décrivez votre jardin idéal</h4>
                                <p className="text-[10px] text-gray-400 font-bold uppercase tracking-widest">Vos envies, vos contraintes, vos rêves...</p>
                            </div>
                        </div>
                        <textarea
                            value={filters.description}
                            onChange={(e) => setFilters({ description: e.target.value })}
                            placeholder="Ex: Je souhaite un jardin avec beaucoup de fleurs blanches, un petit coin pour lire à l'ombre et un éclairage tamisé pour le soir..."
                            className="w-full h-32 bg-transparent border-none focus:ring-0 focus:outline-none text-[var(--color-structure)] placeholder:text-gray-300 resize-none font-medium leading-relaxed"
                        />
                    </div>
                </div>

                {/* Intelligent Suggestions Pills */}
                <div className="px-2 mt-10">
                    <div className="flex items-center gap-2 mb-3">
                        <Lightbulb size={14} className="text-[var(--color-action)]" />
                        <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Idées rapides</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {SUGGESTIONS.ideas.map((idea) => {
                            const isApplied = filters.appliedSuggestions.includes(idea.id);
                            return (
                                <button
                                    key={idea.id}
                                    onClick={() => enrichDescription(idea.phrase, idea.id)}
                                    className={`
                                        px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all duration-300
                                        ${isApplied 
                                            ? 'bg-[var(--color-nature)] text-white shadow-md shadow-[#7b9872]/20' 
                                            : 'bg-white/80 text-gray-400 border border-gray-100 hover:border-[var(--color-nature)]/30 hover:bg-white hover:text-[var(--color-nature)]'}
                                    `}
                                >
                                    {idea.label}
                                </button>
                            );
                        })}
                    </div>
                </div>
            </div>
            {/* ─── BOUTON LANCER L'ANALYSE ─────────────────────────────────── */}
            {previewUrl && !isUploading && (
                <div className="w-full max-w-4xl px-4 pb-8">
                    <button
                        id="btn-launch-analysis"
                        onClick={() => navigate('/processing')}
                        disabled={!previewUrl}
                        className="
                            w-full py-5 px-8 rounded-3xl font-black uppercase tracking-widest text-sm
                            shadow-2xl shadow-[var(--color-nature)]/30 transition-all duration-300
                            flex items-center justify-center gap-4
                            bg-gradient-to-r from-[var(--color-nature)] to-[#6a8763]
                            text-white hover:scale-[1.02] hover:shadow-[var(--color-nature)]/40
                            disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100
                            relative overflow-hidden group
                        "
                    >
                        <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                        <Sprout size={22} />
                        <span>Lancer l&apos;analyse IA</span>
                        {userZone.length > 0 && (
                            <span className="ml-2 px-3 py-1 bg-white/20 rounded-full text-[10px] font-black tracking-widest">
                                Zone définie · {userZone.length} pts
                            </span>
                        )}
                    </button>
                    {!userZone.length && (
                        <p className="text-center text-[10px] text-amber-500 font-bold uppercase tracking-widest mt-3">
                            ⚠ Définissez d&apos;abord votre zone plantable avant de lancer l&apos;analyse
                        </p>
                    )}
                </div>
            )}
        </div>
    )
}

export default UploadPage
