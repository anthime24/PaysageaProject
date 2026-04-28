import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Download, Check, Sparkles, Wand2, Loader2, AlertCircle, Layers, Eye, EyeOff, Crosshair } from 'lucide-react'
import useStore from '../store/useStore'

const DEPTH_BAND_LABELS = { back: 'Fond', mid: 'Plan médian', front: 'Premier plan' }
const DEPTH_BAND_COLORS = { back: '#6366f1', mid: '#10b981', front: '#f59e0b' }

const ResultPage = () => {
    const navigate = useNavigate()
    const {
        previewUrl, filters, projectContext, analysisResult, preprocessData,
        generatedImageUrl, setGeneratedImageUrl, isGenerating, setIsGenerating, plantFilter
    } = useStore()

    const [viewMode, setViewMode] = useState('original') // 'ia' | 'original'
    const [genError, setGenError] = useState(null)
    const [genMeta, setGenMeta] = useState(null) // { segments_used, mask_white_pct, plant_masks[] }
    const [editMode, setEditMode] = useState(false)
    const [activeMaskIndex, setActiveMaskIndex] = useState(null)
    const [maskOffsets, setMaskOffsets] = useState({}) // { [index]: {dx, dy} }
    const draggingRef = useRef(null) // { index, startX, startY, startDx, startDy }

    const handleMaskMouseDown = (e, i) => {
        e.preventDefault()
        e.stopPropagation()
        setActiveMaskIndex(i)
        draggingRef.current = {
            index: i,
            startX: e.clientX,
            startY: e.clientY,
            startDx: maskOffsets[i]?.dx || 0,
            startDy: maskOffsets[i]?.dy || 0,
        }
    }

    useEffect(() => {
        const onMove = (e) => {
            if (!draggingRef.current) return
            const { index, startX, startY, startDx, startDy } = draggingRef.current
            setMaskOffsets(prev => ({
                ...prev,
                [index]: {
                    dx: startDx + (e.clientX - startX),
                    dy: startDy + (e.clientY - startY),
                }
            }))
        }
        const onUp = () => { draggingRef.current = null }
        window.addEventListener('mousemove', onMove)
        window.addEventListener('mouseup', onUp)
        return () => {
            window.removeEventListener('mousemove', onMove)
            window.removeEventListener('mouseup', onUp)
        }
    }, [])

    const resetMaskOffsets = () => setMaskOffsets({})

    // ── Génération ──────────────────────────────────────────────────────────
    const handleGenerateImage = async () => {
        setIsGenerating(true)
        setGenError(null)
        setGenMeta(null)
        setActiveMaskIndex(null)
        setViewMode('ia')

        try {
            const API_URL = import.meta.env.VITE_CLIMATE_API_URL || 'http://localhost:3001'

            // ── Étape 1 : appel RAG pour obtenir les plantes recommandées ────
            const annualSummary = projectContext?.annual_profile?.summary || {}
            // Mappe sun_exposure buildPlantFilter ("sun"/"partial"/"shade") → RAG ("plein_soleil"/"mi_ombre"/"ombre")
            const sunMap = { sun: 'plein_soleil', partial: 'mi_ombre', shade: 'ombre' }
            const exposition = sunMap[plantFilter?.sun_exposure] || 'plein_soleil'
            // Détecte le type de climat depuis le résumé annuel
            const climateRaw = (annualSummary.climate_type || annualSummary.hardiness_zone || '').toLowerCase()
            const climat = climateRaw.includes('médit') || climateRaw.includes('medit') || climateRaw.includes('zone 9') || climateRaw.includes('zone 10')
                ? 'mediterraneen'
                : climateRaw.includes('continent') ? 'continental'
                : climateRaw.includes('océan') || climateRaw.includes('ocean') ? 'oceanique'
                : 'tempere'
            const ragPrefs = {
                style:       filters.style       || 'naturel',
                exposition,
                entretien:   filters.maintenance === 'faible' ? 'faible'
                    : filters.maintenance === 'intensif' ? 'intensif'
                    : 'moyen',
                description: filters.description  || '',
                climat,
                taille:      'moyen',
                budget:      'moyen',
                region:      projectContext?.location?.short_label || 'France',
                usda_zone:   plantFilter?.usda_zone   || null,
                temp_min:    plantFilter?.temp_min_local || null,
                n_plants:    6,
            }
            try {
                const ragRes = await fetch(`${API_URL}/api/rag/recommend`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(ragPrefs),
                })
                if (!ragRes.ok) console.warn('[RAG] Avertissement:', await ragRes.text())
                else console.log('[RAG] Plantes chargées ✓')
            } catch (ragErr) {
                console.warn('[RAG] Erreur (génération continue sans RAG):', ragErr.message)
            }

            const response = await fetch(`${API_URL}/api/project/generate-image`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    plant_density: filters.maintenance === 'minimal' ? 'low'
                        : filters.maintenance === 'intensif' ? 'high'
                        : 'medium',
                    description: filters.description || '',
                    seed: Math.floor(Math.random() * 9999),
                    max_plants: 6,
                })
            })

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}))
                throw new Error(errData.error || `Erreur HTTP ${response.status}`)
            }

            const result = await response.json()
            console.log('✅ [IMAGE-GEN] Résultat:', result)

            if (result.web_url) {
                setGeneratedImageUrl(`${result.web_url}?t=${Date.now()}`)
                setGenMeta({
                    segments_used: result.segments_used || 0,
                    mask_white_pct: result.mask_white_pct || 0,
                    prompt_preview: result.prompt_preview || '',
                    pipeline_json: result.pipeline_json_used || '',
                    plant_masks: result.plant_masks || [],
                })
                setViewMode('ia')
            } else {
                throw new Error("Pas d'URL dans la réponse du serveur")
            }
        } catch (err) {
            console.error('❌ [IMAGE-GEN] Erreur:', err)
            setGenError(err.message)
            setViewMode('original')
        } finally {
            setIsGenerating(false)
        }
    }

    const handleDownload = () => {
        const targetUrl = generatedImageUrl || previewUrl
        if (!targetUrl) return
        const a = document.createElement('a')
        a.href = targetUrl
        a.download = `paysagea-jardin-${new Date().getTime()}.png`
        a.click()
    }

    const getActiveImage = () => {
        if (viewMode === 'ia') return generatedImageUrl || previewUrl
        return preprocessData?.web_url || previewUrl
    }

    const activePlantMask = (genMeta?.plant_masks || [])[activeMaskIndex] || null

    return (
        <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in duration-1000 pb-20">

            {/* Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
                <div className="space-y-3">
                    <button
                        onClick={() => navigate('/upload')}
                        className="group flex items-center gap-2 text-gray-400 hover:text-[var(--color-nature)] transition-colors text-sm font-bold uppercase tracking-widest"
                    >
                        <ArrowLeft size={16} className="transition-transform group-hover:-translate-x-1" />
                        <span>Retour au studio</span>
                    </button>
                    <div>
                        <h2 className="text-4xl font-black tracking-tight text-[var(--color-structure)]">
                            Votre Projet <span className="text-[var(--color-nature)] italic">Paysager</span>
                        </h2>
                        <div className="flex items-center gap-3 mt-1">
                            <div className="w-8 h-8 rounded-lg bg-[var(--color-nature)]/10 flex items-center justify-center text-[var(--color-nature)]">
                                <Sparkles size={16} />
                            </div>
                            <p className="text-gray-400 text-sm font-medium">
                                Design optimisé pour <span className="text-[var(--color-structure)] font-bold">{projectContext?.location?.short_label || 'votre région'}</span>
                            </p>
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-3 w-full md:w-auto">
                    <button
                        onClick={handleDownload}
                        disabled={isGenerating}
                        className="flex-1 md:flex-none p-3.5 bg-white border border-gray-100 rounded-2xl shadow-sm hover:shadow-md transition-all text-gray-400 hover:text-[var(--color-structure)] disabled:opacity-40"
                        title="Télécharger l'image"
                    >
                        <Download size={20} className="mx-auto" />
                    </button>

                    {/* Mode édition masques */}
                    {generatedImageUrl && genMeta?.plant_masks?.length > 0 && (
                        <button
                            onClick={() => { setEditMode(e => !e); setActiveMaskIndex(null) }}
                            className={`flex-1 md:flex-none flex items-center gap-2 px-5 py-3.5 rounded-2xl font-black uppercase tracking-widest text-xs transition-all
                                ${editMode
                                    ? 'bg-[var(--color-structure)] text-white shadow-lg'
                                    : 'bg-white border border-gray-100 text-gray-400 hover:text-[var(--color-structure)] shadow-sm'
                                }`}
                        >
                            <Layers size={18} />
                            <span>{editMode ? 'Fermer' : 'Masques'}</span>
                        </button>
                    )}

                    {/* Bouton génération */}
                    <button
                        id="btn-generate-garden-ia"
                        onClick={handleGenerateImage}
                        disabled={isGenerating}
                        className={`
                            flex-[2] md:flex-none flex items-center justify-center gap-3 py-3.5 px-8 rounded-2xl
                            font-black uppercase tracking-widest text-xs shadow-lg transition-all duration-300
                            ${isGenerating
                                ? 'bg-[var(--color-nature)]/60 text-white cursor-not-allowed'
                                : 'bg-[var(--color-nature)] text-white hover:scale-105 hover:shadow-xl hover:shadow-[var(--color-nature)]/30'
                            }
                        `}
                    >
                        {isGenerating
                            ? <><Loader2 size={20} className="animate-spin" /><span>Génération...</span></>
                            : <><Wand2 size={20} /><span>{generatedImageUrl ? 'Regénérer' : 'Générer le Jardin IA'}</span></>
                        }
                    </button>
                </div>
            </div>

            {/* Error */}
            {genError && (
                <div className="flex items-start gap-3 p-5 bg-red-50 border border-red-100 rounded-2xl text-red-600">
                    <AlertCircle size={20} className="flex-shrink-0 mt-0.5" />
                    <div>
                        <p className="font-bold text-sm">Erreur lors de la génération</p>
                        <p className="text-xs mt-1 text-red-400">{genError}</p>
                    </div>
                </div>
            )}

            {/* Layout principal */}
            <div className="max-w-6xl mx-auto w-full space-y-6">

                {/* ── Vue image + éditeur de masques ── */}
                <div className={`grid gap-6 transition-all duration-500 ${editMode ? 'grid-cols-1 lg:grid-cols-3' : 'grid-cols-1'}`}>

                    {/* Image principale */}
                    <div className={`relative rounded-[3rem] overflow-hidden bg-white shadow-2xl border border-gray-100 group
                        ${editMode ? 'h-[450px] lg:col-span-2' : 'h-[calc(100vh-320px)] min-h-[450px] max-h-[750px]'}`}
                    >
                        <img
                            src={getActiveImage()}
                            className={`w-full h-full object-cover transition-all duration-500
                                ${isGenerating ? 'blur-sm scale-105 opacity-50' : 'opacity-100'}
                            `}
                            key={viewMode + (generatedImageUrl || '')}
                            alt="Garden view"
                        />

                        {/* Overlays masques en mode édition (non-actifs, pointer-events none) */}
                        {editMode && (genMeta?.plant_masks || []).map((mask, i) => {
                            if (!mask.mask_web_url || activeMaskIndex === i) return null
                            return (
                                <img
                                    key={i}
                                    src={`${mask.mask_web_url}?t=1`}
                                    draggable={false}
                                    className="absolute inset-0 w-full h-full object-cover mix-blend-screen opacity-30 pointer-events-none select-none"
                                    style={{
                                        filter: `hue-rotate(${i * 40}deg) saturate(2)`,
                                        zIndex: i + 1,
                                    }}
                                    alt=""
                                />
                            )
                        })}

                        {/* Masque actif draggable — rendu dans un wrapper sans overflow-hidden */}
                        {editMode && activeMaskIndex !== null && (() => {
                            const mask = (genMeta?.plant_masks || [])[activeMaskIndex]
                            if (!mask?.mask_web_url) return null
                            const { dx = 0, dy = 0 } = maskOffsets[activeMaskIndex] || {}
                            const cx = mask.centroid ? mask.centroid[0] * 100 : 50
                            const cy = mask.centroid ? mask.centroid[1] * 100 : 50
                            return (
                                <>
                                    <img
                                        key={`active-${activeMaskIndex}`}
                                        src={`${mask.mask_web_url}?t=1`}
                                        draggable={false}
                                        onMouseDown={(e) => handleMaskMouseDown(e, activeMaskIndex)}
                                        className="absolute inset-0 w-full h-full object-cover mix-blend-screen opacity-75 select-none cursor-grab"
                                        style={{
                                            transform: `translate(${dx}px, ${dy}px)`,
                                            filter: `hue-rotate(${activeMaskIndex * 40}deg) saturate(2)`,
                                            zIndex: 10,
                                        }}
                                        alt="Masque actif"
                                    />
                                    {/* Poignée de déplacement au centroïde */}
                                    <div
                                        onMouseDown={(e) => handleMaskMouseDown(e, activeMaskIndex)}
                                        className="absolute w-8 h-8 rounded-full bg-white/90 shadow-lg border-2 border-[var(--color-nature)] flex items-center justify-center cursor-grab select-none"
                                        style={{
                                            left: `calc(${cx}% + ${dx}px - 16px)`,
                                            top: `calc(${cy}% + ${dy}px - 16px)`,
                                            zIndex: 20,
                                        }}
                                    >
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-[var(--color-nature)]">
                                            <path d="M5 9l-3 3 3 3M9 5l3-3 3 3M15 19l-3 3-3-3M19 9l3 3-3 3M12 12v.01"/>
                                            <path d="M3 12h18M12 3v18"/>
                                        </svg>
                                    </div>
                                </>
                            )
                        })()}

                        {/* Overlay chargement */}
                        {isGenerating && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/30 backdrop-blur-sm">
                                <div className="p-8 bg-white/95 backdrop-blur-xl shadow-2xl rounded-[2rem] text-center space-y-4 border border-gray-100 max-w-sm">
                                    <div className="w-16 h-16 rounded-2xl bg-[var(--color-nature)]/10 flex items-center justify-center text-[var(--color-nature)] mx-auto">
                                        <Wand2 size={32} className="animate-pulse" />
                                    </div>
                                    <p className="text-[10px] font-black text-[var(--color-nature)] uppercase tracking-[0.3em]">BFL FLUX Fill PRO · SAM + Depth</p>
                                    <p className="text-lg font-bold text-[var(--color-structure)]">Génération en cours...</p>
                                    <p className="text-xs text-gray-400">Analyse SAM et profondeur en cours... L'IA place les plantes selon la structure de la scène.</p>
                                    <div className="flex items-center justify-center gap-2">
                                        <Loader2 size={16} className="animate-spin text-[var(--color-nature)]" />
                                        <span className="text-xs text-gray-400 font-bold">Inpainting guidé par segmentation...</span>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Badge IA */}
                        {viewMode === 'ia' && generatedImageUrl && !isGenerating && (
                            <div className="absolute top-6 left-6 flex items-center gap-2 px-4 py-2 bg-[var(--color-nature)] text-white rounded-full text-[10px] font-black uppercase tracking-widest shadow-xl border border-white/20">
                                <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                                Jardin IA · {genMeta?.segments_used || 0} zones SAM
                            </div>
                        )}

                        {/* Tabs */}
                        <div className="absolute top-8 left-1/2 -translate-x-1/2 flex items-center gap-4 animate-in slide-in-from-top-4 duration-700">
                            {[
                                { id: 'ia', label: 'IA DESIGN', disabled: !generatedImageUrl },
                                { id: 'original', label: 'PHOTO RÉELLE', disabled: false }
                            ].map(tab => (
                                <button
                                    key={tab.id}
                                    onClick={() => !tab.disabled && setViewMode(tab.id)}
                                    disabled={tab.disabled}
                                    className={`px-6 py-3 rounded-2xl font-black text-[10px] tracking-[0.2em] shadow-xl transition-all duration-300
                                        ${viewMode === tab.id
                                            ? 'bg-[var(--color-nature)] text-white scale-110'
                                            : tab.disabled
                                                ? 'bg-white/30 backdrop-blur text-gray-300 cursor-not-allowed opacity-50'
                                                : 'bg-white/50 backdrop-blur text-gray-400 opacity-50 hover:bg-white/80 hover:opacity-100'
                                        }`}
                                >
                                    {tab.label}
                                    {tab.id === 'ia' && !generatedImageUrl && (
                                        <span className="ml-1 opacity-60 normal-case">(Non généré)</span>
                                    )}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* ── Panneau masques individuels (mode édition) ── */}
                    {editMode && (
                        <div className="bg-white rounded-[2.5rem] shadow-xl border border-gray-100 overflow-hidden flex flex-col">
                            <div className="p-6 border-b border-gray-50">
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-xl bg-[var(--color-nature)]/10 flex items-center justify-center text-[var(--color-nature)]">
                                        <Layers size={20} />
                                    </div>
                                    <div>
                                        <h3 className="font-black text-[var(--color-structure)] text-sm uppercase tracking-widest">Masques Plantes</h3>
                                        <p className="text-[10px] text-gray-400 font-bold mt-0.5">{genMeta?.plant_masks?.length || 0} zones identifiées par SAM</p>
                                    </div>
                                </div>
                            </div>

                            <div className="flex-1 overflow-y-auto p-4 space-y-3">
                                {(genMeta?.plant_masks || []).map((mask, i) => {
                                    const color = DEPTH_BAND_COLORS[mask.depth_band] || '#6366f1'
                                    const label = DEPTH_BAND_LABELS[mask.depth_band] || mask.depth_band
                                    const isActive = activeMaskIndex === i
                                    return (
                                        <button
                                            key={i}
                                            onClick={() => setActiveMaskIndex(isActive ? null : i)}
                                            className={`w-full flex items-center gap-4 p-4 rounded-2xl border-2 transition-all duration-200 text-left
                                                ${isActive
                                                    ? 'border-[var(--color-nature)] bg-[var(--color-nature)]/5 shadow-md'
                                                    : 'border-transparent bg-stone-50 hover:border-gray-200 hover:bg-white'
                                                }`}
                                        >
                                            {/* Miniature masque */}
                                            <div className="w-14 h-14 rounded-xl overflow-hidden bg-gray-100 flex-shrink-0 relative">
                                                <img
                                                    src={`${mask.mask_web_url}?t=1`}
                                                    alt={`Masque plante ${i + 1}`}
                                                    className="w-full h-full object-cover opacity-80"
                                                    style={{ filter: `hue-rotate(${i * 40}deg) saturate(2)` }}
                                                />
                                                {isActive && (
                                                    <div className="absolute inset-0 bg-[var(--color-nature)]/20 flex items-center justify-center">
                                                        <Eye size={16} className="text-[var(--color-nature)]" />
                                                    </div>
                                                )}
                                            </div>

                                            <div className="flex-1 min-w-0">
                                                <div className="flex items-center gap-2 mb-1">
                                                    <div
                                                        className="w-2 h-2 rounded-full flex-shrink-0"
                                                        style={{ backgroundColor: color }}
                                                    />
                                                    <span className="text-[10px] font-black uppercase tracking-widest text-gray-400">{label}</span>
                                                </div>
                                                <p className="text-sm font-bold text-[var(--color-structure)]">Plante {i + 1}</p>
                                                <div className="flex gap-3 mt-1">
                                                    <span className="text-[10px] text-gray-400">
                                                        Seg <span className="font-mono text-gray-600">#{mask.segment_id}</span>
                                                    </span>
                                                    <span className="text-[10px] text-gray-400">
                                                        Prof: <span className="font-mono text-gray-600">{(mask.mean_depth * 100).toFixed(0)}%</span>
                                                    </span>
                                                    <span className="text-[10px] text-gray-400">
                                                        Zone: <span className="font-mono text-gray-600">{(mask.area_ratio * 100).toFixed(1)}%</span>
                                                    </span>
                                                </div>
                                            </div>

                                            <div className="flex-shrink-0">
                                                {isActive
                                                    ? <EyeOff size={16} className="text-[var(--color-nature)]" />
                                                    : <Eye size={16} className="text-gray-300" />
                                                }
                                            </div>
                                        </button>
                                    )
                                })}
                            </div>

                            <div className="p-4 border-t border-gray-50 bg-stone-50 space-y-3">
                                {activePlantMask && (
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <Crosshair size={14} className="text-[var(--color-nature)]" />
                                            <p className="text-[10px] font-black text-[var(--color-nature)] uppercase tracking-widest">
                                                Plante {activeMaskIndex + 1} — {DEPTH_BAND_LABELS[activePlantMask.depth_band]}
                                            </p>
                                        </div>
                                        <p className="text-[10px] text-gray-400 mt-1">
                                            Offset: ({maskOffsets[activeMaskIndex]?.dx?.toFixed(0) ?? 0}px, {maskOffsets[activeMaskIndex]?.dy?.toFixed(0) ?? 0}px)
                                            · Profondeur: {(activePlantMask.mean_depth * 100).toFixed(0)}%
                                        </p>
                                    </div>
                                )}
                                {Object.keys(maskOffsets).length > 0 && (
                                    <button
                                        onClick={resetMaskOffsets}
                                        className="w-full py-2 px-4 rounded-xl bg-white border border-gray-200 text-[10px] font-black uppercase tracking-widest text-gray-400 hover:text-[var(--color-structure)] hover:border-gray-300 transition-all"
                                    >
                                        Réinitialiser positions
                                    </button>
                                )}
                                <p className="text-[10px] text-gray-300 text-center">Sélectionner une plante puis glisser la poignée ✛ sur l'image</p>
                            </div>
                        </div>
                    )}
                </div>

                {/* Info Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {/* SAM + Depth Stats */}
                    <div className="p-10 bg-white rounded-[3rem] shadow-xl border border-gray-100 space-y-6">
                        <div className="flex items-center justify-between border-b border-gray-50 pb-6">
                            <h3 className="text-xl font-black text-[var(--color-structure)] tracking-tight italic">Analyse IA</h3>
                            <Check className="text-green-500" size={20} />
                        </div>
                        <div className="space-y-4">
                            <div className="flex items-center gap-4 p-4 bg-stone-50 rounded-2xl">
                                <span className="text-xl">🧩</span>
                                <div>
                                    <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Segments SAM utilisés</p>
                                    <p className="text-sm font-bold text-[var(--color-structure)]">
                                        {genMeta?.segments_used ?? (analysisResult?.sam?.num_segments || '—')} zones
                                    </p>
                                </div>
                            </div>
                            <div className="flex items-center gap-4 p-4 bg-stone-50 rounded-2xl">
                                <span className="text-xl">📐</span>
                                <div>
                                    <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Zone masquée</p>
                                    <p className="text-sm font-bold text-[var(--color-structure)]">
                                        {genMeta ? `${genMeta.mask_white_pct}%` : '—'}
                                    </p>
                                </div>
                            </div>
                            <div className="flex items-center gap-4 p-4 bg-stone-50 rounded-2xl">
                                <span className="text-xl">📊</span>
                                <div>
                                    <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Ensoleillement</p>
                                    <p className="text-sm font-bold text-[var(--color-structure)]">{projectContext?.annual_profile?.summary?.sunshine_h_per_day || '6.4'}h/jour</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Design Intent */}
                    <div className="p-10 bg-white rounded-[3rem] shadow-xl border border-gray-100 flex flex-col justify-between">
                        <div className="space-y-6">
                            <h3 className="text-xl font-black text-[var(--color-structure)] tracking-tight italic">Intentions</h3>
                            <p className="text-sm text-gray-500 leading-relaxed font-medium">
                                &ldquo;{filters.description || 'Design harmonisé basé sur vos préférences méditerranéennes et bioclimatiques.'}&rdquo;
                            </p>
                            <div className="flex flex-wrap gap-2">
                                {filters.appliedSuggestions.map((tag, i) => (
                                    <span key={i} className="px-3 py-1 bg-stone-50 rounded-full text-[10px] font-bold text-gray-400 uppercase">{tag}</span>
                                ))}
                            </div>
                        </div>
                        <div className="pt-8 border-t border-gray-50">
                            <button
                                onClick={handleGenerateImage}
                                disabled={isGenerating}
                                className="w-full premium-button py-4 text-xs font-black uppercase tracking-widest shadow-lg disabled:opacity-50 flex items-center justify-center gap-2"
                            >
                                {isGenerating
                                    ? <><Loader2 size={16} className="animate-spin" /> Génération...</>
                                    : <><Wand2 size={16} /> Lancer la génération IA</>
                                }
                            </button>
                        </div>
                    </div>

                    {/* Expertise CTA */}
                    <div className="relative p-10 bg-[var(--color-structure)] rounded-[3rem] text-white overflow-hidden group flex flex-col justify-center">
                        <div className="absolute top-0 right-0 w-40 h-40 bg-white/5 rounded-full -mr-20 -mt-20 transition-transform duration-700 group-hover:scale-150" />
                        <div className="relative z-10">
                            <h4 className="text-2xl font-black italic mb-4 tracking-tight text-white">Expertise Pro</h4>
                            <p className="text-sm text-gray-400 mb-8 leading-relaxed font-medium">Transformez ce concept en réalité. Échangez avec un de nos architectes paysagistes.</p>
                            <button className="flex items-center gap-3 text-[var(--color-nature)] font-black uppercase tracking-widest text-xs hover:text-white transition-colors group/btn">
                                Consulter un expert
                                <ArrowLeft size={16} className="rotate-180 transition-transform group-hover/btn:translate-x-2" />
                            </button>
                        </div>
                    </div>
                </div>

                {/* Bandeau succès */}
                {generatedImageUrl && !isGenerating && (
                    <div className="p-6 bg-[var(--color-nature)]/5 border border-[var(--color-nature)]/20 rounded-2xl flex items-center gap-4 flex-wrap">
                        <div className="w-10 h-10 rounded-xl bg-[var(--color-nature)]/10 flex items-center justify-center text-[var(--color-nature)] flex-shrink-0">
                            <Check size={20} />
                        </div>
                        <div className="flex-1">
                            <p className="text-sm font-black text-[var(--color-nature)] uppercase tracking-widest">
                                Jardin généré · {genMeta?.segments_used || 0} zones SAM · {genMeta?.plant_masks?.length || 0} masques plantes
                            </p>
                            <p className="text-xs text-gray-400 mt-0.5">
                                Pipeline: <span className="font-mono">{genMeta?.pipeline_json}</span>
                            </p>
                        </div>
                        <div className="flex gap-3">
                            {genMeta?.plant_masks?.length > 0 && (
                                <button
                                    onClick={() => { setEditMode(true); setViewMode('ia') }}
                                    className="px-4 py-2 bg-[var(--color-structure)] text-white rounded-xl text-xs font-black uppercase tracking-widest hover:scale-105 transition-transform flex items-center gap-1.5"
                                >
                                    <Layers size={14} /> Mode édition
                                </button>
                            )}
                            <button
                                onClick={() => setViewMode('ia')}
                                className="px-4 py-2 bg-[var(--color-nature)] text-white rounded-xl text-xs font-black uppercase tracking-widest hover:scale-105 transition-transform"
                            >
                                Voir l&apos;image
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}

export default ResultPage
