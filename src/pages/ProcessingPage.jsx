import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { BrainCircuit, Maximize2, Layers, Target, CheckCircle2, Sparkles, Clock, AlertCircle } from 'lucide-react'
import useStore from '../store/useStore'

const STEPS = [
    { id: 'sam',     label: 'Segmentation SAM',           icon: Layers,       hint: 'Chargement du modèle ViT-H...' },
    { id: 'depth',   label: 'Carte de profondeur',         icon: Maximize2,    hint: 'Depth Anything en cours...' },
    { id: 'fuse',    label: 'Fusion SAM + Depth',          icon: BrainCircuit, hint: 'Association zones + profondeur...' },
    { id: 'climate', label: 'Parametres botaniques',       icon: Target,       hint: 'Optimisation géoclimatique...' },
]

// Intervalle de polling (ms)
const POLL_INTERVAL = 4000

const ProcessingPage = () => {
    const navigate = useNavigate()
    const { previewUrl, preprocessData, projectContext, setAnalysisResult, setProjectId } = useStore()

    const [currentStep, setCurrentStep] = useState(0)
    const [progress, setProgress] = useState(0)
    const [elapsed, setElapsed] = useState(0)
    const [errorMsg, setErrorMsg] = useState(null)
    const [jobStatus, setJobStatus] = useState('idle') // idle | starting | running | done | error

    const pollRef = useRef(null)
    const progressRef = useRef(null)
    const elapsedRef = useRef(null)
    const jobIdRef = useRef(null)

    useEffect(() => {
        if (!previewUrl) { navigate('/upload'); return }

        const API_URL = import.meta.env.VITE_CLIMATE_API_URL || 'http://localhost:3001'

        const cleanup = () => {
            clearInterval(pollRef.current)
            clearInterval(progressRef.current)
            clearInterval(elapsedRef.current)
        }

        const startAnalysis = async () => {
            setJobStatus('starting')

            // 1. Envoyer le manifeste (fire-and-forget)
            const manifest = useStore.getState().getProjectManifest()
            fetch(`${API_URL}/api/project/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(manifest)
            }).catch(err => console.warn('[IA BRIDGE] Erreur manifeste:', err))

            // 2. Lancer l'analyse SAM+Depth (non-bloquant → renvoie job_id)
            if (!preprocessData?.preprocessed_json) {
                // Mode démo sans preprocess
                console.warn('[PIPELINE] Pas de preprocessed_json — mode démo')
                setJobStatus('done')
                setProgress(100)
                setCurrentStep(STEPS.length - 1)
                setTimeout(() => {
                    setProjectId('demo-' + Math.random().toString(36).slice(2, 11))
                    navigate('/result/demo')
                }, 1000)
                return
            }

            try {
                const resp = await fetch(`${API_URL}/api/project/analyze`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ preprocess_json: preprocessData.preprocessed_json, force: false })
                })
                if (!resp.ok) {
                    const e = await resp.json().catch(() => ({}))
                    throw new Error(e.error || `HTTP ${resp.status}`)
                }
                const { job_id } = await resp.json()
                jobIdRef.current = job_id
                setJobStatus('running')
                console.log(`[PIPELINE] Job démarré: ${job_id}`)

                // Progression visuelle simulée (avance jusqu'à 88% max)
                progressRef.current = setInterval(() => {
                    setProgress(p => {
                        // Avancement selon l'étape
                        const maxPerStep = [22, 50, 78, 88]
                        const max = maxPerStep[currentStep] ?? 88
                        return p < max ? p + 0.8 : p
                    })
                    setCurrentStep(s => {
                        // Avancer les étapes simulées toutes les ~20s
                        return s
                    })
                }, 400)

                // Avancer les étapes visuelles en fonction du temps écoulé
                const stepTimings = [0, 90, 180, 270] // secondes
                elapsedRef.current = setInterval(() => {
                    setElapsed(e => {
                        const next = e + 1
                        const stepIdx = stepTimings.findLastIndex(t => next >= t)
                        setCurrentStep(Math.min(stepIdx, STEPS.length - 1))
                        return next
                    })
                }, 1000)

                // Polling du statut
                pollRef.current = setInterval(async () => {
                    try {
                        const statusResp = await fetch(`${API_URL}/api/project/analyze-status/${job_id}`)
                        if (!statusResp.ok) return
                        const job = await statusResp.json()

                        if (job.status === 'done') {
                            cleanup()
                            setProgress(100)
                            setCurrentStep(STEPS.length - 1)
                            setJobStatus('done')
                            const isZoneOnly = job.result?.zone_only === true
                            console.log(`[PIPELINE] Analyse terminée (${isZoneOnly ? 'zone-only' : 'SAM+Depth'}):`, job.result)

                            setAnalysisResult({
                                ...job.result,
                                summary: isZoneOnly
                                    ? `Mode zone-only (SAM timeout après ${job.elapsed_s}s)`
                                    : `Analyse terminée en ${job.elapsed_s}s`
                            })
                            setProjectId(preprocessData?.image_id || 'demo-' + Math.random().toString(36).slice(2, 11))
                            setTimeout(() => navigate(`/result/${preprocessData?.image_id || 'demo'}`), 800)

                        } else if (job.status === 'error') {
                            cleanup()
                            setJobStatus('error')
                            setErrorMsg(job.error || "Erreur inconnue lors de l'analyse")
                            console.error('[PIPELINE] Erreur:', job.error)
                        }
                        // Si 'running' on continue de poller
                    } catch (pollErr) {
                        console.warn('[PIPELINE] Erreur polling:', pollErr.message)
                    }
                }, POLL_INTERVAL)

            } catch (err) {
                cleanup()
                setJobStatus('error')
                setErrorMsg(err.message)
                console.error('[PIPELINE] Erreur démarrage:', err)
            }
        }

        startAnalysis()
        return cleanup
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    const globalProgress = Math.min(
        Math.round(((currentStep * 25) + (progress % 25))),
        progress >= 100 ? 100 : 99
    )

    return (
        <div className="max-w-5xl mx-auto space-y-12 animate-in fade-in duration-700">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">

                {/* Image + animation */}
                <div className="relative rounded-[2.5rem] overflow-hidden bg-white shadow-2xl h-[550px] border border-gray-100">
                    {previewUrl && (
                        <img
                            src={previewUrl}
                            className="w-full h-full object-cover opacity-40 transition-all duration-1000 grayscale-[0.5]"
                            alt="Analysis context"
                        />
                    )}

                    <div className="absolute inset-0">
                        {currentStep === 0 && <div className="scan-line !bg-[var(--color-nature)] !shadow-[var(--color-nature)]" />}
                        {currentStep >= 1 && currentStep !== 2 && (
                            <div className="absolute inset-0 bg-[var(--color-nature)]/5 backdrop-blur-[2px] animate-pulse flex items-center justify-center">
                                <div className="px-6 py-3 bg-white/90 shadow-xl rounded-2xl text-[var(--color-nature)] font-bold tracking-widest text-xs uppercase border border-[var(--color-nature)]/20">
                                    {STEPS[currentStep]?.hint || 'En cours...'}
                                </div>
                            </div>
                        )}
                        {currentStep === 2 && (
                            <div className="absolute inset-0 bg-white/40 backdrop-blur-[1px] flex flex-col items-center justify-center space-y-4 animate-in fade-in duration-500">
                                <div className="p-8 bg-white/90 backdrop-blur-xl shadow-2xl rounded-[2rem] text-center space-y-3 max-w-[85%] border border-gray-100">
                                    <div className="w-12 h-12 bg-[var(--color-nature)]/10 rounded-2xl flex items-center justify-center text-[var(--color-nature)] mx-auto mb-2">
                                        <BrainCircuit size={24} />
                                    </div>
                                    <p className="text-[var(--color-nature)] font-bold tracking-[0.2em] text-[10px] uppercase">Atlas Climatique Précis</p>
                                    <p className="text-base font-bold text-[var(--color-structure)]">{projectContext?.location?.label || 'Localisation...'}</p>
                                    <div className="flex flex-wrap gap-2 justify-center mt-4">
                                        <div className="text-[10px] bg-white px-3 py-2 rounded-xl border border-gray-100 shadow-sm font-bold">🌡️ {projectContext?.climate_profile?.current?.temp_c}°C</div>
                                        <div className="text-[10px] bg-white px-3 py-2 rounded-xl border border-gray-100 shadow-sm font-bold">❄️ Min: {projectContext?.annual_profile?.summary?.temp_min_record_c}°C</div>
                                        <div className="text-[10px] bg-white px-3 py-2 rounded-xl border border-gray-100 shadow-sm font-bold">🌿 {projectContext?.annual_profile?.summary?.hardiness_zone}</div>
                                    </div>
                                </div>
                            </div>
                        )}
                        {currentStep === 3 && (
                            <div className="absolute inset-0 overflow-hidden">
                                <div className="absolute top-1/4 left-1/4 w-32 h-32 border-2 border-[var(--color-nature)] rounded-full animate-ping opacity-20" />
                                <div className="absolute bottom-1/3 right-1/4 w-40 h-40 border-2 border-[var(--color-action)] rounded-full animate-ping opacity-20" />
                            </div>
                        )}
                    </div>

                    <div className="absolute bottom-0 inset-x-0 p-10 bg-gradient-to-t from-white via-white/80 to-transparent">
                        <div className="flex items-center gap-5">
                            <div className="w-14 h-14 rounded-2xl bg-[var(--color-nature)]/10 flex items-center justify-center text-[var(--color-nature)] animate-pulse">
                                <Sparkles size={32} />
                            </div>
                            <div>
                                <p className="text-[10px] font-bold text-[var(--color-nature)] uppercase tracking-[0.3em]">Pipeline SAM + Depth</p>
                                <p className="text-xl font-bold text-[var(--color-structure)]">
                                    {jobStatus === 'error' ? 'Erreur détectée' : 'Analyse en cours...'}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Panneau droit */}
                <div className="space-y-8">

                    {/* Barre de progression globale */}
                    <div className="space-y-3">
                        <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest mb-1">
                            <span className="text-gray-400">Progression SAM+Depth</span>
                            <div className="flex items-center gap-2">
                                {jobStatus === 'running' && (
                                    <span className="flex items-center gap-1 text-gray-400">
                                        <Clock size={10} />
                                        {elapsed}s
                                    </span>
                                )}
                                <span className="text-[var(--color-nature)]">{globalProgress}%</span>
                            </div>
                        </div>
                        <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden shadow-inner">
                            <div
                                className="h-full bg-[var(--color-nature)] transition-all duration-1000 rounded-full"
                                style={{ width: `${globalProgress}%` }}
                            />
                        </div>
                        {jobStatus === 'running' && (
                            <p className="text-[10px] text-gray-400 text-center font-bold">
                                ⏳ SAM charge le modèle ViT-H — cela peut prendre 3 à 10 minutes
                            </p>
                        )}
                    </div>

                    {/* Erreur */}
                    {jobStatus === 'error' && errorMsg && (
                        <div className="p-5 bg-red-50 border border-red-100 rounded-2xl space-y-3">
                            <div className="flex items-start gap-3">
                                <AlertCircle size={18} className="text-red-500 flex-shrink-0 mt-0.5" />
                                <p className="text-sm font-bold text-red-600">Erreur lors de l'analyse SAM</p>
                            </div>
                            <pre className="text-[10px] text-red-400 whitespace-pre-wrap break-all font-mono bg-red-100/50 rounded-lg p-3 max-h-40 overflow-y-auto">
                                {errorMsg}
                            </pre>
                            <div className="flex gap-3 flex-wrap">
                                <button
                                    onClick={() => {
                                        // Mode dégradé : passer directement à la génération sans SAM
                                        const { setAnalysisResult, setProjectId, preprocessData } = useStore.getState()
                                        setAnalysisResult({ mode: 'zone_only', summary: 'Mode sans SAM (analyse échouée)' })
                                        setProjectId(preprocessData?.image_id || 'demo-' + Math.random().toString(36).slice(2, 11))
                                        navigate(`/result/${preprocessData?.image_id || 'demo'}`)
                                    }}
                                    className="flex-1 py-2.5 px-4 rounded-xl bg-[var(--color-nature)] text-white text-xs font-black uppercase tracking-widest hover:opacity-90"
                                >
                                    Générer sans SAM →
                                </button>
                                <button
                                    onClick={() => navigate('/upload')}
                                    className="py-2.5 px-4 rounded-xl bg-white border border-red-200 text-red-500 text-xs font-bold"
                                >
                                    Retour
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Étapes */}
                    <div className="space-y-4">
                        {STEPS.map((step, idx) => (
                            <div
                                key={step.id}
                                className={`flex items-center gap-5 p-5 rounded-[1.5rem] border transition-all duration-500
                                    ${idx === currentStep ? 'bg-white border-gray-100 shadow-xl' :
                                    idx < currentStep ? 'bg-[var(--color-nature)]/5 border-transparent text-[var(--color-nature)]' :
                                    'bg-transparent border-transparent text-gray-300'}`}
                            >
                                <div className={`w-12 h-12 rounded-xl flex items-center justify-center transition-colors duration-500
                                    ${idx === currentStep ? 'bg-[var(--color-nature)] text-white shadow-lg shadow-[#7b9872]/30' :
                                    idx < currentStep ? 'bg-[var(--color-nature)]/10 text-[var(--color-nature)]' :
                                    'bg-gray-50 text-gray-300'}`}
                                >
                                    {idx < currentStep ? <CheckCircle2 size={24} /> : <step.icon size={24} />}
                                </div>
                                <div className="flex-1">
                                    <p className={`font-bold transition-colors ${idx === currentStep ? 'text-[var(--color-structure)]' : ''}`}>
                                        {step.label}
                                    </p>
                                    <p className="text-[10px] uppercase tracking-widest font-bold opacity-50">
                                        {idx === currentStep ? step.hint :
                                            idx < currentStep ? 'Terminé ✓' : 'En attente'}
                                    </p>
                                </div>
                                {idx === currentStep && jobStatus === 'running' && (
                                    <div className="w-16 h-1 bg-gray-100 rounded-full overflow-hidden">
                                        <div className="h-full bg-[var(--color-nature)] animate-pulse" style={{ width: `${progress % 100}%` }} />
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>

                    <div className="p-8 rounded-[2rem] bg-stone-50 border border-stone-100 relative overflow-hidden group">
                        <div className="absolute top-0 right-0 w-24 h-24 bg-[var(--color-nature)]/5 rounded-full -mr-12 -mt-12 transition-transform group-hover:scale-110" />
                        <p className="text-sm text-[var(--color-structure)] font-medium italic relative z-10 leading-relaxed">
                            &ldquo;Chaque végétal proposé est sélectionné selon sa résistance hydrique et son harmonie chromatique avec l&apos;existant.&rdquo;
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default ProcessingPage
