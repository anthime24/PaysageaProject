"""
Garden AI — Interface Streamlit v3 (sans QCM).
jardin_complet.json est chargé automatiquement au démarrage.
"""
from __future__ import annotations

import colorsys
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_generation.scene_generator_v2 import dispatch_generation
from image_generation.bfl_provider import has_bfl_key
from image_generation.utils_rag import load_rag

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR    = PROJECT_ROOT / "data"
UI_INPUTS   = OUTPUTS_DIR / "_ui_inputs"
for d in [OUTPUTS_DIR, UI_INPUTS]:
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_IMAGE   = DATA_DIR / "garden.jpg"
JARDIN_COMPLET = DATA_DIR / "jardin_complet.json"
RAG_PATH       = OUTPUTS_DIR / "current_rag_selection.json"


def _shift_mask_binary(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Décale un masque binaire sans wrap-around."""
    h, w = mask.shape[:2]
    shifted = np.zeros_like(mask, dtype=np.uint8)

    src_x1 = max(0, -dx)
    src_y1 = max(0, -dy)
    src_x2 = min(w, w - dx) if dx >= 0 else w
    src_y2 = min(h, h - dy) if dy >= 0 else h

    dst_x1 = max(0, dx)
    dst_y1 = max(0, dy)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        shifted[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return shifted


def _overlay_mask_on_image(
    image_path: Path,
    mask_path_or_pil: Path | Image.Image,
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay rouge : masque binaire redimensionné en NEAREST (bords nets, pas de flou artificiel)."""
    img = Image.open(image_path).convert("RGB")
    if isinstance(mask_path_or_pil, Path):
        m = Image.open(mask_path_or_pil).convert("L")
    else:
        m = mask_path_or_pil.convert("L")
    if m.size != img.size:
        m = m.resize(img.size, Image.Resampling.NEAREST)
    img_arr = np.array(img, dtype=np.uint8)
    mask = np.array(m) >= 128

    out = img_arr.copy()
    red = np.array([255, 0, 0], dtype=np.uint8)
    out[mask] = (alpha * red + (1.0 - alpha) * out[mask]).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _overlay_mask_from_array(image_path: Path, mask_l: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay depuis un masque L déjà chargé (uint8)."""
    return _overlay_mask_on_image(image_path, Image.fromarray(mask_l, mode="L"), alpha=alpha)


def _ensure_mask_hw(mask: np.ndarray, iw: int, ih: int) -> np.ndarray:
    if mask.shape[0] == ih and mask.shape[1] == iw:
        return mask
    return np.array(
        Image.fromarray(mask, mode="L").resize((iw, ih), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )


def _bbox_from_mask(m: np.ndarray) -> list[int]:
    ys, xs = np.where(m >= 128)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _centroid_from_mask(m: np.ndarray) -> list[float]:
    ys, xs = np.where(m >= 128)
    if len(xs) == 0:
        return [0.0, 0.0]
    return [float(xs.mean()), float(ys.mean())]


def _compose_colored_masks_preview(
    image_path: Path,
    buffers: dict[str, np.ndarray],
    iw: int,
    ih: int,
    active_name: str | None,
) -> Image.Image:
    """Toutes les zones masque en couleurs distinctes ; la zone active est plus opaque."""
    base = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
    names = sorted(buffers.keys())
    n = max(len(names), 1)
    for i, name in enumerate(names):
        m = _ensure_mask_hw(buffers[name], iw, ih)
        binm = m >= 128
        if not np.any(binm):
            continue
        hue = (i + 0.5) / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.82, 0.98)
        color = np.array([r * 255, g * 255, b * 255], dtype=np.float32)
        alpha = 0.5 if name != active_name else 0.68
        base[binm] = (1.0 - alpha) * base[binm] + alpha * color
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def _pick_mask_at(
    fx: int,
    fy: int,
    sorted_names: list[str],
    buffers: dict[str, np.ndarray],
    iw: int,
    ih: int,
) -> str | None:
    """Choisit le masque « au-dessus » : ordre inverse (dernière étape = dernier plan)."""
    for name in reversed(sorted_names):
        m = _ensure_mask_hw(buffers[name], iw, ih)
        if 0 <= fy < m.shape[0] and 0 <= fx < m.shape[1] and m[fy, fx] >= 128:
            return name
    return None


def _export_masks_edited_bundle(
    masks_dir: Path,
    buffers: dict[str, np.ndarray],
    ref_image: Path,
    scene_sequential_path: Path,
) -> tuple[Path, bytes]:
    """Écrit scene_masks_edited.json + met à jour les bbox dans scene_sequential.json si présent."""
    manifest = {
        "exported_at": datetime.now().isoformat(),
        "reference_image": str(ref_image.resolve()),
        "note": "Masques édités dans l’UI ; chaque fichier PNG est sauvegardé dans outputs/masks/.",
        "masks": [],
    }
    for name in sorted(buffers.keys()):
        m = buffers[name]
        manifest["masks"].append(
            {
                "file": name,
                "path": str((masks_dir / name).resolve()),
                "bbox": _bbox_from_mask(m),
                "centroid": _centroid_from_mask(m),
            }
        )

    if scene_sequential_path.exists():
        try:
            data = json.loads(scene_sequential_path.read_text(encoding="utf-8"))
            for step in data.get("steps", []):
                mp = step.get("mask_path") or ""
                fname = Path(mp).name
                if fname in buffers:
                    step["bbox"] = _bbox_from_mask(buffers[fname])
            scene_sequential_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            manifest["scene_sequential_updated"] = str(scene_sequential_path.resolve())
        except Exception:
            pass

    out_path = OUTPUTS_DIR / "scene_masks_edited.json"
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    out_path.write_text(text, encoding="utf-8")
    return out_path, text.encode("utf-8")


def _list_plant_only_mask_files(masks_dir: Path) -> list[Path]:
    """
    Uniquement les masques « une plante » (étapes séquentielles), pas plantable_mask_bin,
    ni combinaisons / debug. Ordre : scene_sequential.json (steps) puis fallback plant_plant_*.png.
    """
    masks_dir = masks_dir.resolve()
    seq_path = OUTPUTS_DIR / "scene_sequential.json"
    if seq_path.exists():
        try:
            data = json.loads(seq_path.read_text(encoding="utf-8"))
            out: list[Path] = []
            seen: set[str] = set()
            for step in data.get("steps", []):
                mp = (step.get("mask_path") or "").strip()
                if not mp:
                    continue
                name = Path(mp).name
                if not name.lower().endswith(".png"):
                    continue
                cand = masks_dir / name
                if not cand.is_file():
                    continue
                if name in seen:
                    continue
                seen.add(name)
                out.append(cand)
            if out:
                return sorted(out, key=lambda x: x.name)
        except Exception:
            pass

    plant_plant = sorted(masks_dir.glob("plant_plant_*.png"))
    if plant_plant:
        return plant_plant

    bad_sub = ("plantable", "combined", "debug", "user_zone", "sam", "mask_bin")
    fallback: list[Path] = []
    for p in sorted(masks_dir.glob("plant_*.png")):
        ln = p.name.lower()
        if any(s in ln for s in bad_sub):
            continue
        fallback.append(p)
    return fallback


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html,body,[class*="st-"]{font-family:'Inter',sans-serif;}
    .stApp{background:linear-gradient(180deg,#0f172a 0%,#020617 100%);color:#f8fafc;}
    .main-title{font-size:2.8rem;font-weight:800;
        background:linear-gradient(90deg,#4ade80,#2dd4bf);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        text-align:center;margin-bottom:.2rem;}
    .sub-title{font-size:1.1rem;color:#94a3b8;text-align:center;margin-bottom:2rem;}
    .glass{background:rgba(30,41,59,.5);backdrop-filter:blur(12px);
        border:1px solid rgba(255,255,255,.1);border-radius:16px;
        padding:1.5rem;margin-bottom:1.5rem;}
    [data-testid="stSidebar"]{background-color:#0f172a;
        border-right:1px solid rgba(255,255,255,.1);}
    .stButton>button{border-radius:12px;font-weight:600;transition:all .3s;}
    .stButton>button[kind="primary"]{
        background:linear-gradient(135deg,#22c55e,#10b981);
        border:none;box-shadow:0 4px 12px rgba(34,197,94,.3);}
    .stButton>button[kind="secondary"]{
        background:#1e293b !important;color:#f1f5f9 !important;
        border:1px solid rgba(255,255,255,.25) !important;}
    .badge{display:inline-block;background:rgba(34,197,94,.1);color:#4ade80;
        border:1px solid rgba(34,197,94,.3);padding:3px 10px;
        border-radius:20px;font-size:.82rem;margin:3px;}
    .img-box{border-radius:16px;overflow:hidden;
        border:2px solid rgba(255,255,255,.1);
        box-shadow:0 20px 50px rgba(0,0,0,.5);}
    #MainMenu{visibility:hidden;}footer{visibility:hidden;}
    </style>""", unsafe_allow_html=True)


def _auto_load_jardin():
    """Charge jardin_complet.json automatiquement au premier chargement."""
    if st.session_state.get("rag_ready"):
        return
    if not JARDIN_COMPLET.exists():
        st.error(f"❌ Fichier introuvable : {JARDIN_COMPLET}")
        return
    try:
        _, plants = load_rag(JARDIN_COMPLET)
        shutil.copy2(JARDIN_COMPLET, RAG_PATH)
        st.session_state["rag_ready"]       = True
        st.session_state["rag_path"]        = str(RAG_PATH)
        st.session_state["selected_plants"] = [p["name"] for p in plants]
    except Exception as e:
        st.error(f"Erreur chargement jardin_complet.json : {e}")


def main():
    st.set_page_config(page_title="Garden AI", layout="wide")
    inject_css()

    st.markdown('<h1 class="main-title">Garden AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Génération de jardin plante par plante — Flux Fill</p>',
        unsafe_allow_html=True)

    # Chargement auto jardin_complet.json dès le démarrage
    _auto_load_jardin()

    # ── SIDEBAR ─────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Configuration")

        st.subheader("📸 Photo du jardin")
        img_src = st.radio("", ["Image par défaut", "Uploader une photo"],
                           label_visibility="collapsed")
        if img_src == "Image par défaut":
            image_path = SAMPLE_IMAGE
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
        else:
            f = st.file_uploader("Choisir...", type=["jpg","jpeg","png"])
            if f:
                # Conserver une extension cohérente avec le contenu uploadé.
                # Évite notamment les PNG RGBA sauvés en .jpg (PIL/SAM plante ensuite).
                ext = (Path(f.name).suffix or "").lower()
                if ext not in [".jpg", ".jpeg", ".png"]:
                    ext = ".png"
                image_path = UI_INPUTS / f"uploaded_garden{ext}"
                image_path.write_bytes(f.read())
                st.image(str(image_path), use_container_width=True)
            else:
                image_path = None

        st.divider()

        mode_label = st.selectbox("Mode de génération", [
            "🌱 Séquentiel (plante par plante)",
            "⚡ Global (un seul appel BFL)",
        ])
        st.session_state["mode"] = (
            "sequential" if "Séquentiel" in mode_label else "global"
        )
        max_plants  = st.slider("Nb max de plantes", 1, 10, 6)
        time_of_day = st.select_slider("Heure", options=["Jour","Nuit"])
        night_int   = (
            st.slider("Intensité nuit", 0.0, 1.0, 0.5)
            if time_of_day == "Nuit" else 0.5
        )

        st.divider()
        skip_zone_ui = st.checkbox(
            "🧪 Skip UI zone-selection (OpenCV)",
            value=False,
            help="Pour les tests Streamlit : on évite les fenêtres OpenCV et on fusionne avec le user_zone.json existant.",
        )
        use_backend_zone_mask = st.checkbox(
            "✅ Utiliser la zone plantable (user_zone_mask.png) pour placer les plantes",
            value=True,
            help="Si activé, la génération plante-par-plante est contrainte à la zone sélectionnée.",
        )

        st.divider()
        st.subheader("📂 Changer les plantes")
        st.caption("jardin_complet.json chargé auto. Upload un autre JSON si besoin.")
        up_rag = st.file_uploader("Autre JSON RAG", type=["json"])
        if up_rag:
            try:
                tmp = Path(tempfile.mktemp(suffix=".json"))
                tmp.write_bytes(up_rag.read())
                _, plants = load_rag(tmp)
                shutil.copy2(tmp, RAG_PATH)
                st.session_state["rag_ready"]       = True
                st.session_state["rag_path"]        = str(RAG_PATH)
                st.session_state["selected_plants"] = [p["name"] for p in plants]
                st.success(f"✅ {len(plants)} plantes depuis {up_rag.name}")
            except Exception as e:
                st.error(f"Erreur : {e}")

        st.divider()
        if not has_bfl_key():
            st.warning("⚠️ BFL_API_KEY manquante → mode MOCK actif")
        else:
            st.success("🔑 API BFL connectée")

    # ── MAIN ────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1.3], gap="large")

    with col_left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🌿 Plantes chargées")

        plants_list = st.session_state.get("selected_plants", [])
        if plants_list:
            st.markdown(
                "".join(f'<span class="badge">{p}</span>' for p in plants_list),
                unsafe_allow_html=True)
            st.caption(f"{len(plants_list)} plantes — jardin_complet.json")
        else:
            st.info("Chargement...")

        st.markdown('</div>', unsafe_allow_html=True)

        # Bouton Générer
        can_go = bool(
            image_path
            and st.session_state.get("rag_ready")
            and plants_list
        )
        if not image_path:
            st.warning("Sélectionne une image dans la sidebar.")

        if st.button("🚀 GÉNÉRER MON JARDIN", type="primary",
                     use_container_width=True, disabled=not can_go):
            prog  = st.progress(0)
            msg   = st.empty()
            mode  = st.session_state.get("mode", "sequential")
            msg.text(f"⏳ Génération {'séquentielle' if mode=='sequential' else 'globale'}...")
            prog.progress(10)
            try:
                zone_mask_path = PROJECT_ROOT.parent / "zone-selection" / "outputs" / "user_zone_mask.png"
                if use_backend_zone_mask and not zone_mask_path.exists():
                    st.error("user_zone_mask.png introuvable. Lance d'abord Phase1-B.")
                    st.stop()

                scene = dispatch_generation(
                    image_path            = image_path,
                    rag_json_path         = st.session_state["rag_path"],
                    outputs_dir           = OUTPUTS_DIR,
                    mode                  = mode,
                    time_of_day           = "night" if time_of_day=="Nuit" else "day",
                    night_light_intensity = night_int,
                    max_plants            = max_plants,
                    external_plantable_mask_path=(
                        str(zone_mask_path)
                        if use_backend_zone_mask else None
                    ),
                    debug                 = True,
                )
                st.session_state["steps"] = scene.get("steps", [])
                prog.progress(100)
                msg.text("✅ Génération terminée !")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erreur : {e}")
                st.exception(e)

        st.divider()
        st.subheader("🧪 Test backend Phase1-B (local)")
        pipeline_btn = st.button(
            "Lancer Phase1-B (SAM+Depth+Zone) => final_scene_input.json",
            type="secondary",
            use_container_width=True,
            disabled=not image_path,
        )
        if pipeline_btn:
            import subprocess

            pays_root = PROJECT_ROOT.parent  # .../Paysagea
            orchestrator = pays_root / "run_full_phase1_b_sam_depth_zone.py"
            image_abs = Path(image_path).resolve() if image_path else None
            if not image_abs or not image_abs.exists():
                st.error("Image introuvable.")
                st.stop()

            cmd = [
                str(pays_root / ".venv" / "Scripts" / "python.exe")
                if (pays_root / ".venv" / "Scripts" / "python.exe").exists()
                else sys.executable,
                str(orchestrator),
                "--image",
                str(image_abs),
            ]

            # Utilise directement le checkpoint déjà présent dans le repo (si dispo)
            ckpt = pays_root / "SAM_for_paysagea" / "sam_vit_b_01ec64.pth"
            if ckpt.exists():
                cmd += ["--sam-checkpoint", str(ckpt)]

            if skip_zone_ui:
                cmd += ["--skip-zone-selection"]
                default_user_zone = pays_root / "zone-selection" / "outputs" / "user_zone.json"
                cmd += ["--user-zone-json", str(default_user_zone)]

            with st.spinner("Pipeline Phase1-B en cours..."):
                try:
                    # Windows : sans encoding explicite, cp1252 casse sur la sortie UTF-8 du pipeline.
                    proc = subprocess.run(
                        cmd,
                        cwd=str(pays_root),
                        check=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    st.success("Phase1-B terminée.")
                    # st.write(proc.stdout[-2000:])  # optionnel debug
                except subprocess.CalledProcessError as e:
                    st.error("Erreur pendant Phase1-B.")
                    if e.stdout:
                        st.code(e.stdout[-2000:])
                    if e.stderr:
                        st.code(e.stderr[-2000:])
                    st.stop()

    with col_right:
        final = OUTPUTS_DIR / (
            "final_garden_night.png" if time_of_day=="Nuit" else "final_garden.png"
        )
        tab_r, tab_ab, tab_steps, tab_mask, tab_edit = st.tabs(
            ["✨ Résultat","🔍 Avant/Après","🌿 Étapes","🗺️ Masque","🛠️ Éditer Masques"])

        with tab_r:
            if final.exists():
                st.markdown('<div class="img-box">', unsafe_allow_html=True)
                st.image(str(final), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                with open(final,"rb") as fh:
                    st.download_button("📥 Télécharger", fh,
                        f"jardin_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                        "image/png")
            else:
                st.info("Le résultat apparaîtra ici après la génération.")
                if image_path and Path(str(image_path)).exists():
                    st.image(str(image_path), caption="Jardin actuel",
                             use_container_width=True)

        with tab_ab:
            if final.exists() and image_path and Path(str(image_path)).exists():
                c1,c2 = st.columns(2)
                c1.write("**Avant**"); c1.image(str(image_path), use_container_width=True)
                c2.write("**Après**"); c2.image(str(final), use_container_width=True)
            else:
                st.info("Générez d'abord une image.")

        with tab_steps:
            steps = st.session_state.get("steps", [])
            if not steps:
                seq_p = OUTPUTS_DIR / "scene_sequential.json"
                if seq_p.exists():
                    steps = json.load(open(seq_p)).get("steps", [])
            if steps:
                st.write(f"**{len(steps)} plantes générées :**")
                for s in steps:
                    with st.expander(
                        f"🌱 {s['index']+1}. {s['name']} ({s.get('zone_hint','?')})"
                    ):
                        c1,c2 = st.columns(2)
                        mp,cp = s.get("mask_path",""), s.get("composite_path","")
                        if mp and Path(mp).exists():
                            c1.write("Masque"); c1.image(mp, use_container_width=True)
                        if cp and Path(cp).exists():
                            c2.write("Résultat"); c2.image(cp, use_container_width=True)
                        m1,m2,m3 = st.columns(3)
                        m1.metric("Zone",     s.get("zone_hint","—"))
                        m2.metric("Couleur",  s.get("color","—") or "—")
                        m3.metric("Strength", f"{s.get("strength",0):.2f}")
                        st.caption(f"Prompt : {s.get("prompt","")[:250]}...")
            else:
                st.info("Les étapes apparaîtront après une génération séquentielle.")

        with tab_mask:
            dbg = OUTPUTS_DIR / "mask_debug.png"
            if dbg.exists():
                st.image(str(dbg), use_container_width=True)
            else:
                st.info("Le masque apparaîtra après la génération.")

        with tab_edit:
            with st.expander("ℹ️ Photos des plantes dans le JSON — pourquoi ce n’est pas ce que tu vois ici", expanded=False):
                st.markdown(
                    """
Les champs **`image_path`** (ex. `data/rag_images/plant_01.jpg`) dans `jardin_complet.json` servent de **références**
pour les prompts / RAG, pas comme image affichée dans cet éditeur.

**Éditeur :** seuls les masques **par plante** (`plant_plant_XX.png` listés dans `scene_sequential.json`) sont modifiables — pas le masque global « zone plantable » (`plantable_mask_bin.png`, etc.).

Ici tu vois uniquement les **masques** PNG générés (`outputs/masks/…`) : ce sont des **zones blanches/noires**
à la résolution de ton jardin. Si la zone est petite, elle paraît « en gros pixels » ou floue une fois colorée :  
c’est normal ; pour des bords plus nets il faudrait des masques plus grands ou générés plus finement côté pipeline.

**Remettre / changer les images du JSON :** place tes fichiers (ex. dans `garden_ia_3/data/rag_images/`) et mets dans chaque plante :
`"image_path": "data/rag_images/plant_01.jpg"` (chemin relatif au dossier `garden_ia_3` ou absolu). Puis recharge le JSON dans la barre latérale.
                    """
                )

            st.caption(
                "**Clic court** sur une zone colorée → sélectionne ce masque. **Clic + glisser** → déplace le masque sélectionné. "
                "**Valider** enregistre les PNG + un JSON récapitulatif."
            )
            st.caption(
                "**Uniquement les masques « une plante »** (fichiers `plant_plant_XX.png` du mode séquentiel) — "
                "pas le masque global de la zone plantable (`plantable_*`, etc.)."
            )
            masks_dir = OUTPUTS_DIR / "masks"
            mask_files = _list_plant_only_mask_files(masks_dir)
            if not mask_files:
                st.info(
                    "Aucun masque **plante** trouvé (attendu : étapes dans `scene_sequential.json` "
                    "ou fichiers `plant_plant_*.png`). Lance une génération **séquentielle** avec zone plantable."
                )
            else:
                image_for_overlay = final if final.exists() else (Path(image_path) if image_path else None)
                sig = tuple(p.name for p in mask_files)
                if st.session_state.get("mask_files_sig") != sig:
                    st.session_state["mask_files_sig"] = sig
                    st.session_state["mask_buffers"] = {
                        p.name: np.array(Image.open(p).convert("L")) for p in mask_files
                    }
                    st.session_state["mask_editor_active"] = None
                    st.session_state["mask_editor_last_unix"] = None

                if "mask_buffers" not in st.session_state:
                    st.session_state["mask_buffers"] = {
                        p.name: np.array(Image.open(p).convert("L")) for p in mask_files
                    }
                buffers: dict[str, np.ndarray] = st.session_state["mask_buffers"]
                active = st.session_state.get("mask_editor_active")

                if image_for_overlay and Path(image_for_overlay).exists():
                    with Image.open(image_for_overlay) as im:
                        iw, ih = im.size
                    for k in list(buffers.keys()):
                        buffers[k] = _ensure_mask_hw(buffers[k], iw, ih)

                    preview_pil = _compose_colored_masks_preview(
                        Path(image_for_overlay), buffers, iw, ih, active
                    )

                    # Largeur adaptée au panneau (ratio conservé) — évite l’effet « zoom »
                    # d’une image forcée à 1600px de large.
                    value = streamlit_image_coordinates(
                        preview_pil,
                        key="mask_editor_canvas_all",
                        use_column_width=True,
                        click_and_drag=True,
                        cursor="grab",
                        png_compression_level=0,
                    )

                    names_sorted = sorted(buffers.keys())
                    ts_key = "mask_editor_last_unix"
                    if value and "x1" in value:
                        ev_ts = value.get("unix_time")
                        if ev_ts is not None and ev_ts != st.session_state.get(ts_key):
                            st.session_state[ts_key] = ev_ts
                            disp_w = float(value.get("width") or iw)
                            disp_h = float(value.get("height") or ih)
                            x1d, y1d = float(value["x1"]), float(value["y1"])
                            dx_d = float(value["x2"]) - x1d
                            dy_d = float(value["y2"]) - float(value["y1"])
                            dist = (dx_d * dx_d + dy_d * dy_d) ** 0.5
                            fx = int(np.clip(round(x1d * iw / disp_w), 0, iw - 1))
                            fy = int(np.clip(round(y1d * ih / disp_h), 0, ih - 1))

                            if dist < 6.0:
                                picked = _pick_mask_at(fx, fy, names_sorted, buffers, iw, ih)
                                st.session_state["mask_editor_active"] = picked
                                if picked:
                                    st.session_state["mask_feedback"] = f"Masque actif : **{picked}**"
                                else:
                                    st.session_state["mask_feedback"] = (
                                        "Aucun masque sous ce clic — essaie une zone colorée."
                                    )
                                st.rerun()
                            else:
                                act = st.session_state.get("mask_editor_active")
                                if not act:
                                    st.session_state["mask_feedback"] = (
                                        "Sélectionne d’abord une zone (clic court sur une couleur)."
                                    )
                                    st.rerun()
                                else:
                                    dx_full = int(round(dx_d * (iw / disp_w)))
                                    dy_full = int(round(dy_d * (ih / disp_h)))
                                    b = buffers[act]
                                    buffers[act] = _shift_mask_binary(
                                        (b >= 128).astype(np.uint8) * 255, dx=dx_full, dy=dy_full
                                    )
                                    st.session_state["mask_feedback"] = None
                                    st.rerun()

                    fb = st.session_state.pop("mask_feedback", None)
                    if fb:
                        st.markdown(fb)

                    act = st.session_state.get("mask_editor_active")
                    st.caption(
                        f"Image source : **{iw}×{ih}px** — affichage **à l’échelle du panneau** (pas de zoom forcé). "
                        f"Masque actif : **{act or '—'}**"
                    )
                else:
                    st.warning("Image de fond introuvable (génère le jardin ou charge une photo).")

                c_apply, c_reset = st.columns(2)
                if c_apply.button("✅ Valider et exporter JSON", type="primary", use_container_width=True):
                    if not image_for_overlay or not Path(image_for_overlay).exists():
                        st.error("Pas d’image de référence.")
                    else:
                        with Image.open(image_for_overlay) as im:
                            iw, ih = im.size
                        for k in list(buffers.keys()):
                            buffers[k] = _ensure_mask_hw(buffers[k], iw, ih)
                        for name, arr in buffers.items():
                            Image.fromarray(arr, mode="L").save(masks_dir / name)
                        scene_seq = OUTPUTS_DIR / "scene_sequential.json"
                        out_p, raw = _export_masks_edited_bundle(
                            masks_dir, buffers, Path(image_for_overlay), scene_seq
                        )
                        st.success(f"Masques sauvegardés + {out_p.name}")
                        st.download_button(
                            "📥 Télécharger scene_masks_edited.json",
                            raw,
                            file_name="scene_masks_edited.json",
                            mime="application/json",
                            use_container_width=True,
                        )

                if c_reset.button("♻️ Reset tous les masques", use_container_width=True):
                    st.session_state["mask_buffers"] = {
                        p.name: np.array(Image.open(p).convert("L")) for p in mask_files
                    }
                    st.session_state["mask_editor_active"] = None
                    st.session_state["mask_editor_last_unix"] = None
                    st.rerun()

        with st.expander("🧪 Phase1-B — sorties backend (SAM+Depth+Zone)"):
            pays_root = PROJECT_ROOT.parent
            overlay = pays_root / "zone-selection" / "outputs" / "user_zone_overlay.png"
            mask = pays_root / "zone-selection" / "outputs" / "user_zone_mask.png"
            final_scene = pays_root / "integration" / "final_scene_input.json"
            main_json = pays_root / "main.json"

            st.caption("Images (référence/preprocessed) + JSON final fusionné.")
            if overlay.exists():
                st.image(str(overlay), caption="user_zone_overlay.png", use_container_width=True)
            else:
                st.info("user_zone_overlay.png non trouvée (lance d'abord Phase1-B).")

            if mask.exists():
                st.image(str(mask), caption="user_zone_mask.png", use_container_width=True)
            else:
                st.info("user_zone_mask.png non trouvée (lance d'abord Phase1-B).")

            if main_json.exists():
                with st.expander("main.json (VisionOutput v0 export)"):
                    st.code(main_json.read_text(encoding="utf-8")[:4000] + "...")

            if final_scene.exists():
                with st.expander("final_scene_input.json (Vision + user_zone merge)"):
                    st.code(final_scene.read_text(encoding="utf-8")[:8000] + "...")
            else:
                st.info("final_scene_input.json non trouvée (lance d'abord Phase1-B).")

    st.divider()
    st.caption("Garden AI v3 — Flux.1 Fill | génération plante par plante")


if __name__ == "__main__":
    main()
