/**
 * Paysagea – Climate API
 * Stack : Node.js + Express
 *
 * Endpoints :
 *   GET /geocode/search?q=Lyon              → autocomplete Google Places
 *   GET /geocode/resolve?place_id=...       → lat/lon depuis un place_id Google
 *   GET /climate?lat=45.75&lon=4.85         → profil météo temps réel (7j)
 *   GET /climate/annual?lat=45.75&lon=4.85  → normales annuelles (12 mois glissants)
 */

import express from "express";
import cors from "cors";
import NodeCache from "node-cache";
import "dotenv/config";
import { writeFileSync, mkdirSync, existsSync, readdirSync, unlinkSync, statSync, readFileSync } from "fs";
import multer from "multer";
import { exec, spawn } from "child_process";
import path from "path";
import { promisify } from "util";

const execAsync = promisify(exec);

// Helper function to extract the LAST valid JSON object from stdout
function parsePythonOutput(stdout) {
  const trimmed = stdout.trim();
  let start = trimmed.lastIndexOf('{');
  while (start !== -1) {
    const end = trimmed.lastIndexOf('}');
    if (end > start) {
      try {
        const candidate = trimmed.substring(start, end + 1);
        return JSON.parse(candidate);
      } catch (e) {
        // Not a valid JSON or not the right start, try the previous '{'
      }
    }
    start = trimmed.lastIndexOf('{', start - 1);
  }
  throw new Error("Aucun JSON valide trouvé dans la sortie.");
}

const app = express();
const PORT = process.env.PORT || 3001;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

if (!GOOGLE_API_KEY) {
  console.warn("⚠️  GOOGLE_API_KEY manquante — /geocode/search ne fonctionnera pas");
}

// ─── Cache ────────────────────────────────────────────────────────────────────
const geocodeCache = new NodeCache({ stdTTL: 60 * 60 * 24 });      // 24h
const climateCache = new NodeCache({ stdTTL: 60 * 30 });            // 30 min
const annualCache = new NodeCache({ stdTTL: 60 * 60 * 24 * 7 });  // 7 jours (données stables)

app.use(cors());
app.use(express.json());

// ─── CONFIGURATION IA ─────────────────────────────────────────────────────────
// Chemins relatifs à la racine du monorepo (dossier parent de backend/)
const MONOREPO_ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const SAM_DEPTH_ROOT = path.join(MONOREPO_ROOT, "pipeline");
const WORK_DIR = path.join(MONOREPO_ROOT, "pipeline", "work");
const SHARED_DIR = path.join(MONOREPO_ROOT, "pipeline", "shared");
const PYTHON_PATH = process.env.PYTHON_PATH || "python";

// ─── CONFIGURATION GÉNÉRATION IMAGE IA (garden_ia_3) ─────────────────────────
const GARDEN_IA_3_ROOT = path.join(MONOREPO_ROOT, "pipeline", "generation");
const GARDEN_CLI_SCRIPT = path.join(GARDEN_IA_3_ROOT, "generate_garden_cli.py");

// ─── CONFIGURATION RAG ────────────────────────────────────────────────────────
const RAG_ROOT = path.join(MONOREPO_ROOT, "rag");
const RAG_CLI_SCRIPT = path.join(RAG_ROOT, "rag_cli.py");

// Créer les dossiers nécessaires
if (!existsSync("./uploads")) mkdirSync("./uploads");
if (!existsSync(WORK_DIR)) mkdirSync(WORK_DIR, { recursive: true });

// Servir le dossier work pour accéder aux masques et previews
app.use('/shared', express.static(SHARED_DIR));
app.use('/work', express.static(WORK_DIR)); // Permet d'accéder à user_zone.json via http://localhost:3001/work/user_zone.json
app.use('/uploads', express.static(path.resolve('uploads')));
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, "uploads/"),
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  },
});
const upload = multer({ storage });

// ─── HELPERS ──────────────────────────────────────────────────────────────────
async function fetchWithTimeout(url, options = {}, timeoutMs = 10000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(id);
    return res;
  } catch (err) {
    clearTimeout(id);
    throw err;
  }
}

function avg(arr) {
  const valid = arr.filter((v) => v !== null && v !== undefined && !isNaN(v));
  if (!valid.length) return null;
  return Math.round((valid.reduce((a, b) => a + b, 0) / valid.length) * 10) / 10;
}

function sum(arr) {
  const valid = arr.filter((v) => v !== null && v !== undefined && !isNaN(v));
  if (!valid.length) return null;
  return Math.round(valid.reduce((a, b) => a + b, 0) * 10) / 10;
}

// ─── ROUTE : Autocomplete Google Places ───────────────────────────────────────
app.get("/geocode/search", async (req, res) => {
  const q = req.query.q?.trim();
  if (!q || q.length < 2) return res.json([]);

  const cacheKey = `gplaces:${q.toLowerCase()}`;
  const cached = geocodeCache.get(cacheKey);
  if (cached) return res.json(cached);

  try {
    const url =
      `https://maps.googleapis.com/maps/api/place/autocomplete/json` +
      `?input=${encodeURIComponent(q)}&language=fr&types=geocode&key=${GOOGLE_API_KEY}`;

    const raw = await fetchWithTimeout(url);
    const data = await raw.json();

    if (data.status !== "OK" && data.status !== "ZERO_RESULTS") {
      throw new Error(`Google Places error: ${data.status}`);
    }

    const suggestions = (data.predictions ?? []).map((p) => ({
      label: p.description,
      short_label: p.structured_formatting?.main_text ?? p.description,
      secondary: p.structured_formatting?.secondary_text ?? "",
      place_id: p.place_id,
    }));

    geocodeCache.set(cacheKey, suggestions);
    res.json(suggestions);
  } catch (err) {
    console.error("[geocode/search]", err.message);
    res.status(502).json({ error: "Google Places indisponible", detail: err.message });
  }
});

// ─── ROUTE : Resolve place_id → lat/lon ──────────────────────────────────────
app.get("/geocode/resolve", async (req, res) => {
  const place_id = req.query.place_id?.trim();
  if (!place_id) return res.status(400).json({ error: "Paramètre place_id requis" });

  const cacheKey = `gresolve:${place_id}`;
  const cached = geocodeCache.get(cacheKey);
  if (cached) return res.json(cached);

  try {
    const url =
      `https://maps.googleapis.com/maps/api/place/details/json` +
      `?place_id=${encodeURIComponent(place_id)}` +
      `&fields=geometry,formatted_address,name&language=fr&key=${GOOGLE_API_KEY}`;

    const raw = await fetchWithTimeout(url);
    const data = await raw.json();

    if (data.status !== "OK") throw new Error(`Google Places Details error: ${data.status}`);

    const result = {
      label: data.result.formatted_address,
      short_label: data.result.name,
      lat: data.result.geometry.location.lat,
      lon: data.result.geometry.location.lng,
      place_id,
    };

    geocodeCache.set(cacheKey, result);
    res.json(result);
  } catch (err) {
    console.error("[geocode/resolve]", err.message);
    res.status(502).json({ error: "Google Places indisponible", detail: err.message });
  }
});

// ─── ROUTE : Profil météo temps réel ─────────────────────────────────────────
app.get("/climate", async (req, res) => {
  const lat = parseFloat(req.query.lat);
  const lon = parseFloat(req.query.lon);
  if (isNaN(lat) || isNaN(lon))
    return res.status(400).json({ error: "Paramètres lat et lon requis" });

  const cacheKey = `climate:${lat.toFixed(2)}:${lon.toFixed(2)}`;
  const cached = climateCache.get(cacheKey);
  if (cached) return res.json(cached);

  try {
    const url =
      `https://api.open-meteo.com/v1/forecast` +
      `?latitude=${lat}&longitude=${lon}` +
      `&current_weather=true` +
      `&hourly=relativehumidity_2m,precipitation,windspeed_10m,cloudcover,uv_index` +
      `&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,sunrise,sunset` +
      `&timezone=auto&forecast_days=7`;

    const raw = await fetchWithTimeout(url);
    const data = await raw.json();

    const current = data.current_weather ?? {};

    // Trouver l'index de l'heure actuelle dans le tableau hourly
    const nowISO = current.time;
    const hourlyTimes = data.hourly?.time ?? [];
    const currentHourIndex = nowISO
      ? hourlyTimes.findIndex((t) => t === nowISO)
      : 0;
    const hi = currentHourIndex >= 0 ? currentHourIndex : 0;

    const hNow = (key) => data.hourly?.[key]?.[hi] ?? null;
    const d0 = (key) => data.daily?.[key]?.[0] ?? null;

    const precipWeek = sum(data.daily?.precipitation_sum ?? []);
    const avgHumidity = avg(data.hourly?.relativehumidity_2m?.slice(0, 24) ?? []);

    const climate_profile = {
      current: {
        temp_c: current.temperature ?? null,
        windspeed_kmh: current.windspeed ?? null,
        wind_direction: current.winddirection ?? null,
        is_day: current.is_day ?? null,
        weathercode: current.weathercode ?? null,
      },
      today: {
        temp_max_c: d0("temperature_2m_max"),
        temp_min_c: d0("temperature_2m_min"),
        precip_mm: d0("precipitation_sum"),
        sunrise: d0("sunrise"),
        sunset: d0("sunset"),
        humidity_avg_pct: avgHumidity,
        cloudcover_pct: hNow("cloudcover"),   // ← heure actuelle (fix)
        uv_index: hNow("uv_index"),     // ← heure actuelle (fix)
      },
      week: {
        precip_total_mm: precipWeek,
        temp_max_c: Math.max(...(data.daily?.temperature_2m_max?.filter(Boolean) ?? [])) || null,
        temp_min_c: Math.min(...(data.daily?.temperature_2m_min?.filter(Boolean) ?? [])) || null,
      },
      timezone: data.timezone ?? null,
      timezone_abbreviation: data.timezone_abbreviation ?? null,
      generated_at: new Date().toISOString(),
    };

    climateCache.set(cacheKey, climate_profile);
    res.json(climate_profile);
  } catch (err) {
    console.error("[climate]", err.message);
    res.status(502).json({ error: "Open-Meteo indisponible", detail: err.message });
  }
});

// ─── ROUTE : Normales annuelles (12 mois glissants) ──────────────────────────
// GET /climate/annual?lat=45.75&lon=4.85
//
// Retourne les normales climatiques sur les 12 derniers mois :
// températures moyennes/min/max, précipitations, humidité, ensoleillement,
// jours de gel, classification botanique (zone de rusticité approx.)
app.get("/climate/annual", async (req, res) => {
  const lat = parseFloat(req.query.lat);
  const lon = parseFloat(req.query.lon);
  if (isNaN(lat) || isNaN(lon))
    return res.status(400).json({ error: "Paramètres lat et lon requis" });

  const cacheKey = `annual:${lat.toFixed(2)}:${lon.toFixed(2)}`;
  const cached = annualCache.get(cacheKey);
  if (cached) return res.json(cached);

  try {
    // Calculer les dates : 12 mois glissants
    const endDate = new Date();
    endDate.setDate(endDate.getDate() - 5); // Open-Meteo archive a ~5 jours de délai
    const startDate = new Date(endDate);
    startDate.setFullYear(startDate.getFullYear() - 1);

    const fmt = (d) => d.toISOString().split("T")[0]; // YYYY-MM-DD

    const url =
      `https://archive-api.open-meteo.com/v1/archive` +
      `?latitude=${lat}&longitude=${lon}` +
      `&start_date=${fmt(startDate)}&end_date=${fmt(endDate)}` +
      `&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,` +
      `precipitation_sum,sunshine_duration,windspeed_10m_max` +
      `&hourly=relativehumidity_2m` +
      `&timezone=auto`;

    const raw = await fetchWithTimeout(url, {}, 15000); // timeout plus long pour l'archive
    const data = await raw.json();

    if (data.error) throw new Error(data.reason ?? "Open-Meteo Archive error");

    const daily = data.daily ?? {};
    const hourly = data.hourly ?? {};

    // ── Températures ─────────────────────────────────────────────────────────
    const tempMean = avg(daily.temperature_2m_mean ?? []);
    const tempMax = Math.max(...(daily.temperature_2m_max?.filter(v => v !== null) ?? [-Infinity]));
    const tempMin = Math.min(...(daily.temperature_2m_min?.filter(v => v !== null) ?? [Infinity]));

    // ── Gel (jours où temp min < 0°C) ────────────────────────────────────────
    const frostDays = (daily.temperature_2m_min ?? [])
      .filter((v) => v !== null && v < 0).length;

    // ── Précipitations ───────────────────────────────────────────────────────
    const precipAnnual = sum(daily.precipitation_sum ?? []);

    // ── Ensoleillement (sunshine_duration en secondes → heures/jour) ─────────
    const sunshineHoursPerDay = daily.sunshine_duration?.length
      ? Math.round((avg(daily.sunshine_duration) / 3600) * 10) / 10
      : null;

    // ── Humidité moyenne annuelle ─────────────────────────────────────────────
    const humidityAnnual = avg(hourly.relativehumidity_2m ?? []);

    // ── Moyennes mensuelles (pour graphiques futurs) ──────────────────────────
    const months = Array.from({ length: 12 }, (_, i) => i); // 0=Jan … 11=Dec
    const monthly = months.map((m) => {
      const times = daily.time ?? [];
      const indices = times.reduce((acc, t, i) => {
        if (new Date(t).getMonth() === m) acc.push(i);
        return acc;
      }, []);

      const pick = (key) => indices.map((i) => daily[key]?.[i]).filter((v) => v !== null);

      return {
        month: m + 1, // 1=Jan … 12=Dec
        temp_mean_c: avg(pick("temperature_2m_mean")),
        temp_max_c: avg(pick("temperature_2m_max")),
        temp_min_c: avg(pick("temperature_2m_min")),
        precip_mm: sum(pick("precipitation_sum")),
        frost_days: pick("temperature_2m_min").filter((v) => v < 0).length,
        sunshine_h_day: pick("sunshine_duration").length
          ? Math.round((avg(pick("sunshine_duration")) / 3600) * 10) / 10
          : null,
      };
    });

    // ── Classification botanique approximative (zones USDA simplifiées) ───────
    let hardiness_zone;
    if (tempMin >= 10) hardiness_zone = "Zone 12+ (tropical)";
    else if (tempMin >= 4) hardiness_zone = "Zone 11 (subtropical)";
    else if (tempMin >= -1) hardiness_zone = "Zone 10 (méditerranéen chaud)";
    else if (tempMin >= -7) hardiness_zone = "Zone 9 (méditerranéen)";
    else if (tempMin >= -12) hardiness_zone = "Zone 8 (océanique doux)";
    else if (tempMin >= -18) hardiness_zone = "Zone 7 (tempéré)";
    else if (tempMin >= -23) hardiness_zone = "Zone 6 (continental modéré)";
    else if (tempMin >= -29) hardiness_zone = "Zone 5 (continental froid)";
    else if (tempMin >= -34) hardiness_zone = "Zone 4 (froid)";
    else hardiness_zone = "Zone 3 ou moins (très froid)";

    // ── Type de climat simplifié ──────────────────────────────────────────────
    let climate_type;
    const drySummer = monthly.slice(5, 8).every((m) => (m.precip_mm ?? 0) < 40);
    if (tempMean > 18 && precipAnnual < 400) climate_type = "Aride / semi-aride";
    else if (tempMean > 14 && drySummer) climate_type = "Méditerranéen";
    else if (tempMean > 8 && precipAnnual > 600) climate_type = "Océanique tempéré";
    else if (frostDays > 60) climate_type = "Continental";
    else climate_type = "Tempéré";

    const annual_profile = {
      // Résumé global
      summary: {
        temp_mean_annual_c: tempMean,
        temp_max_record_c: Math.round(tempMax * 10) / 10,
        temp_min_record_c: Math.round(tempMin * 10) / 10,
        precip_annual_mm: precipAnnual,
        humidity_annual_pct: humidityAnnual,
        sunshine_h_per_day: sunshineHoursPerDay,
        frost_days_per_year: frostDays,
        hardiness_zone,        // Zone de rusticité USDA approx.
        climate_type,          // Classification botanique
      },
      // Détail mois par mois
      monthly,
      // Méta
      period: { start: fmt(startDate), end: fmt(endDate) },
      source: "Open-Meteo Historical Archive",
      generated_at: new Date().toISOString(),
    };

    annualCache.set(cacheKey, annual_profile);
    res.json(annual_profile);
  } catch (err) {
    console.error("[climate/annual]", err.message);
    res.status(502).json({ error: "Open-Meteo Archive indisponible", detail: err.message });
  }
});

// ─── ROUTE : IA INTEGRATION BRIDGE ──────────────────────────────────────────
// POST /api/project/generate
// Reçoit le manifeste complet pour le RAG et la Génération d'Images
app.post("/api/project/generate", (req, res) => {
  const manifest = req.body;
  
  console.log("\n🚀 [IA BRIDGE] Nouveau Manifeste Projet Reçu :");
  console.log(`   project_id: ${manifest.project_id}`);
  console.log(`   user_zone:  ${manifest.visual_context?.user_zone?.length || 0} points`);
  console.log(`   image_url:  ${manifest.visual_context?.image_url}`);
  console.log("-------------------------------------------\n");

  // ── Sauvegarder dans shared/latest_project.json (accès par scripts IA) ──
  try {
    mkdirSync(SHARED_DIR, { recursive: true });
    writeFileSync(
      path.join(SHARED_DIR, "latest_project.json"),
      JSON.stringify(manifest, null, 2),
      "utf-8"
    );
    console.log(`💾 Manifeste sauvegardé localement → ${path.join(SHARED_DIR, "latest_project.json")}`);

    // ── Extraire et sauvegarder user_zone dans work/user_zone.json ──────────
    // Les points sont dans visual_context.user_zone (coordonnées normalisées 0-1)
    const userZone = manifest.visual_context?.user_zone;
    const imageSize = manifest.visual_context?.image_size; // [w, h] si disponible

    if (userZone && Array.isArray(userZone) && userZone.length > 0) {
      const zonePath = path.join(WORK_DIR, "user_zone.json");
      
      // Conversion en pixels si les dimensions sont disponibles
      let pointsPx = userZone;
      if (imageSize && Array.isArray(imageSize) && imageSize.length === 2) {
        const [w, h] = imageSize;
        pointsPx = userZone.map(p => ({
          x: Math.round(p.x * w),
          y: Math.round(p.y * h)
        }));
        console.log(`📐 Zone convertie en pixels (${w}x${h})`);
      }

      writeFileSync(zonePath, JSON.stringify({ 
        image_id: manifest.project_id,
        image_size: imageSize || null,
        points: pointsPx,
        normalized_points: userZone,
        created_at: new Date().toISOString()
      }, null, 2), "utf-8");
      console.log(`📍 Zone utilisateur sauvegardée → work/user_zone.json (${userZone.length} pts)`);
    } else {
      console.log("ℹ️  Pas de user_zone dans le manifeste — user_zone.json non mis à jour");
    }
  } catch (err) {
    console.error("⚠️ Erreur écriture fichier:", err.message);
  }

  res.json({ 
    status: "received", 
    timestamp: new Date().toISOString(),
    project_id: manifest.project_id,
  });
});


// ─── ROUTE : UPLOAD & PREPROCESS ─────────────────────────────────────────────
// POST /api/project/upload
// Reçoit une image et lance preprocess_cli.py
app.post("/api/project/upload", upload.single("image"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "Aucun fichier envoyé" });

  const inputPath = path.resolve(req.file.path);
  const scriptPath = path.join(SAM_DEPTH_ROOT, "preprocess_cli.py");
  
    console.log(`\n📸 [PREPROCESS] Exécution : "${PYTHON_PATH}" "${scriptPath}" --input "${inputPath}" --out-dir "${WORK_DIR}"`);

  try {
    const { stdout, stderr } = await execAsync(
      `"${PYTHON_PATH}" "${scriptPath}" --input "${inputPath}" --out-dir "${WORK_DIR}"`,
      { env: { ...process.env, PYTHONUTF8: "1" }, timeout: 90 * 1000 }
    );

    if (stdout) console.log("[PREPROCESS STDOUT]", stdout);
    if (stderr) console.warn("[PREPROCESS STDERR]", stderr);
    
    // On extrait uniquement l'objet JSON final de la sortie stdout
    const result = parsePythonOutput(stdout);
    
    // On ajoute l'URL web pour accéder à l'image pré-traitée
    const preprocessedFilename = path.basename(result.preprocessed_image);
    result.web_url = `http://localhost:${PORT}/work/${preprocessedFilename}`;

    console.log("✅ Preprocess terminé avec succès.");
    res.json(result);
  } catch (err) {
    console.error("❌ [PREPROCESS ERROR]", err.message);
    if (err.stdout) console.log("[PREPROCESS FAILED STDOUT]", err.stdout);
    if (err.stderr) console.error("[PREPROCESS FAILED STDERR]", err.stderr);
    res.status(500).json({ error: "Erreur lors du prétraitement", details: err.message, stderr: err.stderr });
  }
});

// ─── JOB STORE : suivi des analyses en arrière-plan ─────────────────────────────
// Garde en mémoire les jobs d'analyse en cours et terminés
const analysisJobs = new Map(); // job_id -> { status, result, error, startedAt }

// ─── ROUTE : ANALYSE (SAM + DEPTH) ───────────────────────────────────────────
// POST /api/project/analyze
// Lance l'analyse SAM+Depth en FOND (non-bloquant) et retourne un job_id
// Le client interroge ensuite GET /api/project/analyze-status/:job_id
app.post("/api/project/analyze", async (req, res) => {
  const { preprocess_json, force = false } = req.body;
  if (!preprocess_json) return res.status(400).json({ error: "preprocess_json est requis" });

  // ── Cache : si pipeline_result.json existe et est plus récent que le preprocess, on saute SAM ──
  if (!force) {
    try {
      const preprocessMtime = statSync(preprocess_json).mtimeMs;
      const existing = readdirSync(WORK_DIR)
        .filter(f => f.endsWith("_pipeline_result.json"))
        .map(f => { const p = path.join(WORK_DIR, f); return { p, mtime: statSync(p).mtimeMs }; })
        .sort((a, b) => b.mtime - a.mtime)[0];

      if (existing && existing.mtime >= preprocessMtime) {
        const cached = JSON.parse(readFileSync(existing.p, "utf-8"));
        const result = {
          merged_pipeline_json: existing.p,
          pipeline_json_path: existing.p,
          depth_preview_url: cached.files?.depth_preview
            ? `http://localhost:${PORT}/work/${path.basename(cached.files.depth_preview)}`
            : null,
        };
        const jobId = `job_cached_${Date.now()}`;
        analysisJobs.set(jobId, { status: "done", result, error: null, startedAt: Date.now() });
        console.log(`\n⚡ [ANALYSE] Cache SAM trouvé → réutilisation de ${path.basename(existing.p)}`);
        return res.json({ status: "started", job_id: jobId });
      }
    } catch (_) { /* preprocess_json introuvable ou work/ vide → continuer normalement */ }
  }

  const scriptPath = path.join(SAM_DEPTH_ROOT, "sam_depth_cli.py");
  const SAM_TIMEOUT_MS = 6 * 60 * 1000; // 6 minutes max — au-delà → zone-only

  const jobId = `job_${Date.now()}`;
  analysisJobs.set(jobId, { status: "running", result: null, error: null, startedAt: Date.now() });

  console.log(`\n🧠 [ANALYSE] Job ${jobId} démarré (timeout: ${SAM_TIMEOUT_MS / 60000}min)`);

  // Utiliser spawn (pas exec) pour pouvoir tuer le process sur Windows
  const child = spawn(PYTHON_PATH, [
    scriptPath,
    "--preprocess-json", preprocess_json,
    "--fuse",
  ], {
    cwd: SAM_DEPTH_ROOT,
    env: { ...process.env, PYTHONUTF8: "1" },
    windowsHide: true,
  });

  let stdout = "", stderr = "";
  child.stdout.on("data", d => { stdout += d.toString(); });
  child.stderr.on("data", d => {
    const s = d.toString();
    stderr += s;
    process.stdout.write(`[SAM] ${s}`); // stream live vers la console Node
  });

  // Timeout dur : tue tout l'arbre de process sur Windows
  const killTimer = setTimeout(() => {
    console.warn(`⏱️  [ANALYSE ${jobId}] Timeout ${SAM_TIMEOUT_MS / 60000}min — arrêt forcé → mode zone-only`);
    try { exec(`taskkill /F /T /PID ${child.pid}`); } catch (_) {}
    child.kill("SIGKILL");
    analysisJobs.set(jobId, {
      status: "done",
      result: { zone_only: true, warning: "SAM timeout — génération en mode zone-only" },
      error: null,
      startedAt: analysisJobs.get(jobId).startedAt,
    });
  }, SAM_TIMEOUT_MS);

  child.on("close", (code) => {
    clearTimeout(killTimer);
    if (analysisJobs.get(jobId)?.status === "done") return; // déjà résolu par le timeout

    if (code === 0) {
      try {
        const result = parsePythonOutput(stdout);
        if (result.depth_preview) {
          result.depth_preview_url = `http://localhost:${PORT}/work/${path.basename(result.depth_preview)}`;
        }
        if (result.merged_pipeline_json) result.pipeline_json_path = result.merged_pipeline_json;
        analysisJobs.set(jobId, { status: "done", result, error: null, startedAt: analysisJobs.get(jobId).startedAt });
        console.log(`✅ [ANALYSE ${jobId}] Terminée.`);
      } catch (e) {
        const stderrSnippet = stderr.slice(-1500);
        analysisJobs.set(jobId, {
          status: "error",
          result: null,
          error: `Parse error: ${e.message}\n\n--- stderr ---\n${stderrSnippet || "(vide)"}`,
          startedAt: analysisJobs.get(jobId).startedAt,
        });
      }
    } else {
      const stderrSnippet = stderr.slice(-1500);
      const isOom = stderrSnippet.toLowerCase().includes("memory") || stderrSnippet.includes("killed");
      analysisJobs.set(jobId, {
        status: "error",
        result: null,
        error: `SAM exit code ${code}${isOom ? " (mémoire insuffisante)" : ""}\n\n--- stderr ---\n${stderrSnippet || "(vide)"}`,
        startedAt: analysisJobs.get(jobId).startedAt,
      });
      console.error(`❌ [ANALYSE ${jobId}] exit ${code}`);
    }
  });

  // Répondre immédiatement avec le job_id
  res.json({ status: "started", job_id: jobId });
});

// ─── ROUTE : STATUT D'UNE ANALYSE ───────────────────────────────────────────────
// GET /api/project/analyze-status/:job_id
// Pollé par le frontend toutes les 5s pour savoir si l'analyse est terminée
app.get("/api/project/analyze-status/:jobId", (req, res) => {
  const job = analysisJobs.get(req.params.jobId);
  if (!job) return res.status(404).json({ error: "Job inconnu" });
  const elapsed = Math.round((Date.now() - job.startedAt) / 1000);
  res.json({ ...job, elapsed_s: elapsed });
});



// ─── ROUTE : SAUVEGARDER LA ZONE UTILISATEUR ────────────────────────────────
// POST /api/project/save-zone
// Sauvegarde immédiatement user_zone.json dans WORK_DIR
// Appelé par le frontend dès que l'utilisateur valide sa zone
app.post("/api/project/save-zone", (req, res) => {
  const { points, normalized_points, image_size, image_id } = req.body || {};

  if (!normalized_points || !Array.isArray(normalized_points) || normalized_points.length < 3) {
    return res.status(400).json({ error: "normalized_points requis (minimum 3 points)" });
  }

  const zonePath = path.join(WORK_DIR, "user_zone.json");
  const payload = {
    image_id: image_id || "unknown",
    image_size: image_size || null,
    points: points || normalized_points, // pixels si fournis, sinon normalisés
    normalized_points,
    created_at: new Date().toISOString()
  };

  try {
    writeFileSync(zonePath, JSON.stringify(payload, null, 2), "utf-8");
    console.log(`📍 [SAVE-ZONE] Zone sauvegardée → work/user_zone.json (${normalized_points.length} pts)`);
    res.json({ status: "saved", points_count: normalized_points.length, path: zonePath });
  } catch (err) {
    console.error("❌ [SAVE-ZONE ERROR]", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ─── ROUTE : LIST FILES IN WORK (DEBUG) ─────────────────────────────────────
app.get("/api/work-files", (req, res) => {
  try {
    const files = readdirSync(WORK_DIR);
    res.json({ work_dir: WORK_DIR, files });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── ROUTE : RÉCEPTION RÉSULTAT RAG ─────────────────────────────────────────
// POST /api/rag/recommend
// Appelle rag_cli.py avec les préférences utilisateur → génère work/rag_output.json
// Body: { style, exposition, entretien, description, climat, taille, budget, n_plants }
app.post("/api/rag/recommend", async (req, res) => {
  const prefs = {
    style:       req.body.style       || "naturel",
    exposition:  req.body.exposition  || "plein_soleil",
    entretien:   req.body.entretien   || "moyen",
    description: req.body.description || "",
    climat:      req.body.climat      || "tempere",
    taille:      req.body.taille      || "moyen",
    budget:      req.body.budget      || "moyen",
    region:      req.body.region      || "France",
    n_plants:    parseInt(req.body.n_plants) || 6,
  };

  // Écrire les préférences dans un fichier temp (exec ne supporte pas stdin)
  const prefsPath = path.join(WORK_DIR, "_rag_prefs_tmp.json");
  writeFileSync(prefsPath, JSON.stringify(prefs), "utf-8");

  const cmd = [
    `"${PYTHON_PATH}"`,
    `"${RAG_CLI_SCRIPT}"`,
    `--prefs "${prefsPath}"`,
    `--n-plants ${prefs.n_plants}`,
  ].join(" ");

  console.log(`\n🌿 [RAG] Appel rag_cli.py — style=${prefs.style} expo=${prefs.exposition}`);

  try {
    const { stdout, stderr } = await execAsync(cmd, {
      cwd: RAG_ROOT,
      env: { ...process.env, PYTHONUTF8: "1" },
      timeout: 30 * 1000,
    });
    unlinkSync(prefsPath);

    if (stderr) console.warn("[RAG STDERR]", stderr.substring(0, 400));

    let ragData;
    try {
      ragData = JSON.parse(stdout.trim());
    } catch {
      throw new Error(`JSON invalide depuis rag_cli.py: ${stdout.substring(0, 200)}`);
    }

    if (ragData.error) {
      return res.status(500).json({ error: ragData.error });
    }

    // Sauvegarder dans work/rag_output.json (lu par generate_garden_cli.py)
    mkdirSync(WORK_DIR, { recursive: true });
    const outPath = path.join(WORK_DIR, "rag_output.json");
    writeFileSync(outPath, JSON.stringify(ragData, null, 2), "utf-8");

    const plantCount = (ragData.jardin || []).length;
    console.log(`✅ [RAG] ${plantCount} plantes → work/rag_output.json`);

    res.json({ status: "ok", plant_count: plantCount, plants: ragData.jardin });
  } catch (err) {
    console.error("❌ [RAG ERROR]", err.message);
    res.status(500).json({ error: "Erreur rag_cli.py", details: err.message });
  }
});


// POST /api/project/rag-output
// Reçoit le JSON de plantes recommandées (produit par le RAG collègue ou mock)
// et le sauvegarde dans work/rag_output.json pour generate_garden_cli.py
app.post("/api/project/rag-output", (req, res) => {
  const ragData = req.body;
  if (!ragData || typeof ragData !== "object") {
    return res.status(400).json({ error: "Corps JSON invalide" });
  }

  // Accepte les clés "jardin" ou "garden" (les deux formats)
  const plants = ragData.jardin || ragData.garden || ragData.plants;
  if (!Array.isArray(plants) || plants.length === 0) {
    return res.status(400).json({ error: "Aucune plante trouvée (clés attendues: jardin, garden, ou plants)" });
  }

  try {
    mkdirSync(WORK_DIR, { recursive: true });
    const outPath = path.join(WORK_DIR, "rag_output.json");
    writeFileSync(outPath, JSON.stringify(ragData, null, 2), "utf-8");
    console.log(`\n🌿 [RAG-OUTPUT] ${plants.length} plantes sauvegardées → work/rag_output.json`);
    res.json({ status: "ok", plant_count: plants.length, path: outPath });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// ─── ROUTE : GÉNÉRATION IMAGE IA (BFL FLUX Fill PRO + SAM/Depth) ────────────
// POST /api/project/generate-image
// Utilise pipeline_result.json (SAM + Depth) + user_zone.json depuis WORK_DIR
// Lance generate_garden_cli.py et retourne l'URL + masques individuels des plantes
app.post("/api/project/generate-image", async (req, res) => {
  const {
    plant_density = "medium",
    description = "",
    seed = 42,
    max_plants = 6,
    pipeline_json = "",   // chemin complet optionnel, auto-detect si vide
  } = req.body || {};

  const zoneJson  = path.join(WORK_DIR, "user_zone.json");
  const bflApiKey = process.env.BFL_API_KEY || "";

  // Vérifications
  if (!existsSync(zoneJson)) {
    return res.status(400).json({
      error: "user_zone.json introuvable dans work/. Définissez d'abord la zone plantable."
    });
  }
  if (!bflApiKey && process.env.MOCK_BFL?.toLowerCase() !== "true") {
    return res.status(400).json({
      error: "BFL_API_KEY non définie dans .env. Ajoutez BFL_API_KEY=votre_cle dans backend/.env"
    });
  }

  // Auto-detect le pipeline_result.json le plus récent si non fourni
  let pipelineJsonPath = pipeline_json;
  if (!pipelineJsonPath || !existsSync(pipelineJsonPath)) {
    try {
      const workFiles = readdirSync(WORK_DIR);
      const { statSync } = await import("fs");
      const pipelineFiles = workFiles
        .filter(f => f.endsWith("_pipeline_result.json"))
        .map(f => ({ name: f, mtime: statSync(path.join(WORK_DIR, f)).mtimeMs }))
        .sort((a, b) => b.mtime - a.mtime);

      if (pipelineFiles.length > 0) {
        pipelineJsonPath = path.join(WORK_DIR, pipelineFiles[0].name);
        console.log(`   Auto-detect pipeline: ${pipelineFiles[0].name}`);
      } else {
        // Mode zone-only : pas de pipeline SAM, generate_garden_cli.py gère le fallback
        pipelineJsonPath = "";
        console.log("   ⚠️  Pas de pipeline SAM → mode zone-only (génération sans segmentation)");
      }
    } catch(e) {
      pipelineJsonPath = "";
      console.warn(`   Erreur lecture work dir: ${e.message} → mode zone-only`);
    }
  }


  const cmd = [
    `"${PYTHON_PATH}"`,
    `"${GARDEN_CLI_SCRIPT}"`,
    `--work-dir "${WORK_DIR}"`,
    `--pipeline-json "${pipelineJsonPath}"`,
    `--plant-density "${plant_density}"`,
    `--max-plants ${parseInt(max_plants) || 6}`,
    `--seed ${parseInt(seed) || 42}`,
    description ? `--prompt "${description.replace(/"/g, "'").substring(0, 300)}"` : "",
  ].filter(Boolean).join(" ");

  console.log(`\n🌿 [IMAGE-GEN] Démarrage génération BFL (SAM+Depth)...`);
  console.log(`   Plant density : ${plant_density} | Max plants: ${max_plants} | Seed: ${seed}`);
  console.log(`   Pipeline JSON : ${path.basename(pipelineJsonPath)}`);
  console.log(`   BFL_API_KEY   : ${bflApiKey ? `présente (${bflApiKey.length} chars)` : "absente (mode MOCK)"}`);

  try {
    const { stdout, stderr } = await execAsync(cmd, {
      cwd: GARDEN_IA_3_ROOT,
      env: {
        ...process.env,
        PYTHONUTF8: "1",
        BFL_API_KEY: bflApiKey,
        PYTHONPATH: GARDEN_IA_3_ROOT,
      },
      timeout: 5 * 60 * 1000, // 5 minutes max (BFL peut prendre 60s)
    });

    if (stdout) console.log("[IMAGE-GEN STDOUT]", stdout.substring(0, 3000));
    if (stderr) console.warn("[IMAGE-GEN STDERR]", stderr.substring(0, 600));

    const result = parsePythonOutput(stdout);

    // URLs web accessibles depuis le frontend
    const outputFilename = result.output_filename || "final_garden.png";
    result.web_url = `http://localhost:${PORT}/work/${outputFilename}`;

    // URLs des masques individuels (mode édition)
    if (result.plant_masks && Array.isArray(result.plant_masks)) {
      result.plant_masks = result.plant_masks.map(m => ({
        ...m,
        mask_web_url: `http://localhost:${PORT}/work/masks/${m.mask_file}`,
      }));
    }

    console.log(`✅ [IMAGE-GEN] Génération terminée → ${result.web_url}`);
    console.log(`   Segments utilisés: ${result.segments_used || 0} | Masques: ${result.plant_masks?.length || 0}`);
    res.json(result);

  } catch (err) {
    console.error("❌ [IMAGE-GEN ERROR]", err.message);
    if (err.stdout) console.log("[IMAGE-GEN FAILED STDOUT]", err.stdout.substring(0, 600));
    if (err.stderr) console.error("[IMAGE-GEN FAILED STDERR]", err.stderr.substring(0, 300));
    res.status(500).json({
      error: "Erreur lors de la génération d'image",
      details: err.message,
      stderr: err.stderr?.substring(0, 500),
    });
  }
});


app.get("/health", (_req, res) => res.json({ status: "ok", ts: new Date().toISOString() }));

app.listen(PORT, () => console.log(`Paysagea Climate API → http://localhost:${PORT}`));
