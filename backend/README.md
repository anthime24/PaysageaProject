
Pipeline : **Adresse → Nominatim → lat/lon → Open-Meteo → JSON climatique**

---

## Structure

```
backend/
  server.js         ← API Express (3 endpoints)
  package.json

frontend/
  useClimate.js     ← Hook React (autocomplete + fetch)
  AddressSearch.jsx ← Composant clé-en-main (optionnel)
```

---

## Backend

### Installation

```bash
cd backend
npm install
npm run dev       # node --watch (Node 18+)
```

### Endpoints

#### `GET /geocode/search?q=Lyon`
Autocomplétion — retourne 5 suggestions.

```json
[
  {
    "label": "Lyon, Métropole de Lyon, Ain, Auvergne-Rhône-Alpes, France métropolitaine, France",
    "short_label": "Lyon, Métropole de Lyon, France",
    "lat": 45.74846,
    "lon": 4.84671,
    "place_id": 258926509,
    "type": "city"
  }
]
```

#### `GET /geocode/resolve?q=Lyon, France`
Résolution directe (sans autocomplétion).

```json
{
  "label": "Lyon, ...",
  "short_label": "Lyon, Métropole de Lyon, France",
  "lat": 45.74846,
  "lon": 4.84671,
  "place_id": 258926509
}
```

#### `GET /climate?lat=45.75&lon=4.85`
Profil climatique complet.

```json
{
  "current": {
    "temp_c": 13.4,
    "windspeed_kmh": 11.2,
    "wind_direction": 210,
    "is_day": 1,
    "weathercode": 3
  },
  "today": {
    "temp_max_c": 17.1,
    "temp_min_c": 9.3,
    "precip_mm": 0.2,
    "sunrise": "2024-04-15T06:34",
    "sunset": "2024-04-15T20:41",
    "humidity_avg_pct": 68,
    "cloudcover_pct": 45,
    "uv_index": 4
  },
  "week": {
    "precip_total_mm": 12.4,
    "temp_max_c": 19.2,
    "temp_min_c": 7.1
  },
  "timezone": "Europe/Paris",
  "timezone_abbreviation": "CEST",
  "generated_at": "2024-04-15T10:22:00.000Z"
}
```

---

## Frontend

### Variable d'environnement

```env
# .env.local
VITE_CLIMATE_API_URL=http://localhost:3001
```

### Option A — Hook seul (intégration dans votre propre UI)

```jsx
import { useClimate } from "./useClimate";

function MyComponent() {
  const { search, resolve, suggestions, location, climate, projectContext, loading, error } = useClimate();

  return (
    <>
      <input
        onChange={e => search(e.target.value)}    // autocomplete
        onKeyDown={e => e.key === "Enter" && resolve(e.target.value)}
      />

      {suggestions.map(s => (
        <div key={s.place_id} onClick={() => resolve(s)}>
          {s.short_label}
        </div>
      ))}

      {projectContext && <pre>{JSON.stringify(projectContext, null, 2)}</pre>}
    </>
  );
}
```

### Option B — Composant clé-en-main

```jsx
import AddressSearch from "./AddressSearch";

function MyPage() {
  function handleResolve(projectContext) {
    // projectContext = { location, climate_profile, source }
    console.log(projectContext);
    // → envoyer à votre backend / passer au LLM
  }

  return <AddressSearch onResolve={handleResolve} />;
}
```

### projectContext (stable, prêt pour le LLM)

```json
{
  "location": {
    "label": "Lyon, ...",
    "short_label": "Lyon, Métropole de Lyon, France",
    "lat": 45.74846,
    "lon": 4.84671,
    "place_id": 258926509
  },
  "climate_profile": { /* cf. /climate */ },
  "source": {
    "geocoder": "Nominatim/OSM",
    "weather": "Open-Meteo"
  }
}
```

---

## Notes

- **Cache** : géocodage 24h, météo 30 min (in-memory, node-cache)
- **Rate limit Nominatim** : le debounce 300ms + cache évitent tout abus
- **User-Agent** : requis par Nominatim — modifiez `contact@paysagea.fr` dans server.js
- **Pas de clé API** : Nominatim et Open-Meteo sont gratuits et open
