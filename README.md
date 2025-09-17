![Cover image](https://github.com/pavelSh-web/BW-Analyzer/blob/b05536cff4088157cf0e4e11a94ff82a6a7a806d/cover.png)

Music analysis service. Estimates genre probabilities and additional musical characteristics from an audio file, and also detects tempo (BPM), musical key, energy, and brightness of the track.

## Features

- **Synchronous analysis** via a single `POST /analyze`
- **PANNs tags** with grouping and normalization (softmax with temperature)
- **Tempo** (BPM) and **Key**
- **Features**: energy/brightness (both category and value)
- Run with Docker or locally

## Quick Start

### Docker (recommended)

```bash
# Build and run
docker build -t audio-analyzer .
docker run -p 8000:8000 audio-analyzer
```

### Docker Compose (start.sh)

```bash
chmod +x start.sh stop.sh
./start.sh
# stop:
./stop.sh
```

Notes:
- Both options launch the same API. The compose/start.sh path uses docker-compose to build and run, and mounts `./app` into the container for convenient local edits.
- Use `start.sh` for day-to-day development; use `docker build/run` for a quick one-off run without compose.

### Local run

```bash
# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

### Endpoints
- `GET /` — service info
- `GET /modules` — list of available modules
- `GET /tags?desc=false` — tag groups (with descriptions when `desc=true`)
- `POST /analyze` — analyze an audio file (full pipeline, normalized tags)
- `POST /embedding` — analyze + return embedding vector

### Tag groups
- `genre`
- `instruments`
- `vocal`
- `atmosphere`
- `mood`
- `effects`
- `style`

### GET /tags examples
Without descriptions (`?desc=false`, default):
```json
{
  "total": 120,
  "categories": {
    "genre": 32,
    "instruments": 25,
    "vocal": 7,
    ...
  },
  "list": {
    "genre": ["Pop", "Rock", "Hip hop", "Jazz", "Blues", ... ],
    "instruments": ["Guitar", "Electric guitar", "Acoustic guitar", ... ],
    "vocal": ["Singing", "Choir", "A cappella", "Rapping", ... ],
    "atmosphere": ["Drone", "Noise", "Buzz", "Hiss", "Hum", ... ],
    "mood": ["Happy", "Sad", "Tender", "Funny", "Exciting", ... ],
    "effects": ["Echo", "Reverberation", "Distortion", ... ],
    "style": ["Jingle", "Theme", "Background music", "Wedding", ... ]
  }
}
```

With descriptions (`?desc=true`):
```json
{
  "total": 120,
  "categories": {
    "genre": 32,
    "instruments": 25,
    "vocal": 7,
    ...
  },
  "list": {
    "genre": [
      { "name": "Pop", "description": "Popular music genre with wide appeal." },
      ...
    ],
    "instruments": [
      { "name": "Guitar", "description": "Plucked string instrument." },
      ...
    ],
    "vocal": [
      { "name": "Singing", "description": "Musical sounds produced with the human voice." },
      ...
    ],
    "atmosphere": [
      { "name": "Drone", "description": "Sustained tone or minimal harmonic movement." }
      ...
    ],
    "mood": [
      { "name": "Happy", "description": "Positive, cheerful character." }
      ...
    ],
    "effects": [
      { "name": "Reverberation", "description": "Persistence of sound after it is produced." }
      ...
    ],
    "style": [
      { "name": "Jingle", "description": "Short catchy musical phrase, often for ads." }
      ...
    ]
  }
}
```

## Usage Examples

### Example: analyze
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@audio.mp3" \
  
```

Query parameters for `POST /analyze`:
- `temperature` (float, default 1.2): softmax temperature within a group (for tag normalization).

### Response format (main fields)

```json
{
  "tags": {
    "genre": [ { "label": "Electronic", "prob": 0.24 }, ... ],
    "instruments": [ ... ],
    "vocal": [ ... ],
    "atmosphere": [ ... ],
    "mood": [ ... ],
    "effects": [ ... ],
    "style": [ ... ]
  },
  "tempo": 124,
  "key": "Am",
  "energy": "mid",            // category: low | mid | high
  "energy_value": 0.135,       // mean RMS
  "brightness": "high",       // category: low | mid | high (by spectral centroid)
  "brightness_value": 3162.0   // centroid in Hz
}
```

### Example: embedding
```bash
curl -X POST "http://localhost:8000/embedding" \
  -F "file=@audio.mp3"
```

Response (embedding only):
```json
{
  "embedding": [0.02, 0.15, 0.01, ...],
  "dim": 163
}
```

Embedding details:
- **Block-wise L2 normalized** with semantic weights for balanced representation
- **Structure**: Tags[109] + Tempo[1] + Energy/Brightness[2] + Rhythm[3] + Harmony[37] + Timbre[6] + Dynamics[5] = 163
- **L2 normalized**: Global L2 norm = 1.0 for consistent cosine similarity
- **Deterministic**: Same audio always produces identical embedding
- **Block weights**: tags (1.0), tempo (0.5), energy/brightness (0.7), rhythm (0.6), harmony (0.8), timbre (0.7), dynamics (0.6)

### Tag normalization
- Within each group, apply log‑scale + softmax with temperature; return `prob` (sum within a group = 1).
- `temperature` controls distribution sharpness: lower = sharper, higher = more uniform.

## Feature notes
- **energy_value**: mean RMS (>= 0)
  - **energy thresholds**: `< 0.10` → low, `0.10–0.22` → mid, `> 0.22` → high (configurable in `ENERGY_THRESHOLDS`)
- **brightness_value**: 75th percentile spectral centroid over active frames (Hz)
  - Normalization: `centroid / (sr / 2)`
  - **brightness thresholds**: `< 0.20` → low, `0.20–0.50` → mid, `> 0.50` → high (configurable in `BRIGHTNESS_THRESHOLDS`)

## Modules
- **tags**: PANNs-based audio tags grouped into `genre`, `instruments`, `vocal`, `atmosphere`, `mood`, `effects`, `style`. Supports normalization (`normalize`, `temperature`).
- **tempo**: BPM estimation (DeepRhythm + librosa) with minimum-duration validation.
- **key**: Musical key detection (major/minor) using Skey + librosa.
- **features**: Comprehensive analysis with energy, brightness, rhythm, harmony, timbre, and dynamics features.

### Features Details

**Rhythm**:
- `onset_density`: Events per second (distinguishes danceable vs calm tracks)
- `percussive_harmonic_ratio`: Balance of percussive vs harmonic content (via HPSS)
- `beat_histogram_mean/std`: Rhythm structure statistics

**Harmony**:
- `chroma_mean/std`: 12-dimensional note distribution (harmonic "color")
- `tonnetz_mean/std`: 6-dimensional harmonic network features (chord relationships)
- `key_clarity`: How clearly the musical key is expressed

**Timbre/Spectral**:
- `spectral_flatness_mean/std`: Noisiness vs tonality measure
- `spectral_bandwidth_mean/std`: Frequency spread (narrow vocal vs wide electronic)
- `zero_crossing_rate_mean/std`: Indicator of noisy/percussive sounds

**Dynamics**:
- `dynamic_range_db`: Contrast between loud and quiet parts
- `loudness_mean/std/min/max`: RMS energy statistics
- `loudness_range`: Difference between 95th and 5th percentile
- `lufs`: Integrated loudness (if pyloudnorm available)

## Architecture (short)
```
Client → API (FastAPI) → Modules (tags / tempo / key / features) → Response
```
