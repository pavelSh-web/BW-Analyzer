![Cover image](https://github.com/pavelSh-web/BW-Analyzer/blob/d5a13d9c10044f64fd6725cc886a3a1ff38997f9/cover.png)

# BW-Analyzer

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
- `POST /analyze` — analyze an audio file

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
  -F "modules=tags,tempo,key,features"
```

Query parameters for `POST /analyze`:
- `modules`: `tags,tempo,key,features` (comma-separated)
- `normalize` (bool, default true): enable tag normalization
- `temperature` (float, default 1.2): softmax temperature within a group

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

### Tag normalization
- Within each group, apply log‑scale + softmax with temperature; return `prob` (sum within a group = 1).
- `temperature` controls distribution sharpness: lower = sharper, higher = more uniform.

## Feature notes
- **energy_value**: mean RMS (>= 0)
  - **energy thresholds**: `< 0.10` → low, `0.10–0.22` → mid, `> 0.22` → high
- **brightness_value**: 75th percentile spectral centroid over active frames (Hz)
  - Normalization: `centroid / (sr / 2)`
  - **brightness thresholds**: `< 0.20` → low, `0.20–0.50` → mid, `> 0.50` → high

## Modules
- **tags**: PANNs-based audio tags grouped into `genre`, `instruments`, `vocal`, `atmosphere`, `mood`, `effects`, `style`. Supports normalization (`normalize`, `temperature`).
- **tempo**: BPM estimation (DeepRhythm + librosa) with minimum-duration validation.
- **key**: Musical key detection (major/minor) using Skey + librosa.
- **features**: Basic features — `energy`, `energy_value`, `brightness`, `brightness_value`.

## Architecture (short)
```
Client → API (FastAPI) → Modules (tags / tempo / key / features) → Response
```
