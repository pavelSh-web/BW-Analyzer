# Audio Analysis API

Microservice for audio analysis with PANNs tagging and musical characteristics extraction.

## Quick Start

```bash
# Start services
./start.sh

# Test API
./test-api.sh
```

## Services

- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Flower**: http://localhost:5555

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Sync analysis |
| `POST` | `/analyze/async` | Async analysis |
| `GET` | `/analyze/status/{task_id}` | Task status |
| `GET` | `/tags` | Available tags |

## Usage

### Sync Analysis
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@track.mp3"
```

### Async Analysis
```bash
# Start task
curl -X POST "http://localhost:8000/analyze/async" \
  -F "file=@track.mp3"

# Check status
curl "http://localhost:8000/analyze/status/{task_id}"
```

## Response Format

```json
{
  "filename": "track.mp3",
  "duration_seconds": 180.5,
  "tags": {
    "music": [{"label": "Electronic music", "prob": 0.95}],
    "instruments": [{"label": "Synthesizer", "prob": 0.78}],
    "vocal": [{"label": "Singing", "prob": 0.45}]
  },
  "musical_features": {
    "tempo_bpm": 128.0,
    "key": "C minor",
    "time_signature": "4/4",
    "energy": 0.85
  }
}
```

## Management

```bash
# Start
./start.sh

# Stop
./stop.sh

# Logs
docker-compose logs -f
```

## Requirements

- Docker & Docker Compose
- 2GB+ RAM for API, 4GB+ for worker
