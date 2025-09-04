# Audio Analysis Microservice

FastAPI microservice for audio analysis with PANNs tagging and musical feature extraction.

## Features

- **Synchronous Analysis**: Immediate audio analysis results
- **Asynchronous Analysis**: External queue integration (Redis/RabbitMQ)
- **PANNs Audio Tagging**: 527-class audio classification with curated musical tags
- **Musical Features**: Tempo (BPM), key detection, time signature estimation
- **Spectral Analysis**: Energy, brightness, zero-crossing rate, spectral rolloff, MFCCs
- **Docker Support**: Complete containerization
- **External Queue Integration**: Configurable queue backends

## Quick Start

### Docker (Recommended)

```bash
# Build and run
docker build -t audio-analyzer .
docker run -p 8000:8000 audio-analyzer

# With external Redis queue
docker run -p 8000:8000 -e QUEUE_TYPE=redis -e REDIS_URL=redis://host.docker.internal:6379/0 audio-analyzer
```

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# With external queue (optional)
export QUEUE_TYPE=redis
export REDIS_URL=redis://localhost:6379/0
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Synchronous Analysis
- `POST /analyze` - Analyze audio file (immediate response)

### Asynchronous Analysis (requires external queue)
- `POST /analyze/async` - Enqueue analysis task
- `GET /queue/status` - External queue status
- `GET /queue/info` - Detailed queue information

### Information
- `GET /` - Service information and configuration
- `GET /tags` - Available tag groups
- `GET /tags/pretty` - Pretty-printed tag groups

## Usage Examples

### Synchronous Analysis
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@audio.mp3"
```

### Asynchronous Analysis
```bash
# Submit task to external queue
curl -X POST "http://localhost:8000/analyze/async" \
  -F "file=@audio.mp3" \
  -F "callback_url=https://your-server.com/webhook" \
  -F "top_tags_per_group=5"

# Check queue status
curl "http://localhost:8000/queue/status"
```

## External Queue Configuration

Set environment variables to enable external queue integration:

```bash
# Queue type: none, redis, rabbitmq
export QUEUE_TYPE=redis

# Redis configuration
export REDIS_URL=redis://localhost:6379/0

# RabbitMQ configuration
export RABBITMQ_URL=amqp://guest:guest@localhost:5672/
```

## External Worker

Use the provided worker to process queued tasks:

```bash
# Install worker dependencies
pip install httpx

# Run worker
python examples/external_worker.py

# With configuration
export QUEUE_TYPE=redis
export REDIS_URL=redis://localhost:6379/0
python examples/external_worker.py
```

## Architecture

### Synchronous Mode
```
Client → API → Analysis → Response
```

### Asynchronous Mode
```
Client → API → External Queue → Worker → Analysis → Callback URL
```

## Queue Integration

The microservice supports multiple queue backends:

- **None**: Synchronous processing only
- **Redis**: Redis-based message queue
- **RabbitMQ**: AMQP-based message queue

Workers can be deployed separately and process tasks from the external queue, sending results to callback URLs.