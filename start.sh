#!/bin/bash

# Script to start Audio Analysis microservice

echo "🎵 Starting Audio Analysis Microservice..."

# Checking docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install Docker Compose."
    exit 1
fi

# Stopping any existing containers
echo "🛑 Stopping any existing containers..."
docker-compose down

# Building containers
echo "🔨 Building containers..."
docker-compose build

# Starting services
echo "🚀 Starting services..."
docker-compose up -d

# Ждем пока сервисы запустятся
echo "⏳ Waiting for services to start..."
sleep 10

# Checking service status
echo "🔍 Checking service status..."
docker-compose ps

# Checking API health
echo "🏥 Checking API health..."
if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    echo "✅ API is healthy and running on http://localhost:8000"
else
    echo "❌ API health check failed"
fi

echo ""
echo "🎉 Audio Analysis Microservice started!"
echo ""
echo "📊 Available services:"
echo "  - API Server: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Logs:"
echo "  - API logs: docker-compose logs -f api"
echo "  - All logs: docker-compose logs -f"
echo ""
echo "🛑 Stop: ./stop.sh"
