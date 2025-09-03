#!/bin/bash

# Скрипт запуска Audio Analysis микросервиса

echo "🎵 Starting Audio Analysis Microservice..."

# Проверяем наличие docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install Docker Compose."
    exit 1
fi

# Останавливаем если что-то уже запущено
echo "🛑 Stopping any existing containers..."
docker-compose down

# Собираем образы
echo "🔨 Building containers..."
docker-compose build

# Запускаем сервисы
echo "🚀 Starting services..."
docker-compose up -d

# Ждем пока сервисы запустятся
echo "⏳ Waiting for services to start..."
sleep 10

# Проверяем статус
echo "🔍 Checking service status..."
docker-compose ps

# Проверяем здоровье API
echo "🏥 Checking API health..."
if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    echo "✅ API is healthy and running on http://localhost:8000"
else
    echo "❌ API health check failed"
fi

# Проверяем Redis
echo "🔍 Checking Redis..."
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "❌ Redis is not responding"
fi

echo ""
echo "🎉 Audio Analysis Microservice started!"
echo ""
echo "📊 Available services:"
echo "  - API Server: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Flower Monitor: http://localhost:5555"
echo "  - Redis: localhost:6379"
echo ""
echo "📝 Logs:"
echo "  - API logs: docker-compose logs -f api"
echo "  - Worker logs: docker-compose logs -f worker"
echo "  - All logs: docker-compose logs -f"
echo ""
echo "🛑 Stop: ./stop.sh"
