#!/bin/bash

# Скрипт остановки Audio Analysis микросервиса

echo "🛑 Stopping Audio Analysis Microservice..."

# Останавливаем и удаляем контейнеры
docker-compose down

# Опционально: удаляем volumes (раскомментируйте если нужно)
# echo "🗑️  Removing volumes..."
# docker-compose down -v

echo "✅ Audio Analysis Microservice stopped!"

# Показываем статус
echo "📊 Container status:"
docker-compose ps
