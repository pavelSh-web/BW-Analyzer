#!/bin/bash

# Script to stop Audio Analysis microservice

echo "🛑 Stopping Audio Analysis Microservice..."

# Stop and remove containers
docker-compose down

# Optional: remove volumes
# echo "🗑️  Removing volumes..."
# docker-compose down -v

echo "✅ Audio Analysis Microservice stopped!"

# Show container status
echo "📊 Container status:"
docker-compose ps
