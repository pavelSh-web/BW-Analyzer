#!/bin/bash

# Скрипт для тестирования Audio Analysis API

echo "🧪 Testing Audio Analysis API..."

API_URL="http://localhost:8000"

# Проверяем доступность API
echo "📡 Checking API availability..."
if curl -f "$API_URL/" > /dev/null 2>&1; then
    echo "✅ API is available"
else
    echo "❌ API is not available. Please run ./start.sh first"
    exit 1
fi

# Проверяем эндпоинты
echo ""
echo "🔍 Testing endpoints..."

# 1. Root endpoint
echo "1️⃣ Testing root endpoint..."
curl -s "$API_URL/" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f'✅ Service: {data[\"service\"]}, Version: {data[\"version\"]}')" 2>/dev/null || echo "❌ Root endpoint failed"

# 2. Tags endpoint  
echo "2️⃣ Testing tags endpoint..."
curl -s "$API_URL/tags" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f'✅ Total tags: {data[\"total_tags\"]}')" 2>/dev/null || echo "❌ Tags endpoint failed"

# 3. Queue info
echo "3️⃣ Testing queue info..."
curl -s "$API_URL/queue/info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f'✅ Queue available: {data[\"queue_available\"]}')" 2>/dev/null || echo "❌ Queue info failed"

echo ""
echo "📊 API Documentation: $API_URL/docs"
echo "🌸 Flower Monitor: http://localhost:5555"
echo ""

# Если есть тестовый файл, пробуем синхронный анализ
TEST_FILE="test.mp3"
if [ -f "$TEST_FILE" ]; then
    echo "🎵 Testing with $TEST_FILE..."
    curl -X POST "$API_URL/analyze?top_tags_per_group=3" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@$TEST_FILE" \
         -s | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'✅ Analysis completed in {data[\"elapsed_sec\"]}s')
    print(f'   Duration: {data[\"duration_seconds\"]}s')
    print(f'   Music tags: {len(data[\"tags\"][\"music\"])}')
    print(f'   Instruments: {len(data[\"tags\"][\"instruments\"])}')
    print(f'   Vocal: {len(data[\"tags\"][\"vocal\"])}')
except:
    print('❌ Analysis failed')
" 2>/dev/null
else
    echo "ℹ️  No test.mp3 file found. Skipping analysis test."
    echo "   To test analysis, add a test.mp3 file and run again."
fi

echo ""
echo "🎉 API testing completed!"
