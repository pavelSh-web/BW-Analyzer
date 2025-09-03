# Audio Tags & Features API

Микросервис для анализа аудио: автоматическое тегирование + извлечение музыкальных характеристик.

## 🎯 Возможности

- **PANNs теги**: автоматическая классификация по 527 классам (жанры, инструменты, настроения)
- **Темп (BPM)**: определение темпа композиции
- **Тональность**: распознавание музыкального ключа (C, D, E, etc.)
- **Размер**: оценка музыкального размера (4/4, 3/4, 6/8, etc.)
- **Спектральные характеристики**: энергия, яркость, тембральные особенности
- **Временная обработка**: файлы не сохраняются, только анализ

## 🚀 Установка и запуск

### Локально

```bash
# Установка зависимостей
python -m pip install --upgrade pip
pip install -r requirements.txt

# Запуск API
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Docker

```bash
# Сборка образа
docker build -t audio-analyzer:latest .

# Запуск
docker run --rm -p 8001:8001 audio-analyzer:latest
```

## 📡 API

### POST /analyze

Анализирует аудиофайл и возвращает теги + характеристики.

**Параметры:**
- `file`: аудиофайл (WAV, MP3, FLAC, OGG, M4A)
- `top_tags`: количество топ-тегов (по умолчанию 10)

**Пример запроса:**
```bash
curl -X POST "http://localhost:8001/analyze?top_tags=10" \
  -F "file=@your_track.mp3"
```

**Пример ответа:**
```json
{
  "filename": "track.mp3",
  "duration_seconds": 180.5,
  "sample_rate": 44100,
  "panns_top_tags": [
    {"label": "Music", "prob": 0.773},
    {"label": "Electronic", "prob": 0.152},
    {"label": "Techno", "prob": 0.089},
    {"label": "House music", "prob": 0.067}
  ],
  "musical_features": {
    "tempo_bpm": 128.5,
    "key": "C",
    "time_signature": "4/4",
    "energy": 0.45,
    "brightness": 2150.3,
    "zero_crossing_rate": 0.12,
    "spectral_rolloff": 3240.1,
    "mfcc_mean": [-123.4, 45.2, ...]
  },
  "elapsed_sec": 2.34
}
```

### GET /

Информация о сервисе и доступных функциях.

## 🎵 Поддерживаемые форматы

- WAV
- MP3
- FLAC
- OGG
- M4A

## 🧠 Модели

- **PANNs CNN14**: предобученная модель для аудио-классификации (527 классов)
- **librosa**: извлечение музыкальных характеристик

## ⚡ Особенности

- **Без постоянного хранения**: файлы обрабатываются и сразу удаляются
- **Быстрый анализ**: результат за 2-5 секунд на CPU
- **Автоматическая очистка**: временные файлы удаляются после обработки
- **REST API**: простая интеграция с любыми системами

## 📊 Музыкальные характеристики

| Параметр | Описание |
|----------|----------|
| `tempo_bpm` | Темп в ударах в минуту |
| `key` | Музыкальная тональность (C, C#, D, ...) |
| `time_signature` | Размер (4/4, 3/4, 6/8, etc.) |
| `energy` | Общая энергия трека |
| `brightness` | Спектральный центроид (яркость) |
| `zero_crossing_rate` | Частота пересечений нуля |
| `spectral_rolloff` | Спектральный rolloff |
| `mfcc_mean` | MFCC коэффициенты (тембр) |

## 🏷️ PANNs теги

Автоматическая классификация по категориям:
- **Жанры**: Rock, Pop, Jazz, Classical, Electronic, etc.
- **Инструменты**: Guitar, Piano, Drums, Violin, etc.
- **Настроения**: Happy, Sad, Energetic, Calm, etc.
- **Контекст**: Music, Speech, Applause, Silence, etc.

API документация: http://localhost:8001/docs