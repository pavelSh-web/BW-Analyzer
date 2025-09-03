import os, tempfile, time
from typing import List, Optional, Dict, Any

import numpy as np
import librosa
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse


# ---------------------------
# PANNs (только для тегов)
# ---------------------------
try:
    from panns_inference import AudioTagging, labels as PANN_LABELS
    import torch
    _PANN_OK = True
except Exception:
    _PANN_OK = False
    AudioTagging = None
    PANN_LABELS = []

_PANN_MODEL: Optional["AudioTagging"] = None

# Список 125 самых важных музыкальных тегов для анализа
IMPORTANT_TAG_GROUPS = {
    "music": [
        "Pop music", "Rock music", "Hip hop music", "Jazz", "Blues", "Country music", 
        "Electronic music", "Classical music", "Folk music", "Reggae", "R&B", "Soul music",
        "Gospel music", "Christian music", "Ambient music", "Techno", "House music", "Trance music",
        "Heavy metal", "Punk rock", "Alternative rock", "Indie rock", "Funk", "Disco", "Ska",
        "New-age music", "World music", "Latin music", "Bossa nova", "Salsa music", "Flamenco",
        "Opera", "Musical theatre", "Baroque music", "Romantic music", "Impressionist music",
        "Minimalist music", "Experimental music", "Avant-garde music", "Progressive rock",
        "Psychedelic rock", "Grunge", "Emo", "Hardcore punk", "Death metal", "Black metal",
        "Thrash metal", "Power metal", "Symphonic metal", "Industrial music", "Dubstep",
        "Drum and bass", "Breakbeat", "Garage music", "UK garage", "2-step garage",
        "Trap music", "Future bass", "Chillout", "Downtempo", "Trip hop", "IDM", "Glitch",
        "Synthwave", "Vaporwave", "Lo-fi", "Drill music", "Afrobeat", "Highlife", "Kwaito",
        "Bhangra", "Bollywood music", "K-pop", "J-pop", "Anime music", "Video game music",
        "Film score", "Soundtrack", "Instrumental", "Acoustic music", "Unplugged",
        "Rhythm and blues", "Swing music", "Carnatic music", "Middle Eastern music", 
        "Music of Africa", "Music of Asia", "Music of Bollywood", "Music of Latin America",
        "Dance music", "Independent music", "Traditional music",
    ],
    "instruments": [
        # Основные инструменты
        "Guitar", "Electric guitar", "Acoustic guitar", "Bass guitar", "Piano", "Electric piano",
        "Drum", "Drum kit", "Snare drum", "Bass drum", "Hi-hat", "Cymbal",
        
        # Клавишные и синтезаторы
        "Synthesizer", "Keyboard (musical)", "Organ", "Electronic organ", "Harpsichord",
        
        # Струнные
        "Violin", "Cello",
        
        # Духовые
        "Saxophone", "Flute",
        
        # Ударные и перкуссия
        "Percussion", "Bell",
        
        # Этнические и экзотические
        "Sitar", "Tabla", "Didgeridoo", "Bagpipes", "Accordion"
    ],
    "vocal": [
        "Singing", "Vocal", "Choir", "A cappella", "Rapping", "Scat singing", "Whistling",
        "Beatboxing", "Backing vocals"
    ]
}

IMPORTANT_TAGS = (
    IMPORTANT_TAG_GROUPS["music"] +
    IMPORTANT_TAG_GROUPS["instruments"] +
    IMPORTANT_TAG_GROUPS["vocal"]
)

def _pann_model() -> Optional["AudioTagging"]:
    global _PANN_MODEL
    if not _PANN_OK:
        return None
    if _PANN_MODEL is None:
        device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
        _PANN_MODEL = AudioTagging(device=device, checkpoint_path=None)
    return _PANN_MODEL

def get_audio_tags(audio_path: str, topk_per_group: int = 5) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Получить топ-N тегов от PANNs по каждой группе отдельно"""
    model = _pann_model()
    if model is None:
        return None
    
    try:
        y, sr = librosa.load(audio_path, sr=32000, mono=True)
        y_batch = y[np.newaxis, :]  # (1, T)
        result = model.inference(y_batch)
        
        if isinstance(result, tuple) and len(result) == 2:
            clipwise_probs, _ = result
            probs = clipwise_probs[0] if clipwise_probs.ndim > 1 else clipwise_probs
            
            result_groups = {}
            
            # Анализируем каждую группу отдельно
            for group_name, group_tags in IMPORTANT_TAG_GROUPS.items():
                # Находим индексы тегов этой группы
                group_indices = []
                group_labels = []
                
                for i, label in enumerate(PANN_LABELS):
                    if label in group_tags:
                        group_indices.append(i)
                        group_labels.append(label)
                
                if not group_indices:
                    result_groups[group_name] = []
                    continue
                
                # Получаем вероятности только для тегов этой группы
                group_probs = probs[group_indices]
                
                # Сортируем по убыванию вероятности
                sorted_indices = np.argsort(-group_probs)
                
                # Берем топ-N для этой группы
                top_indices = sorted_indices[:topk_per_group]
                
                result_groups[group_name] = [
                    {
                        "label": group_labels[i], 
                        "prob": float(group_probs[i])
                    } 
                    for i in top_indices
                ]
            
            return result_groups
        return None
    except Exception as e:
        print(f"PANNs error: {e}")
        return None



# ---------------------------
# Музыкальные характеристики
# ---------------------------
def analyze_audio_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Анализ музыкальных характеристик трека с улучшенными алгоритмами"""
    features = {}
    
    try:
        # === ТЕМП (улучшенный алгоритм) ===
        # Используем несколько методов и выбираем лучший
        tempo_methods = []
        
        # Метод 1: librosa.beat.tempo (по умолчанию)
        tempo1, _ = librosa.beat.beat_track(y=y, sr=sr, units='time')
        tempo_methods.append(('librosa_default', tempo1))
        
        # Метод 2: с другими параметрами
        tempo2, _ = librosa.beat.beat_track(y=y, sr=sr, units='time', 
                                           start_bpm=60)
        tempo_methods.append(('librosa_wide', tempo2))
        
        # Метод 3: через onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        if len(onset_frames) > 1:
            intervals = np.diff(onset_frames)
            # Убираем выбросы (интервалы больше 2 секунд)
            intervals = intervals[intervals < 2.0]
            if len(intervals) > 0:
                avg_interval = np.median(intervals)
                tempo3 = 60.0 / avg_interval
                tempo_methods.append(('onset_based', tempo3))
        
        # Выбираем темп ближе к разумному диапазону (60-200 BPM)
        valid_tempos = [(name, t) for name, t in tempo_methods if 60 <= t <= 200]
        if valid_tempos:
            # Предпочитаем темпы в диапазоне 80-160 BPM
            preferred_tempos = [(name, t) for name, t in valid_tempos if 80 <= t <= 160]
            if preferred_tempos:
                tempo_method, tempo_bpm = preferred_tempos[0]
            else:
                tempo_method, tempo_bpm = valid_tempos[0]
        else:
            tempo_method, tempo_bpm = tempo_methods[0]
        
        features["tempo_bpm"] = round(tempo_bpm, 1)
        features["tempo_method"] = tempo_method
        
        # === ТОНАЛЬНОСТЬ (улучшенный алгоритм) ===
        # Используем несколько методов
        key_methods = []
        
        # Метод 1: librosa.key (по умолчанию)
        try:
            key1 = librosa.key.estimate_key(y, sr)
            key_methods.append(('librosa_default', key1))
        except:
            pass
        
        # Метод 2: через chroma с разными параметрами
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
            chroma_mean = np.mean(chroma, axis=1)
            # Находим доминирующую ноту
            dominant_note = np.argmax(chroma_mean)
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key2 = note_names[dominant_note]
            key_methods.append(('chroma_dominant', key2))
        except:
            pass
        
        # Метод 3: через chroma с весами для мажор/минор (улучшенный)
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Более точные веса для мажорных и минорных тональностей
            # C major: C D E F G A B
            major_weights = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # C major
            # A minor: A B C D E F G
            minor_weights = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])  # A minor
            
            major_scores = []
            minor_scores = []
            
            for i in range(12):
                # Поворачиваем веса для каждой тональности
                major_rotated = np.roll(major_weights, i)
                minor_rotated = np.roll(minor_weights, i)
                
                major_score = np.dot(chroma_mean, major_rotated)
                minor_score = np.dot(chroma_mean, minor_rotated)
                
                major_scores.append(major_score)
                minor_scores.append(minor_score)
            
            # Находим лучшую мажорную и минорную тональности
            best_major_idx = np.argmax(major_scores)
            best_minor_idx = np.argmax(minor_scores)
            
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            if major_scores[best_major_idx] > minor_scores[best_minor_idx]:
                key3 = note_names[best_major_idx] + ' major'
            else:
                key3 = note_names[best_minor_idx] + ' minor'
            
            key_methods.append(('chroma_weighted', key3))
        except:
            pass
        
        # Метод 4: через chroma с CQT (Constant-Q Transform) - более точный
        try:
            chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
            chroma_cqt_mean = np.mean(chroma_cqt, axis=1)
            
            # Используем те же веса
            major_weights = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            minor_weights = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            
            major_scores = []
            minor_scores = []
            
            for i in range(12):
                major_rotated = np.roll(major_weights, i)
                minor_rotated = np.roll(minor_weights, i)
                
                major_score = np.dot(chroma_cqt_mean, major_rotated)
                minor_score = np.dot(chroma_cqt_mean, minor_rotated)
                
                major_scores.append(major_score)
                minor_scores.append(minor_score)
            
            best_major_idx = np.argmax(major_scores)
            best_minor_idx = np.argmax(minor_scores)
            
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            if major_scores[best_major_idx] > minor_scores[best_minor_idx]:
                key4 = note_names[best_major_idx] + ' major'
            else:
                key4 = note_names[best_minor_idx] + ' minor'
            
            key_methods.append(('chroma_cqt', key4))
        except:
            pass
        
        # Метод 5: через chroma с CENS (Chroma Energy Normalized Statistics)
        try:
            chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=12)
            chroma_cens_mean = np.mean(chroma_cens, axis=1)
            
            major_weights = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            minor_weights = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            
            major_scores = []
            minor_scores = []
            
            for i in range(12):
                major_rotated = np.roll(major_weights, i)
                minor_rotated = np.roll(minor_weights, i)
                
                major_score = np.dot(chroma_cens_mean, major_rotated)
                minor_score = np.dot(chroma_cens_mean, minor_rotated)
                
                major_scores.append(major_score)
                minor_scores.append(minor_score)
            
            best_major_idx = np.argmax(major_scores)
            best_minor_idx = np.argmax(minor_scores)
            
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            if major_scores[best_major_idx] > minor_scores[best_minor_idx]:
                key5 = note_names[best_major_idx] + ' major'
            else:
                key5 = note_names[best_minor_idx] + ' minor'
            
            key_methods.append(('chroma_cens', key5))
        except:
            pass
        
        # Метод 6: Улучшенный анализ через гармонические частоты
        try:
            # Используем более точный анализ гармоник
            chroma_harmonic = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096, hop_length=512)
            chroma_harmonic_mean = np.mean(chroma_harmonic, axis=1)
            
            # Более точные веса для мажорных и минорных тональностей
            # C major: C D E F G A B (1 0 1 0 1 1 0 1 0 1 0 1)
            major_weights = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            # A minor: A B C D E F G (1 0 1 1 0 1 0 1 1 0 1 0)
            minor_weights = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            
            major_scores = []
            minor_scores = []
            
            for i in range(12):
                major_rotated = np.roll(major_weights, i)
                minor_rotated = np.roll(minor_weights, i)
                
                major_score = np.dot(chroma_harmonic_mean, major_rotated)
                minor_score = np.dot(chroma_harmonic_mean, minor_rotated)
                
                major_scores.append(major_score)
                minor_scores.append(minor_score)
            
            best_major_idx = np.argmax(major_scores)
            best_minor_idx = np.argmax(minor_scores)
            
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            if major_scores[best_major_idx] > minor_scores[best_minor_idx]:
                key6 = note_names[best_major_idx] + ' major'
            else:
                key6 = note_names[best_minor_idx] + ' minor'
            
            key_methods.append(('chroma_harmonic', key6))
        except:
            pass
        
        # Метод 7: Анализ через спектральные пики
        try:
            # Ищем доминирующие частоты
            fft = np.fft.fft(y)
            freqs = np.fft.fftfreq(len(y), 1/sr)
            
            # Берем только положительные частоты
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            # Ищем пики в спектре
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(positive_fft, height=np.max(positive_fft)*0.1)
            
            if len(peaks) > 0:
                # Находим доминирующую частоту
                dominant_freq = positive_freqs[peaks[np.argmax(positive_fft[peaks])]]
                
                # Конвертируем частоту в ноту
                # A4 = 440 Hz
                a4_freq = 440.0
                note_offset = 12 * np.log2(dominant_freq / a4_freq)
                note_index = int(round(note_offset)) % 12
                
                note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
                dominant_note = note_names[note_index]
                
                key_methods.append(('spectral_peaks', dominant_note))
        except:
            pass
        
        # Выбираем лучшую тональность (улучшенный алгоритм)
        if key_methods:
            # Подсчитываем голоса за каждую тональность
            key_votes = {}
            method_weights = {
                'chroma_harmonic': 3,
                'chroma_cqt': 2.5,
                'chroma_cens': 2.5,
                'chroma_weighted': 2,
                'librosa_default': 1.5,
                'spectral_peaks': 1,
                'chroma_dominant': 0.5
            }
            
            for method, key in key_methods:
                weight = method_weights.get(method, 1)
                if key in key_votes:
                    key_votes[key] += weight
                else:
                    key_votes[key] = weight
            
            # Выбираем тональность с наибольшим весом
            best_key = max(key_votes, key=key_votes.get)
            
            # Находим метод, который дал эту тональность
            for method, key in key_methods:
                if key == best_key:
                    key_method, key = method, key
                    break
        else:
            key_method, key = 'unknown', 'Unknown'
        
        features["key"] = key
        features["key_method"] = key_method
        
        # === РАЗМЕР (кардинально улучшенный алгоритм) ===
        # Анализируем акценты и паттерны ударений
        time_sig_methods = []
        
        # Метод 1: Анализ акцентированных битов через RMS
        try:
            # Разбиваем на фреймы по 1 секунде
            frame_length = sr
            hop_length = frame_length // 4
            
            rms_frames = []
            for i in range(0, len(y) - frame_length, hop_length):
                frame = y[i:i + frame_length]
                rms = np.sqrt(np.mean(frame**2))
                rms_frames.append(rms)
            
            if len(rms_frames) > 8:
                # Ищем паттерны акцентов (каждый 4-й бит сильнее в 4/4)
                rms_array = np.array(rms_frames)
                
                # Нормализуем RMS
                rms_norm = (rms_array - np.mean(rms_array)) / np.std(rms_array)
                
                # Ищем периодичность в акцентах
                # Для 4/4: акценты каждые 4 бита
                # Для 3/4: акценты каждые 3 бита
                
                # Проверяем 4/4 (акценты каждые 4 фрейма)
                accent_4_4 = 0
                for i in range(0, len(rms_norm) - 4, 4):
                    if rms_norm[i] > np.mean(rms_norm[i:i+4]) + 0.5:
                        accent_4_4 += 1
                
                # Проверяем 3/4 (акценты каждые 3 фрейма)
                accent_3_4 = 0
                for i in range(0, len(rms_norm) - 3, 3):
                    if rms_norm[i] > np.mean(rms_norm[i:i+3]) + 0.5:
                        accent_3_4 += 1
                
                # Выбираем размер с большим количеством акцентов
                if accent_4_4 > accent_3_4:
                    time_sig_methods.append(('accent_analysis', '4/4'))
                else:
                    time_sig_methods.append(('accent_analysis', '3/4'))
        except:
            pass
        
        # Метод 2: Анализ через onset detection с группировкой
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            if len(onset_frames) > 12:
                # Группируем onset'ы по тактам
                onset_intervals = np.diff(onset_frames)
                
                # Ищем регулярные паттерны
                # Для 4/4: 4 сильных onset'а на такт
                # Для 3/4: 3 сильных onset'а на такт
                
                # Анализируем интервалы между onset'ами
                avg_interval = np.median(onset_intervals)
                
                # Группируем onset'ы по тактам
                beats_per_measure_4_4 = 0
                beats_per_measure_3_4 = 0
                
                for i in range(0, len(onset_intervals) - 3, 4):
                    if i + 3 < len(onset_intervals):
                        # Проверяем паттерн 4/4
                        if (onset_intervals[i] < avg_interval * 1.5 and 
                            onset_intervals[i+1] < avg_interval * 1.5 and
                            onset_intervals[i+2] < avg_interval * 1.5 and
                            onset_intervals[i+3] > avg_interval * 1.5):
                            beats_per_measure_4_4 += 1
                
                for i in range(0, len(onset_intervals) - 2, 3):
                    if i + 2 < len(onset_intervals):
                        # Проверяем паттерн 3/4
                        if (onset_intervals[i] < avg_interval * 1.5 and 
                            onset_intervals[i+1] < avg_interval * 1.5 and
                            onset_intervals[i+2] > avg_interval * 1.5):
                            beats_per_measure_3_4 += 1
                
                if beats_per_measure_4_4 > beats_per_measure_3_4:
                    time_sig_methods.append(('onset_grouping', '4/4'))
                else:
                    time_sig_methods.append(('onset_grouping', '3/4'))
        except:
            pass
        
        # Метод 3: Анализ через спектральные характеристики
        try:
            # Анализируем спектральные изменения каждые 4 и 3 бита
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=0)
            
            # Ищем периодичность в спектральных изменениях
            # Для 4/4: изменения каждые 4 фрейма
            # Для 3/4: изменения каждые 3 фрейма
            
            chroma_diff = np.diff(chroma_mean)
            chroma_diff_abs = np.abs(chroma_diff)
            
            # Проверяем периодичность 4/4
            period_4_4 = 0
            for i in range(0, len(chroma_diff_abs) - 4, 4):
                if np.std(chroma_diff_abs[i:i+4]) > np.mean(chroma_diff_abs):
                    period_4_4 += 1
            
            # Проверяем периодичность 3/4
            period_3_4 = 0
            for i in range(0, len(chroma_diff_abs) - 3, 3):
                if np.std(chroma_diff_abs[i:i+3]) > np.mean(chroma_diff_abs):
                    period_3_4 += 1
            
            if period_4_4 > period_3_4:
                time_sig_methods.append(('spectral_periodicity', '4/4'))
            else:
                time_sig_methods.append(('spectral_periodicity', '3/4'))
        except:
            pass
        
        # Метод 4: Простой анализ через ударные (kick drum detection)
        try:
            # Ищем низкочастотные компоненты (kick drum)
            # Kick drum обычно в диапазоне 60-80 Hz
            low_freq = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)[0]
            
            # Фильтруем только низкие частоты
            low_freq_filtered = low_freq[low_freq < 200]
            
            if len(low_freq_filtered) > 8:
                # Ищем периодичность в низкочастотных компонентах
                # Для 4/4: kick каждые 4 бита
                # Для 3/4: kick каждые 3 бита
                
                # Простая эвристика: если много низкочастотных пиков, то 4/4
                low_freq_peaks = 0
                for i in range(0, len(low_freq_filtered) - 4, 4):
                    if np.std(low_freq_filtered[i:i+4]) > np.mean(low_freq_filtered):
                        low_freq_peaks += 1
                
                if low_freq_peaks > len(low_freq_filtered) // 8:
                    time_sig_methods.append(('kick_drum_analysis', '4/4'))
                else:
                    time_sig_methods.append(('kick_drum_analysis', '3/4'))
        except:
            pass
        
        # Метод 5: Анализ через темп (простая эвристика)
        try:
            # Если темп очень медленный (< 60 BPM), вероятно 3/4
            # Если темп нормальный (60-180 BPM), вероятно 4/4
            if tempo_bpm < 60:
                time_sig_methods.append(('tempo_heuristic', '3/4'))
            else:
                time_sig_methods.append(('tempo_heuristic', '4/4'))
        except:
            pass
        
        # Выбираем размер (предпочитаем 4/4 как наиболее распространенный)
        if time_sig_methods:
            # Подсчитываем голоса за каждый размер
            votes_4_4 = sum(1 for method, sig in time_sig_methods if sig == '4/4')
            votes_3_4 = sum(1 for method, sig in time_sig_methods if sig == '3/4')
            

            
            # Требуем явного большинства для 3/4 (3/4 встречается реже)
            if votes_3_4 > votes_4_4 * 1.5:  # 3/4 должно быть значительно больше
                time_sig_method, time_signature = 'majority_vote', '3/4'
            else:
                time_sig_method, time_signature = 'majority_vote', '4/4'
        else:
            time_sig_method, time_signature = 'default', '4/4'
        
        features["time_signature"] = time_signature
        features["time_sig_method"] = time_sig_method
            
        # === ОСТАЛЬНЫЕ ХАРАКТЕРИСТИКИ ===
        # Энергия
        rms = librosa.feature.rms(y=y)[0]
        features["energy"] = round(float(np.mean(rms)), 3)
        
        # Спектральный центроид (яркость)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["brightness"] = round(float(np.mean(spectral_centroids)), 1)
        
        # Ноль-кроссинги (индикатор перкуссии/шума)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zero_crossing_rate"] = round(float(np.mean(zcr)), 3)
        
        # Спектральный rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features["spectral_rolloff"] = round(float(np.mean(rolloff)), 1)
        
        # MFCC для тембральных характеристик
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features["mfcc_mean"] = [round(float(x), 3) for x in np.mean(mfccs, axis=1)]
        
    except Exception as e:
        print(f"Feature analysis error: {e}")
        features["error"] = str(e)
        
    return features


# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI(title="Audio Tags & Features API", version="1.0.0")


@app.post("/analyze")
async def analyze_track(
    file: UploadFile = File(...),
    top_tags_per_group: int = 5
):
    """
    Анализирует аудиофайл и возвращает:
    - PANNs теги по группам (музыкальные жанры, инструменты, вокал)
    - Музыкальные характеристики (темп, тональность, размер, etc.)
    
    После анализа все временные файлы удаляются.
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        raise HTTPException(400, "Unsupported file type")
    
    t0 = time.time()
    temp_path = None
    
    try:
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(file.filename)[1] or ".wav", 
            delete=False
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name
        
        # Загружаем аудио
        y, sr = librosa.load(temp_path, sr=None, mono=True)
        
        # Получаем теги от PANNs по группам
        tags = get_audio_tags(temp_path, topk_per_group=top_tags_per_group)
        
        # Анализируем музыкальные характеристики
        features = analyze_audio_features(y, sr)
        
        # Длительность трека
        duration = float(len(y) / sr)
        
        result = {
            "filename": file.filename,
            "duration_seconds": duration,
            "tags": tags,
            "musical_features": features,
            "elapsed_sec": round(time.time() - t0, 3)
        }
        
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
        
    finally:
        # ОБЯЗАТЕЛЬНО удаляем временный файл
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@app.get("/")
def root():
    return {
        "service": "Audio Tags & Features API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Анализ аудио: PANNs теги + музыкальные характеристики",
        },
        "features": [
            "PANNs audio tagging (527 classes)",
            "Tempo (BPM) detection", 
            "Key detection",
            "Time signature estimation",
            "Energy, brightness, spectral features",
            "No persistent storage - temporary analysis only"
        ]
    }
@app.get("/tags")
def get_important_tag_groups():
    """Получить группы музыкальных тегов"""
    return {
        "total_tags": sum(len(tags) for tags in IMPORTANT_TAG_GROUPS.values()),
        "categories": {key: len(tags) for key, tags in IMPORTANT_TAG_GROUPS.items()},
        "tags": IMPORTANT_TAG_GROUPS
    }
