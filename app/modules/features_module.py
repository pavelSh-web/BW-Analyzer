import numpy as np
from typing import Dict, Any, Optional
from ..core.base_module import BaseAnalysisModule
from ..types.analysis import AnalysisModule
from ..services.tempo_service import TempoService
from ..config import ENERGY_THRESHOLDS, BRIGHTNESS_THRESHOLDS

class FeaturesModule(BaseAnalysisModule):
    """Comprehensive audio features analysis module with energy, brightness, rhythm, harmony, timbre, and dynamics"""
    
    def __init__(self):
        self.tempo_service = TempoService()
    
    @property
    def module_name(self) -> AnalysisModule:
        return AnalysisModule.FEATURES
    
    def analyze(self, 
                audio_data: np.ndarray, 
                sample_rate: int, 
                audio_path: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """Comprehensive audio features analysis including energy, brightness, rhythm, harmony, timbre, and dynamics"""
        import librosa
        
        features = {}
        
        # === 0. BASIC FEATURES (Energy & Brightness) ===
        
        # RMS Energy (value)
        rms = librosa.feature.rms(y=audio_data)[0]
        energy_value = float(np.mean(rms))
        # Categorize energy with configurable thresholds
        if energy_value < ENERGY_THRESHOLDS["low"]:
            energy_cat = "low"
        elif energy_value <= ENERGY_THRESHOLDS["mid"]:
            energy_cat = "mid"
        else:
            energy_cat = "high"
        features["energy"] = energy_cat
        features["energy_value"] = round(energy_value, 3)
        
        # Spectral Centroid (Brightness): рассчитываем по активным кадрам и берём 75-й перцентиль
        sc = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        # Активные кадры по RMS: избавляемся от тишины/пауз
        rms_frames = rms  # уже посчитан выше
        rms_thresh = float(np.median(rms_frames)) * 0.5
        active_mask = rms_frames > rms_thresh
        sc_active = sc[active_mask] if np.any(active_mask) else sc
        # Значение яркости берём как 75-й перцентиль по активным кадрам
        brightness_value = float(np.percentile(sc_active, 75))
        # Нормализуем к Найквисту
        nyquist = max(sample_rate / 2.0, 1e-9)
        b_norm = brightness_value / nyquist
        # Пороговые значения (конфигурируемые): low < 0.20, mid 0.20–0.50, high > 0.50
        if b_norm < BRIGHTNESS_THRESHOLDS["low"]:
            brightness_cat = "low"
        elif b_norm <= BRIGHTNESS_THRESHOLDS["mid"]:
            brightness_cat = "mid"
        else:
            brightness_cat = "high"
        features["brightness"] = brightness_cat
        features["brightness_value"] = round(brightness_value, 1)
        
        # === 1. RHYTHM FEATURES ===
        
        # Onset density (events per second)
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sample_rate, units='frames')
        duration = len(audio_data) / sample_rate
        onset_density = len(onset_frames) / duration if duration > 0 else 0.0
        features["onset_density"] = round(float(onset_density), 3)
        
        # Percussive/Harmonic ratio via HPSS
        y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
        rms_harmonic = float(np.mean(librosa.feature.rms(y=y_harmonic)))
        rms_percussive = float(np.mean(librosa.feature.rms(y=y_percussive)))
        perc_harm_ratio = rms_percussive / (rms_harmonic + 1e-9)
        features["percussive_harmonic_ratio"] = round(perc_harm_ratio, 3)
        
        # Beat histogram stats using accurate tempo detection
        try:
            # Get accurate tempo from TempoService
            tempo_result = self.tempo_service.analyze_tempo(audio_data, sample_rate, audio_path)
            accurate_tempo = tempo_result.get("tempo", 120.0)
            
            # Use onset strength for rhythm analysis
            onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
            
            # Calculate tempo stability around the accurate tempo
            # Look for tempo variations in different parts of the track
            segment_length = len(audio_data) // 4  # 4 segments
            tempo_variations = []
            
            for i in range(4):
                start_idx = i * segment_length
                end_idx = start_idx + segment_length if i < 3 else len(audio_data)
                segment = audio_data[start_idx:end_idx]
                
                if len(segment) > sample_rate * 2:  # At least 2 seconds
                    try:
                        segment_tempo = librosa.beat.tempo(y=segment, sr=sample_rate)[0]
                        tempo_variations.append(segment_tempo)
                    except:
                        tempo_variations.append(accurate_tempo)
            
            if len(tempo_variations) > 0:
                features["beat_histogram_mean"] = round(float(np.mean(tempo_variations)), 2)
                features["beat_histogram_std"] = round(float(np.std(tempo_variations)), 2)
            else:
                features["beat_histogram_mean"] = round(accurate_tempo, 2)
                features["beat_histogram_std"] = 0.0
                
        except Exception:
            # Fallback to simple onset strength analysis
            onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
            features["beat_histogram_mean"] = 120.0
            features["beat_histogram_std"] = 0.0
        
        # === 2. HARMONY FEATURES ===
        
        # Chroma vector statistics
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        features["chroma_mean"] = [round(float(x), 3) for x in chroma_mean]
        features["chroma_std"] = [round(float(x), 3) for x in chroma_std]
        
        # Tonnetz features (harmonic network)
        tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sample_rate)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)
        features["tonnetz_mean"] = [round(float(x), 3) for x in tonnetz_mean]
        features["tonnetz_std"] = [round(float(x), 3) for x in tonnetz_std]
        
        # Key clarity (sharpness of chroma distribution)
        # Higher values = clearer key, lower values = atonal/ambiguous
        chroma_profile = np.mean(chroma, axis=1)
        key_clarity = float(np.max(chroma_profile) - np.mean(chroma_profile))
        features["key_clarity"] = round(key_clarity, 3)
        
        # === 3. TIMBRE/SPECTRAL FEATURES ===
        
        # Spectral flatness (noisiness vs tonality)
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)
        features["spectral_flatness_mean"] = round(float(np.mean(spectral_flatness)), 4)
        features["spectral_flatness_std"] = round(float(np.std(spectral_flatness)), 4)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
        features["spectral_bandwidth_mean"] = round(float(np.mean(spectral_bandwidth)), 1)
        features["spectral_bandwidth_std"] = round(float(np.std(spectral_bandwidth)), 1)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        features["zero_crossing_rate_mean"] = round(float(np.mean(zcr)), 4)
        features["zero_crossing_rate_std"] = round(float(np.std(zcr)), 4)
        
        # === 4. DYNAMICS FEATURES ===
        
        # RMS energy contour for dynamics
        rms = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=512)[0]
        
        # Dynamic range (difference between max and min RMS in dB)
        rms_db = 20 * np.log10(rms + 1e-9)
        dynamic_range = float(np.max(rms_db) - np.min(rms_db))
        features["dynamic_range_db"] = round(dynamic_range, 2)
        
        # Loudness contour statistics
        features["loudness_mean"] = round(float(np.mean(rms_db)), 2)
        features["loudness_std"] = round(float(np.std(rms_db)), 2)
        features["loudness_min"] = round(float(np.min(rms_db)), 2)
        features["loudness_max"] = round(float(np.max(rms_db)), 2)
        
        # Loudness range (difference between 95th and 5th percentile)
        loudness_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
        features["loudness_range"] = round(loudness_range, 2)
        
        # Try to calculate LUFS if pyloudnorm is available
        try:
            import pyloudnorm as pyln
            
            # Normalize audio for LUFS measurement
            meter = pyln.Meter(sample_rate)
            # Ensure audio is in the right format for pyloudnorm
            if len(audio_data.shape) == 1:
                audio_for_lufs = audio_data.reshape(-1, 1)
            else:
                audio_for_lufs = audio_data
            
            loudness_lufs = meter.integrated_loudness(audio_for_lufs)
            if not np.isnan(loudness_lufs) and not np.isinf(loudness_lufs):
                features["lufs"] = round(float(loudness_lufs), 2)
            else:
                features["lufs"] = None
        except Exception:
            # If pyloudnorm fails or not available, skip LUFS
            features["lufs"] = None
        
        return features
