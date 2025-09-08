import numpy as np
from typing import Dict, Any, Optional
from ..core.base_module import BaseAnalysisModule
from ..types.analysis import AnalysisModule

class FeaturesModule(BaseAnalysisModule):
    """Audio features analysis module"""
    
    @property
    def module_name(self) -> AnalysisModule:
        return AnalysisModule.FEATURES
    
    def analyze(self, 
                audio_data: np.ndarray, 
                sample_rate: int, 
                audio_path: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """Audio features analysis - only energy and brightness"""
        import librosa
        
        features = {}
        
        # RMS Energy (value)
        rms = librosa.feature.rms(y=audio_data)[0]
        energy_value = float(np.mean(rms))
        # Categorize energy with fixed thresholds
        if energy_value < 0.10:
            energy_cat = "low"
        elif energy_value <= 0.22:
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
        # Пороговые значения (подкручены для практики): low < 0.20, mid 0.20–0.50, high > 0.50
        if b_norm < 0.20:
            brightness_cat = "low"
        elif b_norm <= 0.50:
            brightness_cat = "mid"
        else:
            brightness_cat = "high"
        features["brightness"] = brightness_cat
        features["brightness_value"] = round(brightness_value, 1)
        
        return features
