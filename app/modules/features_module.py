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
                audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Audio features analysis - only energy and brightness"""
        import librosa
        
        features = {}
        
        # RMS Energy
        rms = librosa.feature.rms(y=audio_data)[0]
        features["energy"] = round(float(np.mean(rms)), 3)
        
        # Spectral Centroid (Brightness)
        sc = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        features["brightness"] = round(float(np.mean(sc)), 1)
        
        return features
