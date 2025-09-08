import numpy as np
from typing import Dict, Any, Optional
from ..core.base_module import BaseAnalysisModule
from ..types.analysis import AnalysisModule
from ..services.tempo_service import TempoService

class TempoModule(BaseAnalysisModule):
    """Tempo analysis module using DeepRhythm and librosa"""
    
    def __init__(self):
        self.service = TempoService()
    
    @property
    def module_name(self) -> AnalysisModule:
        return AnalysisModule.TEMPO
    
    def analyze(self, 
                audio_data: np.ndarray, 
                sample_rate: int, 
                audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Tempo analysis"""
        return self.service.analyze_tempo(audio_data, sample_rate, audio_path)
