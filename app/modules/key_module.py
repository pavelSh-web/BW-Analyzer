import numpy as np
from typing import Dict, Any, Optional
from ..core.base_module import BaseAnalysisModule
from ..types.analysis import AnalysisModule
from ..services.key_service import KeyService

class KeyModule(BaseAnalysisModule):
    """Key analysis module using Skey and librosa"""
    
    def __init__(self):
        self.service = KeyService()
    
    @property
    def module_name(self) -> AnalysisModule:
        return AnalysisModule.KEY
    
    def analyze(self, 
                audio_data: np.ndarray, 
                sample_rate: int, 
                audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Key analysis"""
        return self.service.analyze_key(audio_data, sample_rate, audio_path)
