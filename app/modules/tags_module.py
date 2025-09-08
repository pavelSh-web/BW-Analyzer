import numpy as np
from typing import Dict, Any, Optional
from ..core.base_module import BaseAnalysisModule
from ..types.analysis import AnalysisModule
from ..services.panns_service import PANNsService

class TagsModule(BaseAnalysisModule):
    """Tags analysis module using PANNs"""
    
    def __init__(self):
        # Don't create service immediately to avoid caching issues
        self.service = None
    
    @property
    def module_name(self) -> AnalysisModule:
        return AnalysisModule.TAGS
    
    def analyze(self, 
                audio_data: np.ndarray, 
                sample_rate: int, 
                audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Tags analysis using PANNs"""
        if not audio_path:
            return None
        
        # Create fresh service for each request to avoid caching issues
        service = PANNsService()
        return service.get_audio_tags(audio_path)
