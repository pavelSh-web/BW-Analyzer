from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from ..types.analysis import AnalysisModule

class BaseAnalysisModule(ABC):
    """Base class for all audio analysis modules"""
    
    @property
    @abstractmethod
    def module_name(self) -> AnalysisModule:
        """Module name"""
        pass
    
    @abstractmethod
    def analyze(self, 
                audio_data: np.ndarray, 
                sample_rate: int, 
                audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Main analysis method"""
        pass
    
    def can_analyze(self, audio_data: np.ndarray, sample_rate: int) -> bool:
        """Check if analysis is possible (e.g., minimum duration)"""
        return True
