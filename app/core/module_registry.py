from typing import Dict, Type, List, Optional
import numpy as np
from .base_module import BaseAnalysisModule
from ..types.analysis import AnalysisModule

class ModuleRegistry:
    """Analysis modules registry"""
    
    def __init__(self):
        self._modules: Dict[AnalysisModule, BaseAnalysisModule] = {}
    
    def register(self, module: BaseAnalysisModule):
        """Register module"""
        self._modules[module.module_name] = module
    
    def get_module(self, module_name: AnalysisModule) -> Optional[BaseAnalysisModule]:
        """Get module by name"""
        return self._modules.get(module_name)
    
    def get_available_modules(self) -> List[AnalysisModule]:
        """Get list of available modules"""
        return list(self._modules.keys())
    
    def analyze_with_modules(self, 
                           audio_data: np.ndarray, 
                           sample_rate: int, 
                           audio_path: Optional[str],
                           requested_modules: List[AnalysisModule],
                           module_kwargs: Optional[Dict[str, Dict]] = None) -> Dict[str, any]:
        """Analysis with specified modules"""
        results = {}
        
        for module_name in requested_modules:
            module = self.get_module(module_name)
            if module and module.can_analyze(audio_data, sample_rate):
                try:
                    # Get module-specific kwargs
                    kwargs = module_kwargs.get(module_name.value, {}) if module_kwargs else {}
                    
                    # Call analyze with kwargs
                    results[module_name.value] = module.analyze(
                        audio_data, sample_rate, audio_path, **kwargs
                    )
                except Exception as e:
                    print(f"Error in module {module_name}: {e}")
                    results[module_name.value] = None
            else:
                results[module_name.value] = None
                
        return results
