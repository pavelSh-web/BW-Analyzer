from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from enum import Enum

class AnalysisModule(str, Enum):
    TAGS = "tags"
    TEMPO = "tempo" 
    KEY = "key"
    FEATURES = "features"

class AnalysisRequest(BaseModel):
    modules: List[AnalysisModule] = [
        AnalysisModule.TAGS,
        AnalysisModule.TEMPO,
        AnalysisModule.KEY,
        AnalysisModule.FEATURES,
    ]

class AnalysisResponse(BaseModel):
    tempo: Optional[float] = None
    key: Optional[str] = None
    tags: Optional[Dict[str, List[Dict[str, Any]]]] = None
    energy: Optional[float] = None
    brightness: Optional[float] = None
