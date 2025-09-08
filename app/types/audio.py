from typing import Optional
from pydantic import BaseModel

class AudioFeatures(BaseModel):
    energy: float
    brightness: float

class MusicalFeatures(BaseModel):
    tempo: Optional[float] = None
    key: Optional[str] = None
