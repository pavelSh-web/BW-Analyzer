"""Service for tempo analysis"""

import numpy as np
from typing import Dict, Any, Optional
from ..config import DEEPRHYTHM_MIN_DURATION

# Global variables for lazy loading
_DR_OK = True
_DR_VERSION = None
_DR_MODEL = None

try:
    from deeprhythm import DeepRhythmPredictor
    _DR_VERSION = "0.0.13"
except ImportError:
    _DR_OK = False
    DeepRhythmPredictor = None

class TempoService:
    """Service for tempo analysis using DeepRhythm and librosa"""
    
    def detect_tempo_deeprhythm(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Tempo detection using DeepRhythm"""
        global _DR_MODEL
        
        if not _DR_OK or DeepRhythmPredictor is None:
            return None
        
        try:
            # Check audio duration
            import librosa
            try:
                y, sr = librosa.load(audio_path, sr=None, mono=True)
                duration = len(y) / sr
                if duration < DEEPRHYTHM_MIN_DURATION:
                    return None
            except Exception:
                return None
            
            # Model initialization (lazy loading)
            if _DR_MODEL is None:
                try:
                    _DR_MODEL = DeepRhythmPredictor()
                except Exception:
                    return None
            
            # Method 1: Try predict() without confidence
            try:
                tempo = _DR_MODEL.predict(audio_path)
                return {
                    "tempo_bpm": round(float(tempo), 1),
                    "tempo_confidence": None,
                    "tempo_method": "deeprhythm",
                    "tempo_debug": {"method": "predict_file", "duration": duration}
                }
            except Exception:
                pass
            
            # Method 2: Try predict() with confidence
            try:
                tempo, confidence = _DR_MODEL.predict(audio_path, include_confidence=True)
                return {
                    "tempo_bpm": round(float(tempo), 1),
                    "tempo_confidence": round(float(confidence), 3) if confidence is not None else None,
                    "tempo_method": "deeprhythm", 
                    "tempo_debug": {"method": "predict_file_conf", "duration": duration}
                }
            except Exception:
                pass
            
            # Method 3: Try predict_from_audio()
            try:
                tempo, confidence = _DR_MODEL.predict_from_audio(y, sr, include_confidence=True)
                return {
                    "tempo_bpm": round(float(tempo), 1),
                    "tempo_confidence": round(float(confidence), 3) if confidence is not None else None,
                    "tempo_method": "deeprhythm",
                    "tempo_debug": {"method": "predict_from_audio", "duration": duration}
                }
            except Exception:
                pass
                
        except Exception:
            pass
        
        return None
    
    def fallback_tempo(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Fallback tempo detection using librosa"""
        import librosa
        
        try:
            # librosa.beat.beat_track may fail due to scipy.signal.hann
            # Use simpler method
            oenv = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr)[0]
            return {
                "tempo_bpm": round(float(tempo), 1),
                "tempo_confidence": None,
                "tempo_method": "librosa_onset_strength",
                "tempo_debug": {"onset_envelope_size": len(oenv)}
            }
        except Exception:
            try:
                # Another method
                oenv = librosa.onset.onset_strength(y=y, sr=sr)
                tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr)[0]
                return {
                    "tempo_bpm": round(float(tempo), 1),
                    "tempo_confidence": None,
                    "tempo_method": "librosa_onset_strength",
                    "tempo_debug": {"onset_envelope_size": len(oenv)}
                }
            except Exception:
                return {
                    "tempo_bpm": 120.0,  # Default value
                    "tempo_confidence": None,
                    "tempo_method": "default",
                    "tempo_debug": {"error": "all_methods_failed"}
                }
    
    def analyze_tempo(self, 
                     audio_data: np.ndarray, 
                     sample_rate: int, 
                     audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Main tempo analysis method"""
        # Try DeepRhythm if file path is available
        if audio_path:
            dr_result = self.detect_tempo_deeprhythm(audio_path)
            if dr_result:
                return {"tempo": float(dr_result["tempo_bpm"])}
        
        # Fallback на librosa
        fb_result = self.fallback_tempo(audio_data, sample_rate)
        return {"tempo": float(fb_result["tempo_bpm"])}
