"""Service for key analysis"""

import numpy as np
from typing import Dict, Any, Optional

class KeyService:
    """Service for key analysis using Skey and librosa"""
    
    def detect_key_skey(self, audio_path: str) -> Optional[str]:
        """Key detection using Skey"""
        try:
            from skey import detect_key
            skey_result = detect_key(audio_path, device='cpu')
            if skey_result and len(skey_result) > 0:
                return skey_result[0]
            return None
        except Exception:
            return None
    
    def detect_key_librosa(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Fallback key detection using librosa"""
        try:
            import librosa
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate, n_chroma=12)
            cm = np.mean(chroma, axis=1)
            maj = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            mino = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            maj_scores = [np.dot(cm, np.roll(maj, i)) for i in range(12)]
            min_scores = [np.dot(cm, np.roll(mino, i)) for i in range(12)]
            bi_maj = int(np.argmax(maj_scores))
            bi_min = int(np.argmax(min_scores))
            names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            return (names[bi_maj] + ' major' if maj_scores[bi_maj] > min_scores[bi_min] 
                   else names[bi_min] + ' minor')
        except Exception:
            return "Unknown"
    
    def analyze_key(self, 
                   audio_data: np.ndarray, 
                   sample_rate: int, 
                   audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Main key analysis method"""
        if audio_path:
            # Try Skey
            skey_result = self.detect_key_skey(audio_path)
            if skey_result:
                return {"key": skey_result}
        
        # Fallback to librosa
        librosa_result = self.detect_key_librosa(audio_data, sample_rate)
        return {"key": librosa_result}
