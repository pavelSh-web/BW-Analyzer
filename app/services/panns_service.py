"""Service for working with PANNs (audio tagging)"""

import os
import numpy as np
from typing import Dict, List, Any, Optional
from ..config import BW_TAG_GROUPS, DISPLAY_ALIASES
from .tag_normalization_service import TagNormalizationService

class PANNsService:
    """Service for audio tag analysis using PANNs"""
    
    def __init__(self):
        self._model = None
        self._device = None
        self._normalization_service = TagNormalizationService()
        self._load_model()
    
    def _load_model(self):
        """Lazy loading of PANNs model"""
        try:
            import torch
            import panns_inference
            from panns_inference import AudioTagging
            
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._model = AudioTagging(checkpoint_path=None, device=self._device)
            print(f"PANNs model loaded on {self._device}")
        except Exception as e:
            print(f"Failed to load PANNs model: {e}")
            self._model = None
    
    def canonize_label(self, label: str) -> Optional[str]:
        """Canonicalize labels"""
        # Create reverse dictionary for finding canonical names
        label_to_groups: Dict[str, List[str]] = {}
        for group_name, labels in BW_TAG_GROUPS.items():
            for lbl in labels:
                if lbl not in label_to_groups:
                    label_to_groups[lbl] = []
                label_to_groups[lbl].append(group_name)
        
        # Direct match
        if label in label_to_groups:
            return label
            
        # Partial match search
        label_lower = label.lower()
        
        # Skip too generic terms that would match many genres
        generic_terms = {"music", "sound", "audio", "song"}
        if label_lower in generic_terms:
            print(f"[canonize] skipped generic term: '{label}'")
            return None
        
        for canonical in label_to_groups.keys():
            if canonical.lower() == label_lower:
                return canonical
            if label_lower in canonical.lower() or canonical.lower() in label_lower:
                # Additional check to avoid false positives
                if len(label_lower) > 3 and len(canonical.lower()) > 3:
                    return canonical
        
        # Special cases
        if "electronic" in label_lower:
            return "Electronic dance music"
        if "hip" in label_lower and "hop" in label_lower:
            return "Hip hop music"
        if "classical" in label_lower:
            return "Classical music"
        
        print(f"[canonize] dropped not-in-PANN label: '{label}'")
        return None
    
    def prettify_label(self, label_raw: str, group_name: Optional[str] = None) -> str:
        """Convert label to pretty format"""
        if label_raw in DISPLAY_ALIASES:
            return DISPLAY_ALIASES[label_raw]
        if label_raw.endswith(" music"):
            return label_raw[:-6]
        if "," in label_raw:
            return label_raw.split(",", 1)[0].strip()
        return label_raw
    
    def get_audio_tags(self, audio_path: str, normalize: bool = True, normalization_opts: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get tags for audio file"""
        if not self._model:
            print("PANNs model not loaded")
            return None
            
        try:
            # Load audio using librosa
            import librosa
            audio, sr = librosa.load(audio_path, sr=32000, mono=True)
            
            # PANNs expects (batch_size, channels)
            audio_batch = audio.reshape(1, -1)
            
            clipwise_output, embedding = self._model.inference(audio_batch)
            
            # Create reverse dictionary
            label_to_groups: Dict[str, List[str]] = {}
            for group_name, labels in BW_TAG_GROUPS.items():
                for lbl in labels:
                    if lbl not in label_to_groups:
                        label_to_groups[lbl] = []
                    label_to_groups[lbl].append(group_name)
            
            # Group results
            items_by_group: Dict[str, List[tuple]] = {k: [] for k in BW_TAG_GROUPS.keys()}
            
            for label, prob in zip(self._model.labels, clipwise_output[0]):
                canonical = self.canonize_label(label)
                if canonical and canonical in label_to_groups:
                    p = float(prob)
                    for gname in label_to_groups[canonical]:
                        items_by_group[gname].append((canonical, p))
            
            # Form final result with deduplication
            out: Dict[str, List[Dict[str, Any]]] = {}
            for gname, items in items_by_group.items():
                items.sort(key=lambda x: x[1], reverse=True)
                
                # Deduplicate by prettified label, keeping highest probability
                seen_labels = {}
                for (lbl, prob) in items:
                    pretty_label = self.prettify_label(lbl, group_name=gname)
                    if pretty_label not in seen_labels or prob > seen_labels[pretty_label]["prob"]:
                        seen_labels[pretty_label] = {
                            "label": pretty_label,
                            "prob": round(float(prob), 10)
                        }
                
                out[gname] = list(seen_labels.values())
                # Sort by probability again after deduplication
                out[gname].sort(key=lambda x: x["prob"], reverse=True)
            
            # Apply normalization if requested
            if normalize:
                out = self._normalization_service.normalize_tags_dict(out, normalization_opts)
            
            return out
            
        except Exception as e:
            print(f"Error in PANNs inference: {e}")
            return None
