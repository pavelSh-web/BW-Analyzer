from typing import Dict, Any, List, Union, Optional

from ..config import BW_TAG_GROUPS, DISPLAY_ALIASES, EMBEDDING_WEIGHTS, EMBEDDING_NORMALIZATION, EMBEDDING_BLOCK_WEIGHTS


class EmbeddingService:
    """Builds a deterministic fixed-size embedding vector from analysis results."""

    def __init__(self) -> None:
        # Precompute tag orders for each tag group to align probabilities
        self._pretty_order_by_group: Dict[str, List[str]] = {}
        for group_name, labels in BW_TAG_GROUPS.items():
            # Use original labels from config 
            self._pretty_order_by_group[group_name] = labels

    def _prettify_label(self, label_raw: str, group_name: Optional[str] = None) -> str:
        if label_raw in DISPLAY_ALIASES:
            return DISPLAY_ALIASES[label_raw]
        # Remove " music" suffix for genre, mood, and style groups
        if label_raw.endswith(" music"):
            return label_raw[:-6]
        if "," in label_raw:
            return label_raw.split(",", 1)[0].strip()
        return label_raw

    def _encode_category3(self, value: Optional[str]) -> List[float]:
        vec = [0.0, 0.0, 0.0]
        if not value:
            return vec
        mapping = {"low": 0, "mid": 1, "high": 2}
        idx = mapping.get(value)
        if idx is not None:
            vec[idx] = 1.0
        return vec

    def _normalize_value(self, value: float, feature_name: str) -> float:
        """Normalize a value to 0-1 range based on config."""
        if feature_name not in EMBEDDING_NORMALIZATION:
            return value
        
        norm_config = EMBEDDING_NORMALIZATION[feature_name]
        min_val = norm_config["min"]
        max_val = norm_config["max"]
        
        # Clamp value to range and normalize
        clamped = max(min_val, min(max_val, value))
        return (clamped - min_val) / (max_val - min_val)
    
    def _l2_normalize_block(self, block: List[float], eps: float = 1e-8) -> List[float]:
        """L2 normalize a block of features: block / (||block|| + eps)"""
        if not block:
            return block
        
        # Calculate L2 norm
        l2_norm = sum(x * x for x in block) ** 0.5
        
        # Avoid division by zero
        if l2_norm < eps:
            return block
        
        # Normalize
        return [x / l2_norm for x in block]
    
    def _apply_block_weights_and_normalize(self, blocks: Dict[str, List[float]]) -> List[float]:
        """Apply block-wise L2 normalization, weights, and final global normalization."""
        normalized_blocks = []
        
        # 1. L2 normalize each block separately
        for block_name, block_data in blocks.items():
            if not block_data:
                continue
                
            # L2 normalize the block
            normalized_block = self._l2_normalize_block(block_data)
            
            # Apply block weight
            block_weight = EMBEDDING_BLOCK_WEIGHTS.get(block_name, 1.0)
            weighted_block = [x * block_weight for x in normalized_block]
            
            normalized_blocks.extend(weighted_block)
        
        # 2. Final global L2 normalization
        final_embedding = self._l2_normalize_block(normalized_blocks)
        
        return final_embedding

    def build(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build embedding from analysis response with block-wise L2 normalization.

        Block structure for balanced representation:
        - Tags: genre, instruments, vocal, atmosphere, mood, effects, style [109]
        - Tempo: normalized tempo value [1]  
        - Energy/Brightness: energy_value, brightness_value [2]
        - Rhythm: onset_density, percussive_harmonic_ratio, beat_histogram_mean [3]
        - Harmony: chroma_mean, chroma_std, tonnetz_mean, tonnetz_std, key_clarity [37]
        - Timbre: spectral features [6]
        - Dynamics: dynamic_range, loudness stats, LUFS [5]
        
        Total: ~163 dimensions with block-wise L2 normalization + weights
        """
        blocks = {}

        # 1) Tags block - collect all tag probabilities
        tags_block = []
        tags: Dict[str, List[Dict[str, Any]]] = analysis.get("tags") or {}
        for group_name, order in self._pretty_order_by_group.items():
            group_vec = [0.0] * len(order)
            items = tags.get(group_name) or []
            # Build map tag->prob (use raw canonical 'tag' from PANNs, fallback to 'label')
            for item in items:
                raw = item.get("tag") or item.get("label")
                prob = float(item.get("prob", 0.0))
                if raw in order:
                    idx = order.index(raw)
                    group_vec[idx] = prob
            tags_block.extend(group_vec)
        blocks["tags"] = tags_block

        # 2) Tempo block - single normalized value
        tempo = self._normalize_value(float(analysis.get("tempo", 0.0)), "tempo")
        blocks["tempo"] = [tempo]

        # 3) Energy/Brightness block
        energy_value = self._normalize_value(float(analysis.get("energy_value", 0.0)), "energy_value")
        brightness_value = self._normalize_value(float(analysis.get("brightness_value", 0.0)), "brightness_value")
        blocks["energy_brightness"] = [energy_value, brightness_value]

        # 4) Features blocks - extract and split by semantic meaning
        features = analysis.get("features", {})
        if features:
            # Rhythm block
            rhythm_block = []
            rhythm_block.append(self._normalize_value(float(features.get("onset_density", 0.0)), "onset_density"))
            rhythm_block.append(float(features.get("percussive_harmonic_ratio", 0.0)))  # Already 0-1
            rhythm_block.append(self._normalize_value(float(features.get("beat_histogram_mean", 0.0)), "beat_histogram_mean"))
            blocks["rhythm"] = rhythm_block
            
            # Harmony block
            harmony_block = []
            chroma_mean = features.get("chroma_mean", [0.0] * 12)
            chroma_std = features.get("chroma_std", [0.0] * 12)
            tonnetz_mean = features.get("tonnetz_mean", [0.0] * 6)
            tonnetz_std = features.get("tonnetz_std", [0.0] * 6)
            
            harmony_block.extend([float(x) for x in chroma_mean[:12]])
            harmony_block.extend([float(x) for x in chroma_std[:12]])
            harmony_block.extend([self._normalize_value(float(x), "tonnetz") for x in tonnetz_mean[:6]])
            harmony_block.extend([self._normalize_value(float(x), "tonnetz") for x in tonnetz_std[:6]])
            harmony_block.append(float(features.get("key_clarity", 0.0)))  # Already 0-1
            blocks["harmony"] = harmony_block
            
            # Timbre block
            timbre_block = []
            timbre_block.append(float(features.get("spectral_flatness_mean", 0.0)))  # Already 0-1
            timbre_block.append(float(features.get("spectral_flatness_std", 0.0)))   # Already 0-1
            timbre_block.append(self._normalize_value(float(features.get("spectral_bandwidth_mean", 0.0)), "spectral_bandwidth_mean"))
            timbre_block.append(self._normalize_value(float(features.get("spectral_bandwidth_std", 0.0)), "spectral_bandwidth_std"))
            timbre_block.append(self._normalize_value(float(features.get("zero_crossing_rate_mean", 0.0)), "zero_crossing_rate_mean"))
            timbre_block.append(self._normalize_value(float(features.get("zero_crossing_rate_std", 0.0)), "zero_crossing_rate_std"))
            blocks["timbre"] = timbre_block
            
            # Dynamics block
            dynamics_block = []
            dynamics_block.append(self._normalize_value(float(features.get("dynamic_range_db", 0.0)), "dynamic_range_db"))
            dynamics_block.append(self._normalize_value(float(features.get("loudness_mean", 0.0)), "loudness"))
            dynamics_block.append(self._normalize_value(float(features.get("loudness_std", 0.0)), "loudness_std"))
            dynamics_block.append(self._normalize_value(float(features.get("loudness_range", 0.0)), "loudness_range"))
            lufs = features.get("lufs")
            dynamics_block.append(self._normalize_value(float(lufs) if lufs is not None else 0.0, "lufs"))
            blocks["dynamics"] = dynamics_block
        else:
            # Fill with zeros if features are missing
            blocks["rhythm"] = [0.0] * 3
            blocks["harmony"] = [0.0] * 37
            blocks["timbre"] = [0.0] * 6
            blocks["dynamics"] = [0.0] * 5

        # Apply block-wise L2 normalization, weights, and final global normalization
        final_embedding = self._apply_block_weights_and_normalize(blocks)
        
        # Round all values to 10 decimal places for compactness
        rounded_embedding = [round(val, 10) for val in final_embedding]
        
        return {
            "embedding": rounded_embedding,
            "dim": len(rounded_embedding),
        }

    def _encode_enhanced_features(self, enhanced: Optional[Dict[str, Any]], weight: float = 1.0) -> List[float]:
        """Encode features in fixed order with normalization and weighting."""
        if not enhanced:
            # Return zeros for all optimized features
            return [0.0] * 51  # Total: 3 rhythm + 37 harmony + 6 timbre + 5 dynamics
        
        parts: List[float] = []
        
        # Rhythm features (3) - beat_histogram_std removed as unstable
        onset_density = self._normalize_value(float(enhanced.get("onset_density", 0.0)), "onset_density")
        perc_harm_ratio = float(enhanced.get("percussive_harmonic_ratio", 0.0))  # Already 0-1
        beat_hist_mean = self._normalize_value(float(enhanced.get("beat_histogram_mean", 0.0)), "beat_histogram_mean")
        
        parts.append(onset_density * weight)
        parts.append(perc_harm_ratio * weight)
        parts.append(beat_hist_mean * weight)
        
        # Harmony features (37 = 12+12+6+6+1) - chroma/tonnetz already 0-1
        chroma_mean = enhanced.get("chroma_mean", [0.0] * 12)
        chroma_std = enhanced.get("chroma_std", [0.0] * 12)
        tonnetz_mean = enhanced.get("tonnetz_mean", [0.0] * 6)
        tonnetz_std = enhanced.get("tonnetz_std", [0.0] * 6)
        
        parts.extend([float(x) * weight for x in chroma_mean[:12]])  # Ensure exactly 12
        parts.extend([float(x) * weight for x in chroma_std[:12]])
        parts.extend([self._normalize_value(float(x), "tonnetz") * weight for x in tonnetz_mean[:6]])  # Normalize tonnetz
        parts.extend([self._normalize_value(float(x), "tonnetz") * weight for x in tonnetz_std[:6]])
        
        key_clarity = float(enhanced.get("key_clarity", 0.0))  # Already 0-1
        parts.append(key_clarity * weight)
        
        # Timbre features (6) - normalize where needed
        spectral_flatness_mean = float(enhanced.get("spectral_flatness_mean", 0.0))
        spectral_flatness_std = float(enhanced.get("spectral_flatness_std", 0.0))
        spectral_bandwidth_mean = self._normalize_value(float(enhanced.get("spectral_bandwidth_mean", 0.0)), "spectral_bandwidth_mean")
        spectral_bandwidth_std = self._normalize_value(float(enhanced.get("spectral_bandwidth_std", 0.0)), "spectral_bandwidth_std")
        zcr_mean = self._normalize_value(float(enhanced.get("zero_crossing_rate_mean", 0.0)), "zero_crossing_rate_mean")
        zcr_std = self._normalize_value(float(enhanced.get("zero_crossing_rate_std", 0.0)), "zero_crossing_rate_std")
        
        parts.append(spectral_flatness_mean * weight)
        parts.append(spectral_flatness_std * weight)
        parts.append(spectral_bandwidth_mean * weight)
        parts.append(spectral_bandwidth_std * weight)
        parts.append(zcr_mean * weight)
        parts.append(zcr_std * weight)
        
        # Dynamics features (5) - removed unstable min/max values
        dynamic_range = self._normalize_value(float(enhanced.get("dynamic_range_db", 0.0)), "dynamic_range_db")
        loudness_mean = self._normalize_value(float(enhanced.get("loudness_mean", 0.0)), "loudness")
        loudness_std = self._normalize_value(float(enhanced.get("loudness_std", 0.0)), "loudness_std")
        loudness_range = self._normalize_value(float(enhanced.get("loudness_range", 0.0)), "loudness_range")
        
        parts.append(dynamic_range * weight)
        parts.append(loudness_mean * weight)
        parts.append(loudness_std * weight)
        parts.append(loudness_range * weight)
        
        # LUFS (normalized and weighted)
        lufs = enhanced.get("lufs")
        lufs_normalized = self._normalize_value(float(lufs) if lufs is not None else 0.0, "lufs")
        parts.append(lufs_normalized * weight)
        
        # Pad or truncate to exactly 51 features  
        while len(parts) < 51:
            parts.append(0.0)
        return parts[:51]


