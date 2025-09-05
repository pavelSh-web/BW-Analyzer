import os
import re
import json
import time
import shlex
import tempfile
import subprocess
from shutil import which
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import librosa
import soundfile as sf  # noqa: F401

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ──────────────────────────────────────────────────────────────────────────────
# PANNs (AudioSet)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from panns_inference import AudioTagging, labels as PANN_LABELS
    import torch
    _PANN_OK = True
except Exception:
    _PANN_OK = False
    AudioTagging = None
    PANN_LABELS = []

_PANN_MODEL: Optional["AudioTagging"] = None

# ──────────────────────────────────────────────────────────────────────────────
# DeepRhythm (tempo primary)
# ──────────────────────────────────────────────────────────────────────────────
_DR_OK = False
_DR_VERSION = None
_DR_MODEL = None
try:
    from deeprhythm import DeepRhythmPredictor
    _DR_OK = True
    try:
        import deeprhythm
        _DR_VERSION = getattr(deeprhythm, "__version__", None)
    except Exception:
        _DR_VERSION = None
except Exception:
    DeepRhythmPredictor = None
    _DR_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# 1) Human seed groups → canonize to true PANN labels only (no synonyms)
# ──────────────────────────────────────────────────────────────────────────────
TAG_GROUPS: Dict[str, List[str]] = {
    "genre": [
        "Pop music", "Rock music", "Hip hop music", "Jazz", "Blues",
        "Country", "Electronic music", "Classical music", "Folk music", "Reggae",
        "Rhythm and blues", "Soul music", "Gospel music", "Christian music",
        "Ambient music", "Techno", "House music", "Trance music", "Heavy metal",
        "Punk rock", "Funk", "Disco", "Ska", "New-age music",
        "Salsa music", "Flamenco", "Opera", "Progressive rock",
        "Psychedelic rock", "Grunge", "Dubstep", "Drum and bass",
        "Afrobeat", "Video game music", "Swing music", "Carnatic music",
        "Middle Eastern music", "Music of Africa", "Music of Asia",
        "Music of Bollywood", "Music of Latin America", "Dance music",
        "Traditional music"
    ],
    "instruments": [
        "Guitar", "Electric guitar", "Acoustic guitar", "Bass guitar",
        "Piano", "Electric piano", "Drum", "Drum kit", "Drum machine",
        "Snare drum", "Bass drum", "Hi-hat", "Cymbal", "Synthesizer",
        "Keyboard (musical)", "Organ", "Electronic organ", "Harpsichord",
        "Violin, fiddle", "Cello", "Saxophone", "Flute", "Percussion",
        "Bell", "Sitar", "Tabla", "Didgeridoo", "Bagpipes", "Accordion"
    ],
    "vocal": [
        "Singing", "Choir", "A capella",
        "Rapping", "Whistling", "Beatboxing",
        "Background vocals"
    ],
    "atmosphere": [
        "Drone", "Noise", "Buzz", "Hiss", "Hum", "Rumble",
        "Whoosh, swoosh", "Silence",
        "Environmental noise", "Crackle", "Click", "Sizzle",
        "Applause", "Cheering", "Crowd", "Chatter"
    ],
    "effects": [
        "Echo",
        "Reverberation",
        "Distortion",
        "Chorus effect"
    ],
    "style": [
        "Jingle, tinkle",
        "Theme music",
        "Background music",
        "Wedding music",
        "Christmas music",
        "Meditation music"
    ]
}

# UI aliases (display only)
DISPLAY_ALIASES: Dict[str, str] = {
    "Violin, fiddle": "Violin",
    "Whoosh, swoosh": "Whoosh",
    "Audio clipping": "Clipping",
    "Background vocals": "Backing vocals",
    "A capella": "A cappella",
    "Rhythm and blues": "R&B",
    "Keyboard (musical)": "Keyboard",
    "Christian music": "Inspirational",
}

# Rules engine targets
RULE_TARGETS = ("genre", "style")

# Custom rules (all tag names must be true PANN labels)
CUSTOM_RULES: List[Dict[str, Any]] = [
    # Modern / alt genres
    {
        "label": "Lo-fi",
        "target": "genre",
        "include_any": ["Hip hop music", "Jazz", "Ambient music", "Drum machine", "Keyboard (musical)"],
        "boost": ["Crackle", "Hiss", "Hum", "Click", "Chorus effect"],
        "exclude": ["Heavy metal", "Dubstep", "Trance music"],
        "tempo_bpm_range": [60, 95],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.7, "tempo": 0.6},
        "threshold": 1.2
    },
    {
        "label": "Trap",
        "target": "genre",
        "include_any": ["Hip hop music", "Drum machine"],
        "boost": ["Hi-hat", "Snare drum", "Bass drum"],
        "tempo_bpm_range": [130, 160],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.5, "tempo": 0.6},
        "threshold": 1.1
    },
    {
        "label": "Drill",
        "target": "genre",
        "include_any": ["Hip hop music"],
        "boost": ["Hi-hat", "Snare drum"],
        "tempo_bpm_range": [135, 150],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.4, "tempo": 0.6},
        "threshold": 1.0
    },
    {
        "label": "Synthwave",
        "target": "genre",
        "include_any": ["Electronic music", "House music", "Techno"],
        "boost": ["Synthesizer", "Keyboard (musical)", "Chorus effect", "Flanger"],
        "exclude": ["Dubstep", "Drum and bass"],
        "tempo_bpm_range": [80, 120],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.6, "tempo": 0.4},
        "threshold": 1.2
    },
    {
        "label": "Vaporwave",
        "target": "genre",
        "include_any": ["Electronic music", "Ambient music"],
        "boost": ["Chorus effect", "Reverberation", "Echo"],
        "tempo_bpm_range": [60, 100],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.6, "tempo": 0.4},
        "threshold": 1.0
    },
    {
        "label": "Future bass",
        "target": "genre",
        "include_any": ["Electronic music"],
        "boost": ["Synthesizer", "Keyboard (musical)"],
        "exclude": ["Dubstep", "Drum and bass"],
        "tempo_bpm_range": [135, 160],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.5, "tempo": 0.5},
        "threshold": 1.1
    },
    {
        "label": "Phonk",
        "target": "genre",
        "include_any": ["Hip hop music"],
        "boost": ["Drum machine", "Distortion", "Reverberation", "Chorus effect", "Click"],
        "tempo_bpm_range": [140, 165],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.6, "tempo": 0.5},
        "threshold": 1.2
    },
    {
        "label": "K-pop",
        "target": "genre",
        "include_any": ["Pop music"],
        "boost": ["Electronic music", "Dance music", "Choir", "Background vocals"],
        "exclude": ["Heavy metal"],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.5, "tempo": 0.0},
        "threshold": 1.0
    },
    {
        "label": "J-pop",
        "target": "genre",
        "include_any": ["Pop music"],
        "boost": ["Electronic music", "Dance music", "Choir", "Background vocals"],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.5, "tempo": 0.0},
        "threshold": 1.0
    },
    # Hip hop / jazz adjacent
    {
        "label": "Lo-fi hip hop",
        "target": "genre",
        "include_any": ["Hip hop music"],
        "boost": ["Drum machine", "Keyboard (musical)", "Electric piano", "Piano",
                  "Crackle", "Hiss", "Hum", "Click", "Chorus effect", "Reverberation", "Saxophone"],
        "exclude": ["Dubstep", "Drum and bass", "Heavy metal"],
        "tempo_bpm_range": [60, 95],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.7, "tempo": 0.6},
        "threshold": 1.25
    },
    {
        "label": "Jazz rap",
        "target": "genre",
        "include_all": ["Hip hop music", "Jazz"],
        "boost": ["Piano", "Saxophone", "Electric piano", "Keyboard (musical)", "Drum kit"],
        "tempo_bpm_range": [80, 110],
        "weights": {"include_any": 1.0, "include_all": 1.2, "boost": 0.6, "tempo": 0.4},
        "threshold": 1.3
    },
    {
        "label": "Boom bap",
        "target": "genre",
        "include_any": ["Hip hop music"],
        "boost": ["Drum kit", "Snare drum", "Bass drum"],
        "exclude": ["Drum and bass", "Dubstep"],
        "tempo_bpm_range": [85, 105],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.5, "tempo": 0.6},
        "threshold": 1.1
    },
    {
        "label": "Neo-soul",
        "target": "genre",
        "include_any": ["Rhythm and blues", "Soul music"],
        "boost": ["Jazz", "Electric piano", "Keyboard (musical)", "Background vocals", "Reverberation"],
        "exclude": ["Heavy metal"],
        "tempo_bpm_range": [70, 110],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.6, "tempo": 0.3},
        "threshold": 1.1
    },
    {
        "label": "Acid jazz",
        "target": "genre",
        "include_all": ["Jazz", "Funk"],
        "boost": ["Synthesizer", "Keyboard (musical)", "Chorus effect", "Flanger", "Reverberation"],
        "tempo_bpm_range": [90, 120],
        "weights": {"include_any": 1.0, "include_all": 1.1, "boost": 0.6, "tempo": 0.3},
        "threshold": 1.2
    },
    {
        "label": "Nu jazz",
        "target": "genre",
        "include_all": ["Jazz", "Electronic music"],
        "boost": ["Synthesizer", "Keyboard (musical)", "Chorus effect", "Reverberation", "Piano", "Saxophone"],
        "exclude": ["Dubstep", "Drum and bass"],
        "tempo_bpm_range": [80, 125],
        "weights": {"include_any": 1.0, "include_all": 1.1, "boost": 0.6, "tempo": 0.3},
        "threshold": 1.2
    },
    {
        "label": "Trip hop (modern)",
        "target": "genre",
        "include_all": ["Electronic music", "Hip hop music"],
        "boost": ["Reverberation", "Echo", "Chorus effect", "Drone", "Hiss", "Hum"],
        "exclude": ["Drum and bass", "Dubstep"],
        "tempo_bpm_range": [65, 95],
        "weights": {"include_any": 1.0, "include_all": 1.2, "boost": 0.7, "tempo": 0.5},
        "threshold": 1.25
    },
    {
        "label": "Jazzhop / Chillhop",
        "target": "genre",
        "include_all": ["Hip hop music", "Jazz"],
        "boost": ["Electric piano", "Piano", "Saxophone", "Crackle", "Hiss", "Reverberation"],
        "exclude": ["Drum and bass", "Dubstep"],
        "tempo_bpm_range": [70, 100],
        "weights": {"include_any": 1.0, "include_all": 1.2, "boost": 0.6, "tempo": 0.5},
        "threshold": 1.25
    },

    # Styles
    {
        "label": "Vinyl texture",
        "target": "style",
        "include_any": ["Crackle", "Hiss", "Hum", "Click"],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.0, "tempo": 0.0},
        "threshold": 0.9
    },
    {
        "label": "Live ambience",
        "target": "style",
        "include_any": ["Applause", "Cheering", "Crowd", "Chatter"],
        "boost": ["Reverberation", "Echo", "Environmental noise"],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.4, "tempo": 0.0},
        "threshold": 1.0
    },
    {
        "label": "Cinematic",
        "target": "style",
        "include_any": ["Theme music", "Background music"],
        "boost": ["Reverberation", "Echo"],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.4, "tempo": 0.0},
        "threshold": 1.0
    },
    {
        "label": "Meditative",
        "target": "style",
        "include_any": ["Meditation music", "Chant", "Mantra"],
        "weights": {"include_any": 1.0, "include_all": 1.0, "boost": 0.0, "tempo": 0.0},
        "threshold": 1.0
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    s = (s or "").strip().casefold()
    s = s.replace("&", "and")
    s = re.sub(r"[\u2010-\u2015\-\_/]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _build_label_index() -> Dict[str, str]:
    """normalized(PANN_LABEL) -> PANN_LABEL"""
    idx: Dict[str, str] = {}
    for lbl in PANN_LABELS:
        idx[_norm(lbl)] = lbl
    return idx


INDEX_NORM_TO_TRUE: Dict[str, str] = _build_label_index() if _PANN_OK else {}


def _canonize_groups(raw: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Keep only labels that exist in PANN_LABELS. Drop the rest.
    """
    if not _PANN_OK:
        return raw
    groups: Dict[str, List[str]] = {}
    for g, tags in raw.items():
        out: List[str] = []
        for t in tags:
            n = _norm(t)
            true_lbl = INDEX_NORM_TO_TRUE.get(n)
            if true_lbl and true_lbl not in out:
                out.append(true_lbl)
            else:
                if not true_lbl:
                    print(f"[canonize] dropped not-in-PANN label: '{t}'")
        groups[g] = out
    return groups


def prettify_label(label_raw: str, group_name: Optional[str] = None) -> str:
    """UI-friendly display name; does not affect matching."""
    if label_raw in DISPLAY_ALIASES:
        return DISPLAY_ALIASES[label_raw]
    if group_name == "genre" and label_raw.endswith(" music"):
        return label_raw[:-6]
    if "," in label_raw:
        return label_raw.split(",", 1)[0].strip()
    return label_raw


BW_TAG_GROUPS: Dict[str, List[str]] = _canonize_groups(TAG_GROUPS)


# ──────────────────────────────────────────────────────────────────────────────
# PANNs model loader
# ──────────────────────────────────────────────────────────────────────────────
def _pann_model() -> Optional["AudioTagging"]:
    global _PANN_MODEL
    if not _PANN_OK:
        return None
    if _PANN_MODEL is None:
        device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
        _PANN_MODEL = AudioTagging(device=device, checkpoint_path=None)
    return _PANN_MODEL


# ──────────────────────────────────────────────────────────────────────────────
# Tagging (full list per group, sorted by probability)
# ──────────────────────────────────────────────────────────────────────────────
def get_audio_tags(audio_path: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Returns full list of tags per canonical group (no top-k), sorted by descending probability.
    Each item includes: label_raw (true PANN), label (pretty), prob.
    """
    model = _pann_model()
    if model is None:
        return None

    try:
        y, _ = librosa.load(audio_path, sr=32000, mono=True)
        y_batch = y[np.newaxis, :]
        result = model.inference(y_batch)

        if isinstance(result, tuple) and len(result) == 2:
            clipwise_probs, _ = result
            probs = clipwise_probs[0] if clipwise_probs.ndim > 1 else clipwise_probs

            out: Dict[str, List[Dict[str, Any]]] = {}

            label_to_groups: Dict[str, List[str]] = {}
            for gname, true_list in BW_TAG_GROUPS.items():
                for lbl in true_list:
                    label_to_groups.setdefault(lbl, []).append(gname)

            items_by_group: Dict[str, List[Tuple[str, float]]] = {g: [] for g in BW_TAG_GROUPS.keys()}
            for i, lbl in enumerate(PANN_LABELS):
                if lbl in label_to_groups:
                    p = float(probs[i])
                    for gname in label_to_groups[lbl]:
                        items_by_group[gname].append((lbl, p))

            for gname, items in items_by_group.items():
                items.sort(key=lambda x: x[1], reverse=True)
                out[gname] = [
                    {
                        "label": prettify_label(lbl, group_name=gname),
                        "prob": round(float(prob), 10)
                    }
                    for (lbl, prob) in items
                ]

            return out

        return None
    except Exception as e:
        print(f"PANNs error: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Rules expansion (genre/style)
# ──────────────────────────────────────────────────────────────────────────────
def infer_rule_expansions(
    pann_groups: Optional[Dict[str, List[Dict[str, Any]]]],
    features: Dict[str, Any],
    rules: List[Dict[str, Any]],
    prob_threshold: float = 0.05
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Expand using simple weighted scoring. Confidence is a clipped 0..1 score proxy.
    """
    present = set()
    if pann_groups:
        for _, items in pann_groups.items():
            for it in items:
                lbl = it.get("label")
                if not lbl:
                    continue
                if float(it.get("prob", 0.0)) >= prob_threshold:
                    present.add(lbl)

    tempo = features.get("tempo_bpm")
    result: Dict[str, List[Dict[str, Any]]] = {t: [] for t in RULE_TARGETS}

    for rule in rules:
        target = rule.get("target")
        if target not in RULE_TARGETS:
            continue

        include_any = set(rule.get("include_any", []))
        include_all = set(rule.get("include_all", []))
        boost = set(rule.get("boost", []))
        exclude = set(rule.get("exclude", []))
        tr = rule.get("tempo_bpm_range")
        weights = rule.get("weights", {})
        threshold = float(rule.get("threshold", 1.0))

        score = 0.0

        if include_any and any(tag in present for tag in include_any):
            score += float(weights.get("include_any", 1.0))
        if include_all and all(tag in present for tag in include_all):
            score += float(weights.get("include_all", 1.0))
        if boost:
            n_boost = sum(1 for tag in boost if tag in present)
            score += n_boost * float(weights.get("boost", 0.0))
        if exclude and any(tag in present for tag in exclude):
            score -= 1.0

        if tempo and tr and isinstance(tempo, (int, float, np.floating)):
            lo, hi = tr[0], tr[1]
            if lo <= float(tempo) <= hi:
                score += float(weights.get("tempo", 0.0))
            else:
                score -= 0.2

        if score >= threshold:
            conf = max(0.0, min(1.0, score / (threshold + 1.5)))
            result[target].append({
                "label": rule.get("label", ""),
                "confidence": round(conf, 3)
            })

    for t in result:
        result[t].sort(key=lambda x: x["confidence"], reverse=True)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# DeepRhythm integration
# ──────────────────────────────────────────────────────────────────────────────
def detect_tempo_deeprhythm(audio_path: str) -> Optional[Dict[str, Any]]:
    """
    Run DeepRhythm to estimate tempo using DeepRhythmPredictor.
    DeepRhythm requires longer audio files (at least 5-10 seconds) for reliable tempo detection.

    Returns:
      {
        "tempo_bpm": float,
        "tempo_confidence": float|None,
        "tempo_method": "deeprhythm",
        "tempo_debug": { ... }
      }
    or None if not available/failed.
    """
    if not _DR_OK or DeepRhythmPredictor is None:
        return None
        
    try:
        # Check audio duration first - DeepRhythm needs longer files
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            duration = len(y) / sr
            if duration < 5.0:  # Less than 5 seconds
                return None
        except Exception as e:
            return None
        
        # Initialize DeepRhythm predictor (lazy loading)
        global _DR_MODEL
        if _DR_MODEL is None:
            try:
                _DR_MODEL = DeepRhythmPredictor()
            except Exception as e:
                return None
        
        # Method 1: Try predict() without confidence first (more reliable)
        try:
            tempo = _DR_MODEL.predict(audio_path)
            return {
                "tempo_bpm": round(float(tempo), 1),
                "tempo_confidence": None,
                "tempo_method": "deeprhythm",
                "tempo_debug": {"method": "predict_file", "duration": duration}
            }
        except Exception as e:
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
        except Exception as e:
            pass
        
        # Method 3: Try predict_from_audio() with librosa-loaded audio
        try:
            tempo, confidence = _DR_MODEL.predict_from_audio(y, sr, include_confidence=True)
            return {
                "tempo_bpm": round(float(tempo), 1),
                "tempo_confidence": round(float(confidence), 3) if confidence is not None else None,
                "tempo_method": "deeprhythm",
                "tempo_debug": {"method": "predict_from_audio", "duration": duration}
            }
        except Exception as e:
            pass
            
    except Exception as e:
        pass

    return None


def _fallback_tempo(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Minimal fallback: librosa beat track + simple octave correction heuristics.
    """
    try:
        # base estimate
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=True)
        bpm = float(tempo)
        conf = 0.4  # placeholder; librosa doesn't return confidence

        # octave correction heuristics using tempo range priors
        # map candidate set: {0.5x, 1x, 2x} inside [49, 200] bpm
        candidates = [bpm / 2.0, bpm, bpm * 2.0]
        candidates = [c for c in candidates if 49 <= c <= 200]

        # choose closest to common pop/rock priors (75..170), prefer even backbeat region (~120..170)
        target_ranges = [(75, 170), (100, 140), (120, 170)]
        weights = [1.0, 1.2, 1.4]

        def score_candidate(c: float) -> float:
            s = 0.0
            for (lo, hi), w in zip(target_ranges, weights):
                if lo <= c <= hi:
                    s += w
                else:
                    # soft distance penalty
                    dist = min(abs(c - lo), abs(c - hi))
                    s += max(0.0, w - dist * 0.01)
            return s

        best = max(candidates, key=score_candidate) if candidates else bpm
        if abs(best - bpm) > 1.0:
            bpm = best

        return {
            "tempo_bpm": round(bpm, 1),
            "tempo_confidence": conf,
            "tempo_method": "librosa_fallback",
            "tempo_debug": {"base": tempo, "candidates": candidates}
        }
    except Exception as e:
        return {
            "tempo_bpm": 120.0,
            "tempo_confidence": 0.0,
            "tempo_method": "librosa_fallback_error",
            "tempo_debug": {"error": str(e)}
        }


# ──────────────────────────────────────────────────────────────────────────────
# Musical features (DeepRhythm tempo primary)
# ──────────────────────────────────────────────────────────────────────────────
def analyze_audio_features(y: np.ndarray, sr: int, audio_path: Optional[str] = None) -> Dict[str, Any]:
    """Analyze musical characteristics with DeepRhythm tempo (primary) and librosa fallback."""
    features: Dict[str, Any] = {}

    try:
        # === TEMPO ===
        if audio_path:
            dr = detect_tempo_deeprhythm(audio_path)
        else:
            dr = None

        if dr:
            features["tempo"] = float(dr["tempo_bpm"])
        else:
            fb = _fallback_tempo(y, sr)
            features["tempo"] = fb["tempo_bpm"]

        # === KEY (using Skey from Deezer) ===
        try:
            from skey import detect_key
            skey_result = detect_key(audio_path, device='cpu')
            if skey_result and len(skey_result) > 0:
                features["key"] = skey_result[0]  # skey возвращает список, берем первый элемент
            else:
                features["key"] = "Unknown"
        except Exception as e:
            # Fallback to simple chroma-based detection if skey fails
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
                cm = np.mean(chroma, axis=1)
                maj = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
                mino = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
                maj_scores = [np.dot(cm, np.roll(maj, i)) for i in range(12)]
                min_scores = [np.dot(cm, np.roll(mino, i)) for i in range(12)]
                bi_maj = int(np.argmax(maj_scores))
                bi_min = int(np.argmax(min_scores))
                names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                features["key"] = (names[bi_maj] + ' major' if maj_scores[bi_maj] > min_scores[bi_min] 
                                 else names[bi_min] + ' minor')
            except:
                features["key"] = "Unknown"


        # === Other features ===
        rms = librosa.feature.rms(y=y)[0]
        features["energy"] = round(float(np.mean(rms)), 3)
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["brightness"] = round(float(np.mean(sc)), 1)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zero_crossing_rate"] = round(float(np.mean(zcr)), 3)
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features["spectral_rolloff"] = round(float(np.mean(roll)), 1)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features["mfcc_mean"] = [round(float(x), 3) for x in np.mean(mfccs, axis=1)]

    except Exception as e:
        print(f"Feature analysis error: {e}")
        features["error"] = str(e)

    return features


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Audio Analysis API",
    version="3.1.0",
    description="Audio analysis microservice with PANNs + DeepRhythm tempo + features + rules"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze")
async def analyze_track(
    file: UploadFile = File(...)
):
    """
    Analyze audio file and return:
      - full PANNs tags grouped by canonical groups (no top-k)
      - musical features (tempo by DeepRhythm if available, else fallback)
      - inferred expansions (genre/style) via rules
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        raise HTTPException(400, "Unsupported file type")

    t0 = time.time()
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(file.filename)[1] or ".wav",
            delete=False
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        y, sr = librosa.load(temp_path, sr=None, mono=True)

        tags = get_audio_tags(temp_path)
        features = analyze_audio_features(y, sr, audio_path=temp_path)
        inferred = infer_rule_expansions(tags, features, CUSTOM_RULES)

        duration = float(len(y) / sr)

        return JSONResponse({
            "duration": duration,
            "tags": tags,
            "musical_features": features,
            "inferred": inferred,
            "elapsed_sec": round(time.time() - t0, 3)
        })

    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@app.get("/")
def root():
    return {
        "service": "Audio Tags & Features API",
        "version": "3.1.0",
        "endpoints": {
            "POST /analyze": "Synchronous audio analysis: PANNs tags + features + rule expansions",
            "GET /tags": "Canonical groups (true PANN/AudioSet labels)",
            "GET /tags/pretty": "Groups with pretty display names",
        },
        "tools": {
            "panns": _PANN_OK,
            "deeprhythm": _DR_OK,
            "deeprhythm_version": _DR_VERSION,
        }
    }


@app.get("/tags")
def get_canonical_tag_groups():
    pretty: Dict[str, List[str]] = {}
    for g, tags in BW_TAG_GROUPS.items():
        pretty[g] = [prettify_label(lbl, group_name=g) for lbl in tags]
    return {
        "total_tags": sum(len(tags) for tags in pretty.values()),
        "categories": {key: len(tags) for key, tags in pretty.items()},
        "tags": pretty
    }
