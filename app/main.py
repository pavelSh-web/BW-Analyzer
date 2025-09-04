import os
import re
import time
import tempfile
from typing import List, Optional, Dict, Any

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
# 1) Human seed groups (will be canonized to true PANN labels)
#    These lists should contain real AudioSet labels whenever possible.
#    Any label not found in PANN_LABELS will be dropped during canonization.
# ──────────────────────────────────────────────────────────────────────────────
IMPORTANT_TAG_GROUPS_RAW: Dict[str, List[str]] = {
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
        "Whoosh, swoosh", "Reverberation", "Echo", "Silence",
        "Environmental noise", "Crackle", "Click", "Sizzle",
        "Applause", "Cheering", "Crowd", "Chatter"
    ],
    "spectral": [
        "Distortion", "Audio clipping",
        "Chorus effect", "Flanger"
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


# ──────────────────────────────────────────────────────────────────────────────
# 2) Display aliases (UI only). Do not affect matching or group canonization.
# ──────────────────────────────────────────────────────────────────────────────
DISPLAY_ALIASES: Dict[str, str] = {
    "Violin, fiddle": "Violin",
    "Whoosh, swoosh": "Whoosh",
    "Audio clipping": "Clipping",
    "Background vocals": "Backing vocals",
    "A capella": "A cappella",
    "Rhythm and blues": "R&B",
    "Keyboard (musical)": "Keyboard",
}


# ──────────────────────────────────────────────────────────────────────────────
# 3) Rules engine (generic). We expand to targets (e.g., genre, style).
#    All rule tag names must be true PANN labels.
# ──────────────────────────────────────────────────────────────────────────────
RULE_TARGETS = ("genre", "style")

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

    # Styles (target: style)
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
    No synonym mapping is applied here by design.
    """
    if not _PANN_OK:
        # Model/labels not available at import time: return raw as-is.
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


IMPORTANT_TAG_GROUPS: Dict[str, List[str]] = _canonize_groups(IMPORTANT_TAG_GROUPS_RAW)


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
# Tagging (returns full list per group, sorted by probability)
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

            # Pre-build index of group membership: pann_index -> group_name (multi)
            label_to_groups: Dict[str, List[str]] = {}
            for gname, true_list in IMPORTANT_TAG_GROUPS.items():
                for lbl in true_list:
                    label_to_groups.setdefault(lbl, []).append(gname)

            # Iterate over all PANN labels and collect those that belong to our groups
            items_by_group: Dict[str, List[Tuple[str, float]]] = {g: [] for g in IMPORTANT_TAG_GROUPS.keys()}
            for i, lbl in enumerate(PANN_LABELS):
                if lbl in label_to_groups:
                    p = float(probs[i])
                    for gname in label_to_groups[lbl]:
                        items_by_group[gname].append((lbl, p))

            # Sort and format
            for gname, items in items_by_group.items():
                items.sort(key=lambda x: x[1], reverse=True)
                out[gname] = [
                    {
                        "name": prettify_label(lbl, group_name=gname),
                        "prob": float(prob)
                    }
                    for (lbl, prob) in items
                ]

            return out

        return None
    except Exception as e:
        print(f"PANNs error: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Generic rules expansion (genre/style inference)
# ──────────────────────────────────────────────────────────────────────────────
def infer_rule_expansions(
    pann_groups: Optional[Dict[str, List[Dict[str, Any]]]],
    features: Dict[str, Any],
    rules: List[Dict[str, Any]],
    prob_threshold: float = 0.08
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Apply generic rule expansions to derive labels into target collections (e.g., genre/style).
    Returns: {"genre": [{"label": ..., "confidence": ...}, ...], "style": [...]}.
    """
    present = set()
    if pann_groups:
        for _, items in pann_groups.items():
            for it in items:
                lbl = it.get("label_raw") or it.get("label")
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
# Musical features (kept as in your previous version)
# ──────────────────────────────────────────────────────────────────────────────
def analyze_audio_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    features = {}

    try:
        # Tempo
        tempo_methods = []
        t1, _ = librosa.beat.beat_track(y=y, sr=sr, units='time')
        tempo_methods.append(('librosa_default', t1))
        t2, _ = librosa.beat.beat_track(y=y, sr=sr, units='time', start_bpm=60)
        tempo_methods.append(('librosa_wide', t2))
        onset = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        if len(onset) > 1:
            iv = np.diff(onset)
            iv = iv[iv < 2.0]
            if len(iv) > 0:
                tempo_methods.append(('onset_based', 60.0 / np.median(iv)))

        valid = [(n, t) for n, t in tempo_methods if 60 <= t <= 200]
        if valid:
            pref = [(n, t) for n, t in valid if 80 <= t <= 160]
            tempo_method, tempo_bpm = (pref[0] if pref else valid[0])
        else:
            tempo_method, tempo_bpm = tempo_methods[0]
        features["tempo_bpm"] = round(float(tempo_bpm), 1)
        features["tempo_method"] = tempo_method

        # Key (voting of multiple methods)
        key_methods: List[tuple] = []
        try:
            k1 = librosa.key.estimate_key(y, sr)
            key_methods.append(('librosa_default', k1))
        except:
            pass
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
            cm = np.mean(chroma, axis=1)
            dom = int(np.argmax(cm))
            names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_methods.append(('chroma_dominant', names[dom]))
        except:
            pass
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
            key_methods.append((
                'chroma_weighted',
                names[bi_maj] + ' major' if maj_scores[bi_maj] > min_scores[bi_min] else names[bi_min] + ' minor'
            ))
        except:
            pass
        try:
            cqt = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
            cm = np.mean(cqt, axis=1)
            maj = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            mino = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            maj_scores = [np.dot(cm, np.roll(maj, i)) for i in range(12)]
            min_scores = [np.dot(cm, np.roll(mino, i)) for i in range(12)]
            bi_maj = int(np.argmax(maj_scores))
            bi_min = int(np.argmax(min_scores))
            names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_methods.append((
                'chroma_cqt',
                names[bi_maj] + ' major' if maj_scores[bi_maj] > min_scores[bi_min] else names[bi_min] + ' minor'
            ))
        except:
            pass
        try:
            cens = librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=12)
            cm = np.mean(cens, axis=1)
            maj = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            mino = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            maj_scores = [np.dot(cm, np.roll(maj, i)) for i in range(12)]
            min_scores = [np.dot(cm, np.roll(mino, i)) for i in range(12)]
            bi_maj = int(np.argmax(maj_scores))
            bi_min = int(np.argmax(min_scores))
            names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_methods.append((
                'chroma_cens',
                names[bi_maj] + ' major' if maj_scores[bi_maj] > min_scores[bi_min] else names[bi_min] + ' minor'
            ))
        except:
            pass
        try:
            # simple spectral peak heuristic
            fft = np.fft.fft(y)
            freqs = np.fft.fftfreq(len(y), 1/sr)
            pf = freqs[:len(freqs)//2]
            pa = np.abs(fft[:len(fft)//2])
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(pa, height=np.max(pa)*0.1)
            if len(peaks) > 0:
                dom = pf[peaks[np.argmax(pa[peaks])]]
                a4 = 440.0
                off = 12 * np.log2(max(dom, 1e-12) / a4)
                idx = int(np.round(off)) % 12
                names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
                key_methods.append(('spectral_peaks', names[idx]))
        except:
            pass

        if key_methods:
            key_votes: Dict[str, float] = {}
            method_w = {
                'chroma_cqt': 2.5,
                'chroma_cens': 2.5,
                'chroma_weighted': 2.0,
                'librosa_default': 1.5,
                'spectral_peaks': 1.0,
                'chroma_dominant': 0.5,
            }
            for m, k in key_methods:
                key_votes[k] = key_votes.get(k, 0.0) + method_w.get(m, 1.0)
            best_key = max(key_votes, key=key_votes.get)
            key_method = next((m for m, k in key_methods if k == best_key), 'vote')
            features["key"] = best_key
            features["key_method"] = key_method
        else:
            features["key"] = "Unknown"
            features["key_method"] = "unknown"

        # Time signature (heuristics)
        methods_ts = []
        try:
            frame_length = sr
            hop_length = frame_length // 4
            rms_frames = []
            for i in range(0, len(y) - frame_length, hop_length):
                frame = y[i:i + frame_length]
                rms_frames.append(np.sqrt(np.mean(frame**2)))
            if len(rms_frames) > 8:
                arr = np.array(rms_frames)
                arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-12)
                a44 = sum(1 for i in range(0, len(arr) - 4, 4) if arr[i] > np.mean(arr[i:i+4]) + 0.5)
                a34 = sum(1 for i in range(0, len(arr) - 3, 3) if arr[i] > np.mean(arr[i:i+3]) + 0.5)
                methods_ts.append(('accent_analysis', '4/4' if a44 > a34 else '3/4'))
        except:
            pass
        try:
            onset = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            if len(onset) > 12:
                ints = np.diff(onset)
                avg = np.median(ints)
                b4 = 0
                for i in range(0, len(ints) - 3, 4):
                    if i + 3 < len(ints):
                        if (ints[i] < avg * 1.5 and ints[i+1] < avg * 1.5 and
                            ints[i+2] < avg * 1.5 and ints[i+3] > avg * 1.5):
                            b4 += 1
                b3 = 0
                for i in range(0, len(ints) - 2, 3):
                    if i + 2 < len(ints):
                        if (ints[i] < avg * 1.5 and ints[i+1] < avg * 1.5 and
                            ints[i+2] > avg * 1.5):
                            b3 += 1
                methods_ts.append(('onset_grouping', '4/4' if b4 > b3 else '3/4'))
        except:
            pass
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            cm = np.mean(chroma, axis=0)
            diff = np.abs(np.diff(cm))
            p44 = sum(1 for i in range(0, len(diff) - 4, 4) if np.std(diff[i:i+4]) > np.mean(diff))
            p34 = sum(1 for i in range(0, len(diff) - 3, 3) if np.std(diff[i:i+3]) > np.mean(diff))
            methods_ts.append(('spectral_periodicity', '4/4' if p44 > p34 else '3/4'))
        except:
            pass
        try:
            low_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)[0]
            low = low_centroid[low_centroid < 200]
            if len(low) > 8:
                peaks4 = sum(1 for i in range(0, len(low) - 4, 4) if np.std(low[i:i+4]) > np.mean(low))
                methods_ts.append(('kick_drum_analysis', '4/4' if peaks4 > len(low) // 8 else '3/4'))
        except:
            pass
        try:
            if float(features["tempo_bpm"]) < 60:
                methods_ts.append(('tempo_heuristic', '3/4'))
            else:
                methods_ts.append(('tempo_heuristic', '4/4'))
        except:
            pass

        if methods_ts:
            v4 = sum(1 for _, s in methods_ts if s == '4/4')
            v3 = sum(1 for _, s in methods_ts if s == '3/4')
            ts = '3/4' if v3 > v4 * 1.5 else '4/4'
            features["time_signature"] = ts
            features["time_sig_method"] = "majority_vote"
        else:
            features["time_signature"] = "4/4"
            features["time_sig_method"] = "default"

        # Other features
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
    version="2.2.0",
    description="Audio analysis microservice with PANNs tagging + feature extraction + rules"
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
      - musical features
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
        features = analyze_audio_features(y, sr)
        inferred = infer_rule_expansions(tags, features, CUSTOM_RULES)

        duration = float(len(y) / sr)

        return JSONResponse({
            "filename": file.filename,
            "duration_seconds": duration,
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
        "version": "3.0.0",
        "endpoints": {
            "POST /analyze": "Synchronous audio analysis: PANNs tags + musical features + rule expansions",
            "GET /tags": "Canonical groups (true PANN/AudioSet labels)",
            "GET /tags/pretty": "Groups with pretty display names",
        },
        "features": [
            "Synchronous analysis for immediate results",
            "PANNs audio tagging with curated musical tags",
            "Musical feature extraction (tempo, key, time signature)",
            "Rule-based tag expansion",
            "External queue integration (configurable)"
        ],
        "queue_config_hint": "Set QUEUE_TYPE and connection URLs to enable external queue in the future"
    }


@app.get("/tags")
def get_canonical_tag_groups():
    return {
        "total_tags": sum(len(tags) for tags in IMPORTANT_TAG_GROUPS.values()),
        "categories": {key: len(tags) for key, tags in IMPORTANT_TAG_GROUPS.items()},
        "tags": IMPORTANT_TAG_GROUPS
    }


@app.get("/tags/pretty")
def get_pretty_tag_groups():
    pretty: Dict[str, List[str]] = {}
    for g, tags in IMPORTANT_TAG_GROUPS.items():
        pretty[g] = [prettify_label(lbl, group_name=g) for lbl in tags]
    return {
        "total_tags": sum(len(tags) for tags in pretty.values()),
        "categories": {key: len(tags) for key, tags in pretty.items()},
        "tags": pretty
    }


# ──────────────────────────────────────────────────────────────────────────────
# External Queue Integration
# ──────────────────────────────────────────────────────────────────────────────

# External queue endpoints removed for now. Future integration can reuse
# form parameters and async handlers here.