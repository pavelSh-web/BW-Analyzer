"""Configuration and constants for BW Analyzer"""

# BW tag groups
BW_TAG_GROUPS = {
    "genre": [
        "Pop music", "Rock music", "Hip hop music", "Jazz", "Blues", "Bluegrass",
        "Country", "Electronic music", "Classical music", "Folk music", "Reggae",
        "Rhythm and blues", "Soul music", "Gospel music", "Christian music",
        "Ambient music", "Techno", "House music", "Trance music", 
        "Heavy metal", "Punk rock", "Funk", "Disco", "Ska", "New-age music",
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
        "Rapping", "Whistling", "Beatboxing"
    ],
    "atmosphere": [
        "Noise", "Buzz", "Hiss", "Hum", "Rumble",
        "Whoosh, swoosh", "Silence",
        "Environmental noise", "Crackle", "Click", "Sizzle",
        "Applause", "Cheering", "Crowd", "Chatter"
    ],
    "mood": [
        "Happy music", "Sad music", "Tender music", 
        "Funny music", "Exciting music", "Scary music"
    ],
    "effects": [
        "Echo", "Reverberation", "Distortion", "Chorus effect"
    ],
    "style": [
        "Jingle", "Theme music", "Background music", 
        "Wedding music", "Christmas music"
    ]
}

# Constants for DeepRhythm
DEEPRHYTHM_MIN_DURATION = 5.0  # minimum duration for DeepRhythm

# Temperature for softmax normalization
SOFTMAX_TEMPERATURE = 1.2

# Energy classification thresholds
ENERGY_THRESHOLDS = {
    "low": 0.10,
    "mid": 0.22
}

# Brightness classification thresholds (normalized to Nyquist)
BRIGHTNESS_THRESHOLDS = {
    "low": 0.20,
    "mid": 0.50
}

# Embedding weights for different feature groups
EMBEDDING_WEIGHTS = {
    "tags": 1.0,           # Balanced weight for tags (109 features)
    "musical_features": 1.0,  # Base features: energy, brightness, tempo, key (27 features)
    "features": 0.7  # Features: rhythm, harmony, timbre, dynamics (54 features)
}

# Block-wise weights for balanced embedding (applied after L2 normalization of each block)
EMBEDDING_BLOCK_WEIGHTS = {
    "tags": 1.0,           # Base level - tags carry maximum info
    "tempo": 0.5,          # Single number, reduce to avoid noise
    "energy_brightness": 0.7,  # Energy/brightness values
    "rhythm": 0.6,         # Rhythm features (onset, beat)
    "harmony": 0.8,        # Harmony features (chroma, tonnetz, key_clarity)
    "timbre": 0.7,         # Timbre features (spectral)
    "dynamics": 0.6        # Dynamics features (loudness, LUFS)
}

# Normalization ranges for embedding features
EMBEDDING_NORMALIZATION = {
    "tempo": {"min": 60, "max": 200},      # BPM range
    "lufs": {"min": -60, "max": 0},        # LUFS range
    "energy_value": {"min": 0, "max": 1},  # Already normalized
    "brightness_value": {"min": 0, "max": 12000},  # Hz range (typical for music)
    "onset_density": {"min": 0, "max": 10},    # Events per second
    "percussive_harmonic_ratio": {"min": 0, "max": 1},  # Already normalized
    "key_clarity": {"min": 0, "max": 1},   # Already normalized
    "spectral_flatness": {"min": 0, "max": 1},  # Already normalized
    "spectral_bandwidth_mean": {"min": 0, "max": 8000},  # Hz range (typical for music)
    "spectral_bandwidth_std": {"min": 0, "max": 2000},   # Hz std range
    "zero_crossing_rate_mean": {"min": 0, "max": 1},  # Already normalized
    "zero_crossing_rate_std": {"min": 0, "max": 0.5},   # ZCR std range
    "dynamic_range_db": {"min": 0, "max": 80},   # dB range (expanded for compressed music)
    "loudness": {"min": -60, "max": 0},     # dB range for mean/min/max
    "loudness_std": {"min": 0, "max": 20},  # dB std range
    "loudness_range": {"min": 0, "max": 40},  # dB range (empirical)
    "beat_histogram_mean": {"min": 60, "max": 200},  # BPM range
    "beat_histogram_std": {"min": 0, "max": 80},     # BPM variability (empirical)
    "tonnetz": {"min": -1.0, "max": 1.0},  # Theoretical range for tonnetz components
}

# Display aliases for tags
DISPLAY_ALIASES = {
    "Violin, fiddle": "Violin",
    "Whoosh, swoosh": "Whoosh",
    "Audio clipping": "Clipping",
    "A capella": "A cappella",
    "Rhythm and blues": "R&B",
    "Keyboard (musical)": "Keyboard",
    "Christian music": "Inspirational",
}
