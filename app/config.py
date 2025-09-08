"""Configuration and constants for BW Analyzer"""

# BW tag groups
BW_TAG_GROUPS = {
    "genre": [
        "Pop music", "Rock music", "Hip hop music", "Jazz", "Blues", "Bluegrass",
        "Country", "Electronic music", "Classical music", "Folk music", "Reggae",
        "Rhythm and blues", "Soul music", "Gospel music", "Christian music",
        "Ambient music", "Techno", "Grime music", "House music", "Trance music", 
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
        "Rapping", "Whistling", "Beatboxing",
        "Background vocals"
    ],
    "atmosphere": [
        "Drone", "Noise", "Buzz", "Hiss", "Hum", "Rumble",
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

# Display aliases for tags
DISPLAY_ALIASES = {
    "Violin, fiddle": "Violin",
    "Whoosh, swoosh": "Whoosh",
    "Audio clipping": "Clipping",
    "Background vocals": "Backing vocals",
    "A capella": "A cappella",
    "Rhythm and blues": "R&B",
    "Keyboard (musical)": "Keyboard",
    "Christian music": "Inspirational",
}
