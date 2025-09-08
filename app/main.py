import os
import time
import tempfile
import numpy as np
import json
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import librosa

# Module imports
from .core.module_registry import ModuleRegistry
from .modules import (
    TagsModule, TempoModule, KeyModule, FeaturesModule
)
from .types.analysis import AnalysisRequest, AnalysisResponse, AnalysisModule
from .config import BW_TAG_GROUPS, DISPLAY_ALIASES

# Load tags descriptions
def load_tags_descriptions():
    """Load tags descriptions from JSON file"""
    try:
        with open(os.path.join(os.path.dirname(__file__), 'tags_info.json'), 'r', encoding='utf-8') as f:
            return {item['name']: item['description'] for item in json.load(f)}
    except Exception as e:
        print(f"Warning: Could not load tags descriptions: {e}")
        return {}

TAGS_DESCRIPTIONS = load_tags_descriptions()

def prettify_label(label_raw: str, group_name: Optional[str] = None) -> str:
    """Convert label to pretty format"""
    if label_raw in DISPLAY_ALIASES:
        return DISPLAY_ALIASES[label_raw]
    if group_name == "genre" and label_raw.endswith(" music"):
        return label_raw[:-6]
    if "," in label_raw:
        return label_raw.split(",", 1)[0].strip()
    return label_raw

# FastAPI initialization
app = FastAPI(
    title="BW Analyzer 3.0 - Modular",
    description="Modular audio analysis API with selective processing",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module registry initialization
module_registry = ModuleRegistry()

# Module registration
module_registry.register(TagsModule())
module_registry.register(TempoModule())
module_registry.register(KeyModule())
module_registry.register(FeaturesModule())

@app.get("/")
async def root():
    """Service information"""
    available_modules = module_registry.get_available_modules()
    return {
        "service": "BW Analyzer 3.0 - Modular",
        "version": "3.0.0",
        "available_modules": [m.value for m in available_modules],
        "endpoints": {
            "analyze": "POST /analyze - Audio analysis with module selection",
            "modules": "GET /modules - List of available modules",
            "tags": "GET /tags - List of tags",
            "docs": "GET /docs - API documentation"
        }
    }

@app.get("/modules")
async def get_available_modules():
    """Get list of available modules"""
    modules = module_registry.get_available_modules()
    return [m.value for m in modules]

@app.get("/tags")
async def get_canonical_tag_groups(desc: bool = Query(False, description="Include descriptions for tags")):
    """Get canonical tag groups"""
    if desc:
        # Return with descriptions
        result: Dict[str, List[Dict[str, str]]] = {}
        for g, tags in BW_TAG_GROUPS.items():
            result[g] = []
            for tag in tags:
                result[g].append({
                    "name": prettify_label(tag, g),  # Normalized value using prettify_label
                    "description": TAGS_DESCRIPTIONS.get(tag, "")
                })
        
        return {
            "total": sum(len(tags) for tags in BW_TAG_GROUPS.values()),
            "categories": {key: len(tags) for key, tags in BW_TAG_GROUPS.items()},
            "list": result
        }
    else:
        # Return simple list
        pretty: Dict[str, List[str]] = {}
        for g, tags in BW_TAG_GROUPS.items():
            pretty[g] = [prettify_label(t, g) for t in tags]
        
        return {
            "total": sum(len(tags) for tags in BW_TAG_GROUPS.values()),
            "categories": {key: len(tags) for key, tags in BW_TAG_GROUPS.items()},
            "list": pretty
        }

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    modules: str = "tags,tempo,key,features"
):
    """
    Audio analysis with module selection
    
    modules: comma-separated list of modules (tags,tempo,key,features)
    """
    t0 = time.time()
    
    # Parse requested modules
    try:
        requested_modules = []
        for module_str in modules.split(','):
            module_str = module_str.strip()
            if module_str in [m.value for m in AnalysisModule]:
                requested_modules.append(AnalysisModule(module_str))
            else:
                raise ValueError(f"Unknown module: {module_str}")
    except ValueError as e:
        raise HTTPException(400, f"Invalid modules parameter: {e}")
    
    # File type validation
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        raise HTTPException(400, "Unsupported file type")
    
    temp_path = None
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            suffix=ext,
            delete=False
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name
        
        # Load audio
        y, sr = librosa.load(temp_path, sr=None, mono=True)
        
        # Analysis with selected modules
        results = module_registry.analyze_with_modules(
            y, sr, temp_path, requested_modules
        )
        
        # Form response in new format
        response_data = {}
        
        # Add results to root (tempo and key)
        if AnalysisModule.TEMPO in requested_modules and results.get('tempo'):
            response_data["tempo"] = results['tempo']['tempo']
            
        if AnalysisModule.KEY in requested_modules and results.get('key'):
            response_data["key"] = results['key']['key']
        
        # Add other modules as is
        if AnalysisModule.TAGS in requested_modules:
            response_data["tags"] = results.get('tags')
            
        # Process features module - move energy and brightness to top level
        if AnalysisModule.FEATURES in requested_modules:
            features = results.get('features', {})
            if 'energy' in features:
                response_data["energy"] = features['energy']
            if 'brightness' in features:
                response_data["brightness"] = features['brightness']
        
        
        return JSONResponse(response_data)
        
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
