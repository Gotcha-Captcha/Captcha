import json
from pathlib import Path
from ..core.config import STATE, MODELS_DIR

def load_metadata():
    # Load V1 Metadata
    v1_path = MODELS_DIR / "model_metadata_v1.json"
    if not v1_path.exists():
        v1_path = MODELS_DIR / "model_metadata.json" # Fallback
        
    if v1_path.exists():
        try:
            with open(v1_path, "r") as f:
                data = json.load(f)
                STATE["v1_metadata"].update(data)
                STATE["last_accuracy"] = data.get("accuracy", "0.00%")
                STATE["last_char_accuracy"] = data.get("char_accuracy", "0.00%")
                print(f"✅ V1 Metadata loaded from {v1_path.name}")
        except Exception as e:
            print(f"❌ Failed to load V1 metadata: {e}")

    # Load V2 Metadata
    v2_path = MODELS_DIR / "model_metadata_v2.json"
    if v2_path.exists():
        try:
            with open(v2_path, "r") as f:
                data = json.load(f)
                STATE["v2_metadata"].update(data)
                print(f"✅ V2 Metadata loaded from {v2_path.name}")
        except Exception as e:
            print(f"❌ Failed to load V2 metadata: {e}")
    
    STATE["status"] = "Model metadata loaded"
    return STATE["v1_metadata"], STATE["v2_metadata"]
