import json
from pathlib import Path
from ..core.config import STATE, MODELS_DIR

def load_metadata():
    meta_path = MODELS_DIR / "model_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                data = json.load(f)
                # Update the specialized metadata dict directly
                STATE["model_metadata"].update(data)
                
                # Update root state properties for quick access
                STATE["last_accuracy"] = data.get("accuracy", "0.00%")
                STATE["last_char_accuracy"] = data.get("char_accuracy", "0.00%")
                STATE["status"] = "Production model metadata loaded"
                print(f"✅ Metadata loaded successfully: {list(data.keys())}")
                return data
        except Exception as e:
            print(f"❌ Failed to load metadata: {e}")
    else:
        print(f"⚠️ No metadata found at {meta_path.absolute()}")
    return None
