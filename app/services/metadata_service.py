import json
from pathlib import Path
from ..core.config import STATE, BASE_DIR

def load_metadata():
    meta_path = BASE_DIR.parent / "models" / "model_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                data = json.load(f)
                STATE["model_metadata"] = data
                STATE["last_accuracy"] = data.get("accuracy", "0.00%")
                STATE["last_char_accuracy"] = data.get("char_accuracy", "0.00%")
                STATE["status"] = "Pre-trained model loaded"
                return data
        except Exception as e:
            print(f"❌ Failed to load metadata: {e}")
    else:
        print(f"⚠️ No metadata found at {meta_path}")
    return None
