from pathlib import Path
import os
import glob

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"
MODELS_DIR = BASE_DIR.parent / "models"

# Global state to keep track of training
STATE = {
    "training_in_progress": False,
    "last_accuracy": "0.00%", 
    "last_char_accuracy": "0.00%",
    "progress": 0,
    "status": "Ready",
    "dataset_path": "",
    "model_metadata": {
        "accuracy": "N/A",
        "char_accuracy": "N/A",
        "type": "N/A",
        "loss": "N/A",
        "features": "N/A",
        "preprocessing": "N/A"
    }
}

def get_dataset_path():
    if not STATE["dataset_path"]:
        paths = glob.glob("/Users/parkyoungdu/.cache/kagglehub/datasets/fournierp/captcha-version-2-images/versions/*/samples")
        if paths:
            STATE["dataset_path"] = paths[0]
        else:
            STATE["dataset_path"] = "samples"
    return STATE["dataset_path"]
