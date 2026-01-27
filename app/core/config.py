from pathlib import Path
import os
import glob

BASE_DIR = Path(__file__).parent.parent.parent
UPLOAD_DIR = BASE_DIR / "app" / "uploads"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
MODELS_DIR = BASE_DIR / "models"

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
        "precision": "N/A",
        "recall": "N/A",
        "f1_score": "N/A",
        "loss_value": "N/A",
        "type": "N/A",
        "loss": "N/A",
        "features": "N/A",
        "preprocessing": "N/A"
    }
}

import kagglehub

def get_dataset_path():
    # Priority 1: Docker Volume Mount Path
    docker_v1 = Path("/app/data/samples_v1")
    # Priority 2: User's Downloads folder (checked via run_command)
    downloads_v1 = Path("/Users/parkyoungdu/Downloads/samples_v1")
    # Priority 3: User's project root fallback
    local_v1 = Path(__file__).parent.parent.parent / "samples_v1"
    
    if docker_v1.exists():
        return str(docker_v1)
    if downloads_v1.exists():
        return str(downloads_v1)
    if local_v1.exists():
        return str(local_v1)
    return "samples"

def get_v2_dataset_path():
    # Priority 1: Docker Volume Mount Path
    paths_to_check = [
        Path("/app/data/samples_v2/images"),
        Path("/Users/parkyoungdu/Downloads/samples_v2/images"),
        Path(__file__).parent.parent.parent / "samples_v2" / "images"
    ]
    
    for p in paths_to_check:
        if p.exists():
            # Check if there's a nested "Google_Recaptcha_V2_Images_Dataset" folder
            nested = p / "Google_Recaptcha_V2_Images_Dataset"
            if nested.exists():
                return str(nested)
            return str(p)
    return ""
