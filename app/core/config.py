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
    docker_v2 = Path("/app/data/samples_v2/images")
    # Priority 2: User's Downloads folder (images subfolder)
    downloads_v2_img = Path("/Users/parkyoungdu/Downloads/samples_v2/images")
    # Priority 3: User's project root fallback
    local_v2_img = Path(__file__).parent.parent.parent / "samples_v2" / "images"
    
    if docker_v2.exists():
        return str(docker_v2)
    if downloads_v2_img.exists():
        return str(downloads_v2_img)
    if local_v2_img.exists():
        return str(local_v2_img)
    return ""
