import os
import json
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

# Import the verified predictor from inference.py
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from scripts.inference import RecaptchaPredictor
from app.core.config import get_v2_dataset_path

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')

# Paths
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "cnn_best_model.pth"
OUTPUT_PATH = MODELS_DIR / "model_metadata_v2.json"

def evaluate(predictor):
    data_path_str = get_v2_dataset_path()
    if not data_path_str:
        raise FileNotFoundError("Dataset path not found")
    
    dataset_root = Path(data_path_str)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_root}")

    print(f"Evaluating on dataset at {dataset_root}...")
    
    # Get all images with their ground truth categories
    all_images = []
    all_labels = []
    
    for category_dir in dataset_root.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        
        category_name = category_dir.name
        if category_name not in predictor.classes:
            print(f"⚠️ Skipping {category_name} (not in model classes)")
            continue
            
        category_idx = predictor.classes.index(category_name)
        
        for img_file in category_dir.glob("*.jpg"):
            all_images.append(img_file)
            all_labels.append(category_idx)
        for img_file in category_dir.glob("*.png"):
            all_images.append(img_file)
            all_labels.append(category_idx)
    
    print(f"Total images to evaluate: {len(all_images)}")
    
    # Predict
    all_preds = []
    running_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    for img_path, true_label in tqdm(zip(all_images, all_labels), total=len(all_images), desc="Evaluating"):
        result = predictor.predict(img_path)
        if result and result["class"] in predictor.classes:
            pred_idx = predictor.classes.index(result["class"])
            all_preds.append(pred_idx)
            
            # Calculate loss (approximate, since we don't have raw logits)
            # Skip loss calculation for simplicity or use dummy
        else:
            all_preds.append(-1) # Invalid prediction
    
    # Calculate Metrics
    valid_preds = [(p, l) for p, l in zip(all_preds, all_labels) if p != -1]
    preds_only = [p for p, _ in valid_preds]
    labels_only = [l for _, l in valid_preds]
    
    correct = sum(p == l for p, l in zip(preds_only, labels_only))
    accuracy = correct / len(labels_only) if labels_only else 0
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_only, preds_only, average='weighted', zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total_samples": len(all_images),
        "valid_predictions": len(valid_preds)
    }

def main():
    try:
        print(f"Using device: {DEVICE}")
        
        # Use the verified RecaptchaPredictor from inference.py
        predictor = RecaptchaPredictor(str(MODEL_PATH), device=str(DEVICE))
        
        metrics = evaluate(predictor)
        
        # Format Metadata
        metadata = {
            "accuracy": f"{metrics['accuracy']*100:.2f}%",
            "char_accuracy": f"{metrics['accuracy']*100:.2f}%",
            "precision": f"{metrics['precision']:.2f}",
            "recall": f"{metrics['recall']:.2f}",
            "f1_score": f"{metrics['f1_score']:.2f}",
            "loss_value": "N/A", # Not calculated in this approach
            "type": "EfficientNet-B1 (Verified)",
            "loss": "CrossEntropy",
            "features": "Image Classification",
            "preprocessing": "Resize (224x224), Normalization",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "classes": predictor.classes,
            "total_samples": metrics['total_samples'],
            "valid_predictions": metrics['valid_predictions']
        }
        
        # Save
        with open(OUTPUT_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
            
        print(f"✅ Metadata saved to {OUTPUT_PATH}")
        print(json.dumps(metadata, indent=4))
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
