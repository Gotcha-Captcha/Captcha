import os
import json
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "cnn_best_model.pth"
CLASSES_PATH = MODELS_DIR / "v2_classes.txt"
OUTPUT_PATH = MODELS_DIR / "model_metadata_v2.json"

# Import app config to get correct dataset path
import sys
sys.path.append(str(BASE_DIR))
from app.core.config import get_v2_dataset_path

DATA_DIR = Path(get_v2_dataset_path())

def load_classes():
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Classes file not found at {CLASSES_PATH}")
    with open(CLASSES_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]

def load_model(num_classes):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
    print(f"Loading model from {MODEL_PATH}...")
    # It's EfficientNet-B1 based on block structure
    model = models.efficientnet_b1(weights=None)
    
    # Match the nested classifier structure found in the state_dict (classifier.1.1)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Identity(), # 1.0
        nn.Linear(num_ftrs, num_classes) # 1.1
    )
    
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Handle 'model.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
            
    print(f"Sample keys in new_state_dict: {list(new_state_dict.keys())[:5]}")
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def evaluate(model, classes):
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found at {DATA_DIR}")

    print(f"Evaluating on dataset at {DATA_DIR}...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(str(DATA_DIR), transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    total_loss = running_loss / len(dataset)
    
    # Calculate Metrics
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(dataset)
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        "loss": total_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def main():
    try:
        print(f"Using device: {DEVICE}")
        
        # Determine actual num_classes from model state_dict first
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        actual_num_classes = 0
        for k, v in state_dict.items():
            if 'classifier.1.1.weight' in k or 'classifier.1.weight' in k:
                actual_num_classes = v.size(0)
                break
        
        if actual_num_classes == 0:
            # Fallback to loading from classes file
            classes = load_classes()
            actual_num_classes = len(classes)
        else:
            print(f"Detected {actual_num_classes} classes from model weights.")
            # We still need class names for metadata, but we'll handle the mismatch if any
            classes = load_classes()
            if len(classes) != actual_num_classes:
                print(f"⚠️ Warning: Classes file has {len(classes)} classes, but model has {actual_num_classes}.")
                classes = classes[:actual_num_classes] # Simple truncation for now
        
        model = load_model(actual_num_classes)
        
        metrics = evaluate(model, classes)
        
        # Format Metadata
        metadata = {
            "accuracy": f"{metrics['accuracy']*100:.2f}%",
            "char_accuracy": f"{metrics['accuracy']*100:.2f}%", # Reuse for consistent UI if needed, or omit
            "loss": f"{metrics['loss']:.4f}", # Display string
            "loss_value": f"{metrics['loss']:.4f}", # Raw value string
            "precision": f"{metrics['precision']:.2f}",
            "recall": f"{metrics['recall']:.2f}",
            "f1_score": f"{metrics['f1_score']:.2f}",
            "type": "EfficientNet-B0 (Re-eval)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "classes": classes,
            "dataset_path": str(DATA_DIR)
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
