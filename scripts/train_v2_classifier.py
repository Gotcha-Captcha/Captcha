import os
import argparse
import shutil
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import kagglehub

# ============================
# 1. Configuration & Constants
# ============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Force CPU on Mac IF MPS is unstable, but usually MPS is fine for CNNs. 
# Safe bet is CPU or MPS. efficientnet might differ.
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Using MPS (Metal Performance Shaders) acceleration")
else:
    print(f"Using device: {DEVICE}")

IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ============================
# 2. Data Preparation
# ============================
def prepare_data(data_root="data"):
    """
    Ensures data exists at data_root/samples_v2/images.
    Downloads from Kaggle if missing.
    """
    v2_dest = Path(data_root) / "samples_v2" / "images"
    
    if v2_dest.exists() and any(v2_dest.iterdir()):
        print(f"✅ Dataset found at {v2_dest}")
        return v2_dest
        
    print(f"Dataset not found at {v2_dest}. Downloading...")
    try:
        # Download from Kaggle
        path = kagglehub.dataset_download("mikhailma/test-dataset")
        print(f"Downloaded to cache: {path}")
        
        # Copy to local project data dir for structure
        v2_dest.parent.mkdir(parents=True, exist_ok=True)
        if v2_dest.exists():
            shutil.rmtree(v2_dest)
            
        # The dataset structure from kaggle might be specific, let's copy the content
        # Check if 'path' contains folders directly or an 'images' folder
        # For 'mikhailma/test-dataset', based on download_datasets.py logic, it seems direct.
        shutil.copytree(path, v2_dest)
        print(f"✅ Copied to {v2_dest}")
        return v2_dest
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        # Fallback to local if running in Docker/Expected Env
        return v2_dest

def perform_eda(dataset_path):
    """
    Prints class distribution and checks for corrupt images.
    """
    print("\n--- EDA: Dataset Statistics ---")
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    print(f"Classes ({len(classes)}): {classes}")
    
    class_counts = {}
    total_images = 0
    
    for cls in classes:
        cls_dir = dataset_path / cls
        count = len(list(cls_dir.glob("*")))
        class_counts[cls] = count
        total_images += count
        
    print(f"Total Images: {total_images}")
    print("Class Distribution:")
    for cls, count in class_counts.items():
        ratio = count / total_images
        print(f"  - {cls}: {count} ({ratio:.1%})")
        
    return classes, class_counts

# ============================
# 3. Training Logic
# ============================
def train_model(args):
    dataset_path = prepare_data()
    classes, class_counts = perform_eda(dataset_path)
    
    # Transforms
    # Helper to clean/check images? ImageFolder deals with basic loading.
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset
    full_dataset = datasets.ImageFolder(root=str(dataset_path))
    
    # Train/Val Split (indices)
    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        stratify=full_dataset.targets,
        random_state=42
    )
    
    # Subsets with different transforms
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    # Hack to apply different transform to subset? 
    # Proper way is creating two Datasets or Wrapper. 
    # For simplicity, we'll apply transform in the Dataset but ImageFolder applies ONE transform.
    # Pattern: Re-init ImageFolder with transform, or custom wrapper.
    # Let's use two ImageFolders for simplicity (Valid since data is same file structure)
    
    train_ds = datasets.ImageFolder(str(dataset_path), transform=train_transform)
    val_ds = datasets.ImageFolder(str(dataset_path), transform=val_transform)
    
    train_subset = torch.utils.data.Subset(train_ds, train_idx)
    val_subset = torch.utils.data.Subset(val_ds, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model Setup
    print(f"\nInitializing EfficientNet-B0 for {len(classes)} classes...")
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    
    # Replace Head
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(classes))
    
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.cv_lr)
    
    # MLflow
    mlflow.set_experiment("Captcha_V2_EfficientNet")
    
    with mlflow.start_run():
        mlflow.log_params({
            "model": "EfficientNet-B0",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.cv_lr
        })
        
        best_acc = 0.0
        
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            
            # Train
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(train_loader, desc="Training"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            epoch_loss = running_loss / len(train_subset)
            epoch_acc = correct / total
            
            # Val
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc="Validation"):
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_epoch_loss = val_loss / len(val_subset)
            val_epoch_acc = val_correct / val_total
            
            print(f"  Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            print(f"  Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
            
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_acc", epoch_acc, step=epoch)
            mlflow.log_metric("val_loss", val_epoch_loss, step=epoch)
            mlflow.log_metric("val_acc", val_epoch_acc, step=epoch)
            
            # Save Best Model
            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), "models/efficientnet_v2_best.pth")
                print(f"  🌟 New Best Model Saved! ({best_acc:.4f})")
                
        # Save Final Metadata
        print("Training Complete.")
        # Log classes to artifacts for inference mapping
        with open("models/v2_classes.txt", "w") as f:
            f.write("\n".join(classes))
        mlflow.log_artifact("models/v2_classes.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cv_lr", type=float, default=0.001)
    args = parser.parse_args()
    
    train_model(args)
