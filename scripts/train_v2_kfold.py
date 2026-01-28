import os
import argparse
import json
import time
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import KFold
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

# Add project root to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from app.core.config import get_v2_dataset_path
except ImportError:
    # Fallback if running outside expected structure
    def get_v2_dataset_path():
        return "data/samples_v2"

# ============================
# 1. Configuration & Constants
# ============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')

IMG_SIZE = 224

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            self.best_loss = val_loss
            self.counter = 0

def plot_history(metadata):
    if not metadata or "fold_results" not in metadata:
        return
        
    folds = metadata["fold_results"]
    plt.figure(figsize=(15, 6))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    for f in folds:
        plt.plot(f['history']['train_loss'], alpha=0.3, label=f"Fold {f['fold']} Train")
        plt.plot(f['history']['val_loss'], label=f"Fold {f['fold']} Val")
    plt.title("Loss per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    for f in folds:
        plt.plot(f['history']['train_acc'], alpha=0.3, label=f"Fold {f['fold']} Train")
        plt.plot(f['history']['val_acc'], label=f"Fold {f['fold']} Val")
    plt.title("Accuracy per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    os.makedirs("docs/visualizations", exist_ok=True)
    save_path = "docs/visualizations/v2_training_history.png"
    plt.savefig(save_path)
    print(f"📊 Training history saved to {save_path}")

def train_kfold(args):
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Data
    data_path_str = get_v2_dataset_path()
    if not data_path_str or not os.path.exists(data_path_str):
        print(f"❌ Dataset not found. Please ensure images are available.")
        # Try to use download logic if needed or just fail
        return
        
    print(f"✅ Loading dataset from: {data_path_str}")
    dataset_path = Path(data_path_str)
    
    # Basic Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Load Full Dataset (We will apply transforms via custom wrapper or subset logic)
    # Since ImageFolder applies one transform, we cheat slightly by reloading or using a wrapper.
    # Simple approach: Load ImageFolder with 'val' transform for underlying data, 
    # but that's wrong for training.
    # Proper way for K-Fold: Indicies.
    # We will instantiate two ImageFolders, one for train (augmented) and one for val (clean),
    # pointing to the SAME data directory. Then use Subsets.
    
    full_train_dataset = datasets.ImageFolder(data_path_str, transform=data_transforms['train'])
    full_val_dataset = datasets.ImageFolder(data_path_str, transform=data_transforms['val'])
    
    classes = full_train_dataset.classes
    print(f"Classes: {classes}")
    
    # Export Metadata early
    metadata = {
        "classes": classes,
        "num_classes": len(classes),
        "config": vars(args),
        "dataset_path": data_path_str,
        "fold_results": []
    }
    
    # 2. K-Fold Setup
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    # MLflow
    mlflow.set_experiment("Captcha_V2_KFold")
    
    best_overall_acc = 0.0
    
    # Indices for the whole dataset
    indices = np.arange(len(full_train_dataset))
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\n--- Fold {fold+1}/{args.folds} ---")
        
        # Subsets
        train_subset = Subset(full_train_dataset, train_idx)
        val_subset = Subset(full_val_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # Init Model
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(classes))
        model = model.to(DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        early_stopping = EarlyStopping(patience=10, delta=0.001)
        
        fold_best_acc = 0.0
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        with mlflow.start_run(run_name=f"Fold_{fold+1}"):
            mlflow.log_params(vars(args))
            mlflow.log_param("fold", fold+1)
            
            for epoch in range(args.epochs):
                # Train
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                # Progress bar for Training
                pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
                for inputs, labels in pbar_train:
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
                    
                    # Update pbar description with current loss
                    pbar_train.set_postfix({'loss': loss.item()})

                    
                train_loss = running_loss / len(train_subset)
                train_acc = correct / total
                
                # Val
                model.eval()
                val_loss_accum = 0.0
                val_correct = 0
                val_total = 0
                
                # Progress bar for Validation
                with torch.no_grad():
                    for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss_accum += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_loss = val_loss_accum / len(val_subset)
                val_acc = val_correct / val_total
                
                # History
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f}/{val_loss:.4f} | Acc: {train_acc:.4f}/{val_acc:.4f}")
                
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                
                # Updated Best Model Logic (Per Fold and Overall)
                if val_acc > fold_best_acc:
                    fold_best_acc = val_acc
                
                if val_acc > best_overall_acc:
                    best_overall_acc = val_acc
                    os.makedirs("models", exist_ok=True)
                    torch.save(model.state_dict(), "models/efficientnet_v2_best.pth")
                    print(f"  🌟 New Overall Best Model! ({val_acc:.4f})")
                
                # Early Stopping
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(f"  ⏹️ Early stopping at epoch {epoch+1}")
                    break
            
            # Log Fold Result
            metadata["fold_results"].append({
                "fold": fold + 1,
                "best_val_acc": fold_best_acc,
                "epochs_run": len(history['train_loss']),
                "history": history
            })

    # Save Metadata
    print("\nTraining Complete.")
    print(f"Best Overall Accuracy: {best_overall_acc:.4f}")
    
    with open("models/model_metadata_v2.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("MetaData saved to models/model_metadata_v2.json")
    
    # Save Class List separately for simpler loading
    with open("models/v2_classes.txt", "w") as f:
        f.write("\n".join(classes))

    # Plot History
    plot_history(metadata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50) # Recommendation: 50 with EarlyStop
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    
    train_kfold(args)
