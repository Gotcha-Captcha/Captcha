import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import kagglehub
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cpu')
print(f"Using device: {device}")

def preprocess_image(img, target_size):
    """Preprocess with configurable size"""
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        img_gray = rgb2gray(img)
    else:
        img_gray = img
    
    img_resized = resize(img_gray, target_size)
    return img_resized

# Deeper CNN - adapted for different input sizes
class DeeperCNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.2):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate size after 3 pooling layers
        size_after_pools = input_size // 8  # 3 pools: /2/2/2 = /8
        fc_input_size = 128 * size_after_pools * size_after_pools
        
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.size_after_pools = size_after_pools
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * self.size_after_pools * self.size_after_pools)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, X_train, y_train, X_test, y_test,
                lr=0.002, batch_size=32, epochs=30):
    """Train model"""
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Val: {val_acc:.2f}%")
    
    training_time = time.time() - start_time
    
    return best_val_acc, training_time

def main():
    print("=" * 60)
    print("Phase 3: Input Resolution Scaling")
    print("=" * 60)
    print("\nUsing best configuration from Phase 1 & 2:")
    print("  - Architecture: Deeper CNN (3 Conv layers)")
    print("  - Learning Rate: 0.002")
    print("  - Dropout: 0.2")
    print("  - Batch Size: 32")
    
    # Load data
    print("\nLoading dataset...")
    path = kagglehub.dataset_download("fournierp/captcha-version-2-images")
    images_dir = Path(path) / "samples"
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    
    print(f"Found {len(image_files)} images")
    
    # Test different input sizes
    input_sizes = [28, 32, 48]
    results = []
    
    print("\n" + "=" * 60)
    print("Testing Input Sizes")
    print("=" * 60)
    
    for size in input_sizes:
        print(f"\n[{len(results)+1}/3] Testing input size: {size}x{size}")
        
        X = []
        y = []
        
        print(f"  Processing images...")
        for path in tqdm(image_files, desc=f"  {size}x{size}"):
            try:
                img = imread(str(path))
                label_text = path.stem
                
                img_processed = preprocess_image(img, (size, size))
                X.append(img_processed)
                y.append(label_text[0])
                
            except Exception:
                continue

        X = np.array(X, dtype=np.float32)
        X = X.reshape(-1, 1, size, size)
        y = np.array(y)
        
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        
        print(f"  Data shape: {X.shape}")
        
        # Create model
        model = DeeperCNN(size, len(le.classes_), dropout=0.2)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params:,}")
        
        # Train
        print(f"  Training...")
        val_acc, train_time = train_model(
            model, X_train, y_train, X_test, y_test,
            lr=0.002, batch_size=32, epochs=30
        )
        
        results.append({
            'input_size': f'{size}x{size}',
            'parameters': total_params,
            'val_accuracy': val_acc,
            'training_time_sec': train_time
        })
        
        print(f"  → Best Accuracy: {val_acc:.2f}%")
        print(f"  → Training Time: {train_time:.1f}s ({train_time/60:.1f}m)")
    
    # Save results
    df = pd.DataFrame(results)
    df = df.sort_values('val_accuracy', ascending=False)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # Save to CSV
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    df.to_csv('results/phase3_results.csv', index=False)
    print(f"\n✅ Results saved to results/phase3_results.csv")
    
    # Best configuration
    best = df.iloc[0]
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(f"Input Size: {best['input_size']}")
    print(f"Parameters: {int(best['parameters']):,}")
    print(f"Validation Accuracy: {best['val_accuracy']:.2f}%")
    print(f"Training Time: {best['training_time_sec']:.1f}s")
    
    # Compare with Phase 2
    phase2_best = 90.19
    improvement = best['val_accuracy'] - phase2_best
    print(f"\nPhase 2 Best (28x28): {phase2_best:.2f}%")
    print(f"Phase 3 Improvement: {improvement:+.2f}%")

if __name__ == "__main__":
    main()
