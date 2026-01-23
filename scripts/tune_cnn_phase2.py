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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cpu')
print(f"Using device: {device}")

def simple_preprocess(img):
    """Simplified preprocessing"""
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        img_gray = rgb2gray(img)
    else:
        img_gray = img
    
    img_resized = resize(img_gray, (28, 28))
    return img_resized

# Architecture Variant A: Deeper (3 Conv layers)
class DeeperCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After 3 pools: 28 -> 14 -> 7 -> 3 (with padding)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28->14
        x = self.pool(self.relu(self.conv2(x)))  # 14->7
        x = self.pool(self.relu(self.conv3(x)))  # 7->3
        x = x.view(-1, 128 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Architecture Variant B: Wider (more filters)
class WiderCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super(WiderCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After 2 pools: 28 -> 14 -> 7
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 128 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Architecture Variant C: Larger kernels
class LargerKernelCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super(LargerKernelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        # After 2 pools: 28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, X_train, y_train, X_test, y_test,
                lr=0.002, batch_size=32, epochs=30):
    """Train model with given architecture"""
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        
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
            print(f"  Epoch [{epoch+1}/{epochs}] Train: {train_acc:.2f}% Val: {val_acc:.2f}%")
    
    return best_val_acc

def main():
    print("=" * 60)
    print("Phase 2: Architecture Improvements")
    print("=" * 60)
    print("\nUsing Phase 1 optimal hyperparameters:")
    print("  - Learning Rate: 0.002")
    print("  - Dropout: 0.2")
    print("  - Batch Size: 32")
    
    # Load data
    print("\nLoading dataset...")
    path = kagglehub.dataset_download("fournierp/captcha-version-2-images")
    images_dir = Path(path) / "samples"
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    
    print(f"Found {len(image_files)} images")
    
    X = []
    y = []

    print("Processing images...")
    
    for path in tqdm(image_files):
        try:
            img = imread(str(path))
            label_text = path.stem
            
            img_processed = simple_preprocess(img)
            X.append(img_processed)
            y.append(label_text[0])
            
        except Exception:
            continue

    X = np.array(X, dtype=np.float32)
    X = X.reshape(-1, 1, 28, 28)
    y = np.array(y)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Test architectures
    architectures = {
        'Deeper (3 Conv)': DeeperCNN(len(le.classes_), dropout=0.2),
        'Wider (More Filters)': WiderCNN(len(le.classes_), dropout=0.2),
        'Larger Kernels (5x5)': LargerKernelCNN(len(le.classes_), dropout=0.2)
    }
    
    results = []
    
    print("\n" + "=" * 60)
    print("Testing Architectures")
    print("=" * 60)
    
    for name, model in architectures.items():
        print(f"\n[{len(results)+1}/3] Testing: {name}")
        
        val_acc = train_model(
            model, X_train, y_train, X_test, y_test,
            lr=0.002, batch_size=32, epochs=30
        )
        
        results.append({
            'architecture': name,
            'val_accuracy': val_acc
        })
        
        print(f"  → Best Validation Accuracy: {val_acc:.2f}%")
    
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
    df.to_csv('results/phase2_results.csv', index=False)
    print(f"\n✅ Results saved to results/phase2_results.csv")
    
    # Best architecture
    best = df.iloc[0]
    print("\n" + "=" * 60)
    print("BEST ARCHITECTURE")
    print("=" * 60)
    print(f"Name: {best['architecture']}")
    print(f"Validation Accuracy: {best['val_accuracy']:.2f}%")
    
    # Compare with Phase 1
    phase1_best = 85.51
    improvement = best['val_accuracy'] - phase1_best
    print(f"\nPhase 1 Best: {phase1_best:.2f}%")
    print(f"Phase 2 Improvement: {improvement:+.2f}%")

if __name__ == "__main__":
    main()
