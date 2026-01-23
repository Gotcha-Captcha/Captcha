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
import itertools

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

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(X_train, y_train, X_test, y_test, num_classes, 
                lr=0.001, dropout=0.2, batch_size=16, epochs=20):
    """Train model with given hyperparameters"""
    
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
    
    # Model
    model = SimpleCNN(num_classes, dropout=dropout).to(device)
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
    
    return best_val_acc

def main():
    print("=" * 60)
    print("Phase 1: Hyperparameter Grid Search")
    print("=" * 60)
    
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
            y.append(label_text[0])  # First character
            
        except Exception:
            continue

    X = np.array(X, dtype=np.float32)
    X = X.reshape(-1, 1, 28, 28)
    y = np.array(y)
    
    print(f"\nData Shape: {X.shape}")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Hyperparameter grid
    learning_rates = [0.0005, 0.001, 0.002]
    dropouts = [0.2, 0.3, 0.4]
    batch_sizes = [16, 32]
    
    results = []
    
    print("\n" + "=" * 60)
    print("Starting Grid Search")
    print("=" * 60)
    
    total_experiments = len(learning_rates) * len(dropouts) * len(batch_sizes)
    experiment_num = 0
    
    for lr, dropout, batch_size in itertools.product(learning_rates, dropouts, batch_sizes):
        experiment_num += 1
        print(f"\n[{experiment_num}/{total_experiments}] Testing: lr={lr}, dropout={dropout}, batch_size={batch_size}")
        
        val_acc = train_model(
            X_train, y_train, X_test, y_test, 
            num_classes=len(le.classes_),
            lr=lr, dropout=dropout, batch_size=batch_size,
            epochs=20
        )
        
        results.append({
            'learning_rate': lr,
            'dropout': dropout,
            'batch_size': batch_size,
            'val_accuracy': val_acc
        })
        
        print(f"  → Validation Accuracy: {val_acc:.2f}%")
    
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
    df.to_csv('results/phase1_results.csv', index=False)
    print(f"\n✅ Results saved to results/phase1_results.csv")
    
    # Best configuration
    best = df.iloc[0]
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(f"Learning Rate: {best['learning_rate']}")
    print(f"Dropout: {best['dropout']}")
    print(f"Batch Size: {int(best['batch_size'])}")
    print(f"Validation Accuracy: {best['val_accuracy']:.2f}%")
    
    # Compare with baseline
    baseline_acc = 82.71
    improvement = best['val_accuracy'] - baseline_acc
    print(f"\nBaseline: {baseline_acc:.2f}%")
    print(f"Improvement: {improvement:+.2f}%")

if __name__ == "__main__":
    main()
