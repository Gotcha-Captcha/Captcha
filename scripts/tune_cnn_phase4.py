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

def preprocess_whole_image(img):
    """Preprocess entire CAPTCHA image"""
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        img_gray = rgb2gray(img)
    else:
        img_gray = img
    
    # Resize to 50x200 (original CAPTCHA proportions)
    img_resized = resize(img_gray, (50, 200))
    return img_resized

# Multi-Character CNN
class MultiCharCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super(MultiCharCNN, self).__init__()
        
        # Shared feature extractor (based on Deeper CNN from Phase 2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # After 3 pools: 50->25->12->6, 200->100->50->25
        # Feature map size: 128 x 6 x 25
        feature_size = 128 * 6 * 25
        
        # Shared dense layer
        self.fc_shared = nn.Linear(feature_size, 512)
        
        # 5 separate classifier heads (one per character position)
        self.char1_head = nn.Linear(512, num_classes)
        self.char2_head = nn.Linear(512, num_classes)
        self.char3_head = nn.Linear(512, num_classes)
        self.char4_head = nn.Linear(512, num_classes)
        self.char5_head = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Shared feature extraction
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc_shared(x))
        x = self.dropout(x)
        
        # 5 separate predictions
        out1 = self.char1_head(x)
        out2 = self.char2_head(x)
        out3 = self.char3_head(x)
        out4 = self.char4_head(x)
        out5 = self.char5_head(x)
        
        return out1, out2, out3, out4, out5

def train_model(model, X_train, y_train, X_test, y_test,
                lr=0.002, batch_size=32, epochs=30):
    """Train multi-output model"""
    
    # y_train/y_test shape: (N, 5) - 5 characters per sample
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
    best_full_word_acc = 0
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            out1, out2, out3, out4, out5 = model(inputs)
            
            # Multi-task loss (average of 5 character losses)
            loss = (
                criterion(out1, labels[:, 0]) +
                criterion(out2, labels[:, 1]) +
                criterion(out3, labels[:, 2]) +
                criterion(out4, labels[:, 3]) +
                criterion(out5, labels[:, 4])
            ) / 5
            
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        char_correct = [0, 0, 0, 0, 0]
        full_word_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                out1, out2, out3, out4, out5 = model(inputs)
                
                _, pred1 = torch.max(out1.data, 1)
                _, pred2 = torch.max(out2.data, 1)
                _, pred3 = torch.max(out3.data, 1)
                _, pred4 = torch.max(out4.data, 1)
                _, pred5 = torch.max(out5.data, 1)
                
                # Per-character accuracy
                char_correct[0] += (pred1 == labels[:, 0]).sum().item()
                char_correct[1] += (pred2 == labels[:, 1]).sum().item()
                char_correct[2] += (pred3 == labels[:, 2]).sum().item()
                char_correct[3] += (pred4 == labels[:, 3]).sum().item()
                char_correct[4] += (pred5 == labels[:, 4]).sum().item()
                
                # Full word accuracy
                full_word_match = (
                    (pred1 == labels[:, 0]) &
                    (pred2 == labels[:, 1]) &
                    (pred3 == labels[:, 2]) &
                    (pred4 == labels[:, 3]) &
                    (pred5 == labels[:, 4])
                )
                full_word_correct += full_word_match.sum().item()
                
                val_total += labels.size(0)
        
        avg_char_acc = sum(char_correct) / (5 * val_total) * 100
        full_word_acc = full_word_correct / val_total * 100
        
        if avg_char_acc > best_val_acc:
            best_val_acc = avg_char_acc
        if full_word_acc > best_full_word_acc:
            best_full_word_acc = full_word_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Char Acc: {avg_char_acc:.2f}% Full Word: {full_word_acc:.2f}%")
    
    return best_val_acc, best_full_word_acc

def main():
    print("=" * 60)
    print("Phase 4: Multi-Character Prediction")
    print("=" * 60)
    print("\nGoal: Predict all 5 characters of CAPTCHA")
    print("Using Phase 2 best architecture (Deeper CNN)")
    
    # Load data
    print("\nLoading dataset...")
    path = kagglehub.dataset_download("fournierp/captcha-version-2-images")
    images_dir = Path(path) / "samples"
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    
    print(f"Found {len(image_files)} images")
    
    X = []
    y = []

    print("Processing whole CAPTCHA images...")
    
    for path in tqdm(image_files):
        try:
            img = imread(str(path))
            label_text = path.stem
            
            # Must be exactly 5 characters
            if len(label_text) != 5:
                continue
            
            img_processed = preprocess_whole_image(img)
            X.append(img_processed)
            y.append(list(label_text))  # ['a', 'b', 'c', 'd', 'e']
            
        except Exception:
            continue

    X = np.array(X, dtype=np.float32)
    X = X.reshape(-1, 1, 50, 200)
    y = np.array(y)  # Shape: (N, 5)
    
    print(f"\nData Shape: {X.shape}")
    print(f"Labels Shape: {y.shape}")
    
    # Encode each character position separately
    le = LabelEncoder()
    
    # Fit on all characters
    all_chars = y.flatten()
    le.fit(all_chars)
    
    print(f"Character classes: {le.classes_}")
    print(f"Number of classes: {len(le.classes_)}")
    
    # Encode each position
    y_enc = np.zeros((len(y), 5), dtype=int)
    for i in range(5):
        y_enc[:, i] = le.transform(y[:, i])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Create model
    model = MultiCharCNN(len(le.classes_), dropout=0.2)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train
    print("\nStarting Training...")
    char_acc, word_acc = train_model(
        model, X_train, y_train, X_test, y_test,
        lr=0.002, batch_size=32, epochs=30
    )
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Per-Character Accuracy: {char_acc:.2f}%")
    print(f"Best Full-Word Accuracy: {word_acc:.2f}%")
    
    # Compare with SVM baseline
    svm_char_acc = 45.61
    svm_word_acc = (svm_char_acc / 100) ** 5 * 100
    
    print(f"\nSVM Baseline:")
    print(f"  Per-Character: {svm_char_acc:.2f}%")
    print(f"  Full-Word: {svm_word_acc:.2f}%")
    
    print(f"\nCNN Improvement:")
    print(f"  Per-Character: {char_acc - svm_char_acc:+.2f}%")
    print(f"  Full-Word: {word_acc - svm_word_acc:+.2f}%")
    
    # Save model
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    torch.save(model.state_dict(), 'models/captcha_multichar_cnn.pth')
    joblib.dump(le, 'models/label_encoder_multichar.joblib')
    
    # Save results
    results = {
        'per_char_accuracy': char_acc,
        'full_word_accuracy': word_acc,
        'svm_baseline_char': svm_char_acc,
        'svm_baseline_word': svm_word_acc
    }
    
    df = pd.DataFrame([results])
    df.to_csv('results/phase4_results.csv', index=False)
    
    print(f"\n✅ Model saved to models/captcha_multichar_cnn.pth")
    print(f"✅ Results saved to results/phase4_results.csv")

if __name__ == "__main__":
    main()
