import numpy as np
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

# Force CPU
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
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("fournierp/captcha-version-2-images")
    images_dir = Path(path) / "samples"
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    
    # Use all images for better accuracy
    print(f"Found {len(image_files)} images")
    
    X = []
    y = []

    print(f"Processing images...")
    
    for path in tqdm(image_files):
        try:
            img = imread(str(path))
            label_text = path.stem
            
            img_processed = simple_preprocess(img)
            X.append(img_processed)
            y.append(label_text[0])  # First character
            
        except Exception as e:
            continue

    X = np.array(X, dtype=np.float32)
    X = X.reshape(-1, 1, 28, 28)  # PyTorch format: (N, C, H, W)
    y = np.array(y)
    
    print(f"\nData Shape: {X.shape}")
    print(f"Unique labels: {np.unique(y)}")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Model
    model = SimpleCNN(len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nStarting Training...")
    epochs = 30  # Increased for better convergence
    
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
        
        print(f'Epoch [{epoch+1}/{epochs}] '
              f'Loss: {running_loss/len(train_loader):.4f} '
              f'Train Acc: {train_acc:.2f}% '
              f'Val Acc: {val_acc:.2f}%')
    
    print(f"\n✅ Final Test Accuracy: {val_acc:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), 'models/captcha_cnn_pytorch.pth')
    joblib.dump(le, 'models/label_encoder_pytorch.joblib')
    print("Model saved!")

if __name__ == "__main__":
    main()
