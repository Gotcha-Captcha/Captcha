import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from tqdm import tqdm
import kagglehub
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary

# ============================
# 1. Configuration & Constants
# ============================
# Force CPU for reliability on Mac, as MPS can sometimes have operator mismatches
DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")

IMG_WIDTH = 200
IMG_HEIGHT = 50
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.0003  # Lowered for better CTC stability

# Vocabulary will be built dynamically from labels
VOCAB = ""
CHAR_TO_INT = {}
INT_TO_CHAR = {}
NUM_CLASSES = 0

# ============================
# 2. Data Loading & Preprocessing
# ============================
class CaptchaDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_text = img_path.stem
        
        # Preprocess image
        img = imread(str(img_path))
        if img.ndim == 3:
            if img.shape[2] == 4: img = img[..., :3]
            img = rgb2gray(img)
        
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), anti_aliasing=True)
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
        
        # Standardize for PyTorch: (C, H, W)
        img_tensor = torch.FloatTensor(img).unsqueeze(0)
        
        # Prepare label for CTC
        target = [CHAR_TO_INT[c] for c in label_text]
        target_tensor = torch.LongTensor(target)
        
        return img_tensor, target_tensor, len(target)

def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    
    # Pad sequences to same length for DataLoader
    # target_lengths will be used by CTCLoss
    targets_concat = torch.cat(targets)
    target_lengths = torch.IntTensor(target_lengths)
    
    return images, targets_concat, target_lengths

# ============================
# 3. Model Architecture (CRNN)
# ============================
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        
        # Feature Extraction (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.ReLU(True)
        )
        
        # Recurrent Layers (Bidirectional LSTM)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(1024, 256, 256),
            BidirectionalLSTM(256, 256, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x: [batch, 1, 50, 200]
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        
        # Conv feature map: [batch, 512, 2, 24]
        # Collapse height: 512 * 2 = 1024
        # Reshape to [width, batch, combined_features] for RNN
        conv = conv.view(b, c * h, w) 
        conv = conv.permute(2, 0, 1) # [24, batch, 1024]
        
        output = self.rnn(conv)
        
        # Output: [seq_len, batch, num_classes] (Logits for LogSoftmax)
        return nn.functional.log_softmax(output, dim=2)

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.embedding = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec) # [T * b, output_dim]
        output = output.view(T, b, -1)
        return output

# ============================
# 4. Training Utilities
# ============================
def decode_ctc(output, int_to_char):
    # output: [T, batch, num_classes]
    output = output.permute(1, 0, 2) # [batch, T, classes]
    _, max_indices = torch.max(output, 2)
    
    decoded_texts = []
    for row in max_indices:
        text = ""
        prev = 0
        for char_idx in row:
            char_idx = char_idx.item()
            if char_idx != 0 and char_idx != prev:
                text += int_to_char.get(char_idx, "")
            prev = char_idx
        decoded_texts.append(text)
    return decoded_texts

def validate(model, dataloader, criterion):
    model.eval()
    losses = []
    correct_words = 0
    total_words = 0
    total_chars = 0
    correct_chars = 0
    
    samples_to_show = 3
    samples_logged = 0
    
    with torch.no_grad():
        for images, targets_concat, target_lengths in dataloader:
            images = images.to(DEVICE)
            
            log_probs = model(images)
            batch_size = images.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.int32)
            
            pos = 0
            original_texts = []
            for length in target_lengths:
                t = targets_concat[pos:pos+length]
                original_texts.append("".join([INT_TO_CHAR[c.item()] for c in t]))
                pos += length

            loss = criterion(log_probs, targets_concat, input_lengths, target_lengths)
            losses.append(loss.item())
            
            decoded_texts = decode_ctc(log_probs.cpu(), INT_TO_CHAR)
            
            for pred, target in zip(decoded_texts, original_texts):
                # Word Accuracy (Exact Match)
                if pred == target:
                    correct_words += 1
                total_words += 1
                
                # Simple Character Accuracy (for early feedback)
                # Count matching characters at same position (naive)
                min_len = min(len(pred), len(target))
                for i in range(min_len):
                    if pred[i] == target[i]:
                        correct_chars += 1
                total_chars += len(target)
                
                # Log a few samples
                if samples_logged < samples_to_show:
                    print(f"    Sample: Target={target} | Pred={pred if pred else '<empty>'}")
                    samples_logged += 1
                
    return np.mean(losses), correct_words / total_words, correct_chars / total_chars

# ============================
# 5. Main Training Loop
# ============================
def main():
    global VOCAB, CHAR_TO_INT, INT_TO_CHAR, NUM_CLASSES
    print(f"Using device: {DEVICE}")
    
    # MLflow Setup
    mlflow.set_experiment("Captcha_CRNN_CTC")
    
    # Data Preparation
    print("Downloading dataset...")
    data_path = kagglehub.dataset_download("fournierp/captcha-version-2-images")
    samples_dir = Path(data_path) / "samples"
    all_files = list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg"))
    
    # Extract vocabulary from all labels
    all_labels = "".join([f.stem for f in all_files])
    VOCAB = "".join(sorted(list(set(all_labels))))
    print(f"Alphabet: {VOCAB}")
    CHAR_TO_INT = {char: i + 1 for i, char in enumerate(VOCAB)}
    INT_TO_CHAR = {i: char for char, i in CHAR_TO_INT.items()}
    NUM_CLASSES = len(VOCAB) + 1  # +1 for blank at index 0
    
    # Split
    train_files, val_files = train_test_split_files(all_files, test_size=0.1)
    
    train_ds = CaptchaDataset(train_files)
    val_ds = CaptchaDataset(val_files)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Model, Criterion, Optimizer
    model = CRNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Summary
    print("\nModel Summary:")
    summary(model, input_size=(BATCH_SIZE, 1, IMG_HEIGHT, IMG_WIDTH))

    train_losses, val_losses, val_word_accs, val_char_accs = [], [], [], []

    with mlflow.start_run():
        mlflow.log_params({
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "architecture": "CRNN_CTC"
        })
        
        print("\nStarting Training...")
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            
            for images, targets_concat, target_lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                images = images.to(DEVICE)
                
                optimizer.zero_grad()
                
                log_probs = model(images)
                batch_size = images.size(0)
                input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.int32)
                
                loss = criterion(log_probs, targets_concat, input_lengths, target_lengths)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            avg_val_loss, val_word_acc, val_char_acc = validate(model, val_loader, criterion)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_word_accs.append(val_word_acc)
            val_char_accs.append(val_char_acc)
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_word_accuracy", val_word_acc, step=epoch)
            mlflow.log_metric("val_char_accuracy", val_char_acc, step=epoch)
            
            print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  Word Acc: {val_word_acc:.2%} | Char Acc: {val_char_acc:.2%}")

            # Save checkoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                save_model(model, f"models/crnn_v5_epoch_{epoch+1}.pth")

        # Save Final Model
        save_model(model, "models/crnn_v5_final.pth")
        
        # MLflow artifact logging
        mlflow.pytorch.log_model(model, "model")
        
        # Save visualization
        plot_training_results(train_losses, val_losses, val_word_accs, val_char_accs)
        mlflow.log_artifacts("docs/visualizations/")

def train_test_split_files(files, test_size=0.1):
    np.random.shuffle(files)
    split_idx = int(len(files) * (1 - test_size))
    return files[:split_idx], files[split_idx:]

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def plot_training_results(train_losses, val_losses, val_word_accs, val_char_accs):
    os.makedirs("docs/visualizations", exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('CTC Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_word_accs, label='Word Accuracy (Exact)', color='blue')
    plt.plot(val_char_accs, label='Char Accuracy (Naive)', color='green', linestyle='--')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("docs/visualizations/training_history.png")
    plt.close()

if __name__ == "__main__":
    main()
