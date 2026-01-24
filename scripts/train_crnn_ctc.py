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
import matplotlib
matplotlib.use("Agg") # Set headless backend
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
EPOCHS = 300
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

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ml_models.crnn import CRNN

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=15, min_delta=0, path='models/crnn_v5_best.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        print(f"  Validation loss decreased. Saving best model to {self.path}")

# ============================
# 4. Training Utilities
# ============================
def save_prediction_samples(model, dataloader, int_to_char, epoch, run_id):
    model.eval()
    os.makedirs("docs/visualizations/samples", exist_ok=True)
    
    with torch.no_grad():
        images, targets_concat, target_lengths = next(iter(dataloader))
        images = images.to(DEVICE)
        log_probs = model(images)
        
        decoded_texts = decode_ctc(log_probs.cpu(), int_to_char)
        
        # Get original texts
        pos = 0
        original_texts = []
        for length in target_lengths:
            t = targets_concat[pos:pos+length]
            original_texts.append("".join([int_to_char[c.item()] for c in t]))
            pos += length
            
        plt.figure(figsize=(12, 8))
        for i in range(min(8, images.size(0))):
            plt.subplot(4, 2, i + 1)
            img = images[i].cpu().squeeze().numpy()
            img = (img * 0.5) + 0.5
            plt.imshow(img, cmap='gray')
            plt.title(f"Target: {original_texts[i]} | Pred: {decoded_texts[i]}")
            plt.axis('off')
            
        plt.tight_layout()
        sample_path = f"docs/visualizations/samples/epoch_{epoch+1}.png"
        plt.savefig(sample_path)
        plt.close()
        return sample_path

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

from sklearn.metrics import precision_recall_fscore_support

def validate(model, dataloader, criterion):
    model.eval()
    losses = []
    correct_words = 0
    total_words = 0
    total_chars = 0
    correct_chars = 0
    
    # For advanced metrics
    all_targets = []
    all_preds = []
    
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
                label_indices = [c.item() for c in t]
                all_targets.extend(label_indices)
                original_texts.append("".join([INT_TO_CHAR[c.item()] for c in t]))
                pos += length

            loss = criterion(log_probs, targets_concat, input_lengths, target_lengths)
            losses.append(loss.item())
            
            decoded_texts = decode_ctc(log_probs.cpu(), INT_TO_CHAR)
            
            # For metrics, we need to match predicted sequence to target sequence
            # This is tricky with CTC, but for accuracy we compare strings
            for pred, target in zip(decoded_texts, original_texts):
                if pred == target:
                    correct_words += 1
                total_words += 1
                
                # Match characters for p/r/f1 (naive alignment)
                pred_indices = [CHAR_TO_INT.get(c, 0) for c in pred]
                target_indices = [CHAR_TO_INT.get(c, 0) for c in target]
                
                # Simple padding/truncation for alignment
                if len(pred_indices) < len(target_indices):
                    pred_indices += [0] * (len(target_indices) - len(pred_indices))
                elif len(pred_indices) > len(target_indices):
                    pred_indices = pred_indices[:len(target_indices)]
                
                all_preds.extend(pred_indices)
                
                # Simple Character Accuracy
                min_len = min(len(pred), len(target))
                for i in range(min_len):
                    if pred[i] == target[i]:
                        correct_chars += 1
                total_chars += len(target)
                
                if samples_logged < samples_to_show:
                    print(f"    Sample: Target={target} | Pred={pred if pred else '<empty>'}")
                    samples_logged += 1
                
    # Calculate advanced metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
                
    return np.mean(losses), correct_words / total_words, correct_chars / total_chars, precision, recall, f1

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
    early_stopping = EarlyStopping(patience=15)
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
            avg_val_loss, val_word_acc, val_char_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_word_accs.append(val_word_acc)
            val_char_accs.append(val_char_acc)
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_word_accuracy", val_word_acc, step=epoch)
            mlflow.log_metric("val_char_accuracy", val_char_acc, step=epoch)
            mlflow.log_metric("val_precision", val_precision, step=epoch)
            mlflow.log_metric("val_recall", val_recall, step=epoch)
            mlflow.log_metric("val_f1_score", val_f1, step=epoch)
            
            print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  Word Acc: {val_word_acc:.2%} | Char Acc: {val_char_acc:.2%}")
            print(f"  P: {val_precision:.2f} | R: {val_recall:.2f} | F1: {val_f1:.2f}")

            # Early Stopping check (Start after Epoch 200)
            if (epoch + 1) >= 200:
                early_stopping(avg_val_loss, model)
                if early_stopping.early_stop:
                    print(f"  Early stopping triggered at Epoch {epoch+1}. Training stopped.")
                    break
            else:
                # Still track best loss to save models/crnn_v5_best.pth even before E200
                early_stopping(avg_val_loss, model)
                early_stopping.early_stop = False # Override to prevent premature stopping

            # Log prediction samples
            if (epoch + 1) % 5 == 0:
                sample_path = save_prediction_samples(model, val_loader, INT_TO_CHAR, epoch, mlflow.active_run().info.run_id)
                mlflow.log_artifact(sample_path, "prediction_samples")

            # Save checkpoint every 10 epochs
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
