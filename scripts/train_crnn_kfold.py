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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchinfo import summary
from sklearn.model_selection import KFold
import sys

# Path injection for 'app' module
sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.ml_models.crnn import CRNN

# 1. Configuration & Constants
DEVICE = torch.device('cpu')
IMG_WIDTH = 200
IMG_HEIGHT = 50
BATCH_SIZE = 32
EPOCHS = 200 # Sufficient for K-fold comparison
LEARNING_RATE = 0.0003
K_FOLDS = 5

# 2. Data Loading & Preprocessing
class CaptchaDataset(Dataset):
    def __init__(self, image_files, char_to_int):
        self.image_files = image_files
        self.char_to_int = char_to_int
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_text = img_path.stem
        
        img = imread(str(img_path))
        if img.ndim == 3:
            if img.shape[2] == 4: img = img[..., :3]
            img = rgb2gray(img)
        
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), anti_aliasing=True)
        img = (img - 0.5) / 0.5
        img_tensor = torch.FloatTensor(img).unsqueeze(0)
        
        target = [self.char_to_int[c] for c in label_text]
        target_tensor = torch.LongTensor(target)
        
        return img_tensor, target_tensor, len(target)

def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets_concat = torch.cat(targets)
    target_lengths = torch.IntTensor(target_lengths)
    return images, targets_concat, target_lengths

def decode_ctc(output, int_to_char):
    output = output.permute(1, 0, 2)
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

def validate(model, dataloader, criterion, int_to_char):
    model.eval()
    losses = []
    correct_words = 0
    total_words = 0
    
    # For advanced metrics
    all_targets = []
    all_preds = []
    char_to_int = {v: k for k, v in int_to_char.items()}
    
    with torch.no_grad():
        for images, targets_concat, target_lengths in dataloader:
            images = images.to(DEVICE)
            log_probs = model(images)
            batch_size = images.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.int32)
            
            loss = criterion(log_probs, targets_concat, input_lengths, target_lengths)
            losses.append(loss.item())
            
            decoded_texts = decode_ctc(log_probs.cpu(), int_to_char)
            
            # Reconstruction of original texts
            pos = 0
            original_texts = []
            for length in target_lengths:
                t = targets_concat[pos:pos+length]
                label_indices = [c.item() for c in t]
                all_targets.extend(label_indices)
                original_texts.append("".join([int_to_char[c.item()] for c in t]))
                pos += length

            for pred, target in zip(decoded_texts, original_texts):
                if pred == target:
                    correct_words += 1
                total_words += 1
                
                # Match characters for p/r/f1 (naive alignment)
                pred_indices = [char_to_int.get(c, 0) for c in pred]
                target_indices = [char_to_int.get(c, 0) for c in target]
                
                if len(pred_indices) < len(target_indices):
                    pred_indices += [0] * (len(target_indices) - len(pred_indices))
                elif len(pred_indices) > len(target_indices):
                    pred_indices = pred_indices[:len(target_indices)]
                
                all_preds.extend(pred_indices)
                
    # Calculate advanced metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
                
    return np.mean(losses), correct_words / total_words, precision, recall, f1

# 3. Early Stopping
class EarlyStopping:
    def __init__(self, patience=15, path='models/kfold_best.pth'):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# 4. Main Training Loop
def main():
    mlflow.set_experiment("Captcha_CRNN_KFold")
    
    print("Downloading dataset...")
    data_path = kagglehub.dataset_download("fournierp/captcha-version-2-images")
    samples_dir = Path(data_path) / "samples"
    all_files = np.array(list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg")))
    
    all_labels = "".join([f.stem for f in all_files])
    vocab = "".join(sorted(list(set(all_labels))))
    char_to_int = {char: i + 1 for i, char in enumerate(vocab)}
    int_to_char = {i: char for char, i in char_to_int.items()}
    num_classes = len(vocab) + 1
    
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_files)):
        print(f"\n🚀 Starting Fold {fold+1}/{K_FOLDS}")
        
        train_files = all_files[train_idx]
        val_files = all_files[val_idx]
        
        train_ds = CaptchaDataset(train_files, char_to_int)
        val_ds = CaptchaDataset(val_files, char_to_int)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        model = CRNN(num_classes).to(DEVICE)
        criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        early_stopping = EarlyStopping(patience=15, path=f'models/crnn_kfold_best_fold{fold+1}.pth')

        with mlflow.start_run(run_name=f"Fold_{fold+1}"):
            mlflow.log_params({"fold": fold+1, "epochs": EPOCHS, "batch_size": BATCH_SIZE})
            
            for epoch in range(EPOCHS):
                model.train()
                epoch_loss = 0
                for images, targets_concat, target_lengths in train_loader:
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
                avg_val_loss, val_word_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion, int_to_char)
                
                mlflow.log_metrics({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_word_accuracy": val_word_acc, "val_precision": val_precision, "val_recall": val_recall, "val_f1_score": val_f1}, step=epoch)
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_word_acc:.2%}")
                    print(f"  P: {val_precision:.2f} | R: {val_recall:.2f} | F1: {val_f1:.2f}")
                
                # Early Stopping
                if (epoch + 1) >= 100:
                    early_stopping(avg_val_loss, model)
                    if early_stopping.early_stop:
                        print(f"  Early stopping triggered at Epoch {epoch+1}")
                        break
            
            fold_results.append(early_stopping.best_loss)
            mlflow.log_metric("final_val_loss", early_stopping.best_loss)

    print(f"\n✅ K-Fold Cross Validation Complete!")
    print(f"Average Best Val Loss across {K_FOLDS} folds: {np.mean(fold_results):.4f}")

if __name__ == "__main__":
    main()
