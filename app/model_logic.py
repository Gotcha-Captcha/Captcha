
import os
import joblib
import numpy as np
import time
from pathlib import Path
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.feature import hog
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.morphology import opening, footprint_rectangle, remove_small_objects
import torch
import torch.nn as nn

# ============================
# Core Logic
# ============================

def preprocess_image_v3(img):
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    img_gray = rgb2gray(img)
    img_resized = resize(img_gray, (50, 200))
    
    # Global Otsu is more stable if background is relatively flat
    thresh = threshold_otsu(img_resized)
    img_bin = img_resized > thresh
    
    # Better background detection: Sample the 4 corners
    corners = [img_bin[0,0], img_bin[0,-1], img_bin[-1,0], img_bin[-1,-1]]
    if np.mean(corners) < 0.5: # 0 is black, 1 is white. If corners are dark, background is 0.
        # Background is dark, text is light. We want text=1.
        img_inv = img_bin.astype(np.uint8)
    else:
        # Background is light, text is dark. We want text=1.
        img_inv = (1 - img_bin).astype(np.uint8)
    
    # Clean up random noise points
    img_cleaned = opening(img_inv, footprint_rectangle((2, 2)))
    img_cleaned = remove_small_objects(img_cleaned.astype(bool), min_size=20).astype(np.uint8)
    
    return img_cleaned

def extract_hog_features(img_binary: np.ndarray, pixels_per_cell=(4, 4)) -> np.ndarray:
    # Using dynamic cells provides flexibility for tuning
    features = hog(
        img_binary,
        orientations=9,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True
    )
    return features

def segment_characters_v2(img_cleaned, num_chars=5):
    """
    Uses vertical projection with adjusted threshold and split logic.
    """
    projection = np.sum(img_cleaned, axis=0)
    # Threshold for finding gaps
    threshold = np.max(projection) * 0.1
    is_char = projection > threshold
    
    char_indices = []
    in_char = False
    start = 0
    for i, val in enumerate(is_char):
        if val and not in_char:
            start = i
            in_char = True
        elif not val and in_char:
            width = i - start
            if width >= 2:
                # Optimized ligature splitting: average width is ~30-40
                if width > 50:
                    num_splits = round(width / 32)
                    split_w = width / num_splits
                    for s in range(num_splits):
                        char_indices.append((int(start + s*split_w), int(start + (s+1)*split_w)))
                else:
                    char_indices.append((start, i))
            in_char = False
    if in_char:
        char_indices.append((start, len(is_char)))

    characters = []
    for (start, end) in char_indices[:num_chars]:
        char_img = img_cleaned[:, start:end]
        h, w = char_img.shape
        if h == 0 or w == 0: continue
        
        # Add padding to make it square to maintain aspect ratio
        diff = abs(h - w)
        p1, p2 = diff // 2, diff - (diff // 2)
        if h > w:
            char_img = np.pad(char_img, ((0, 0), (p1, p2)), mode='constant')
        else:
            char_img = np.pad(char_img, ((p1, p2), (0, 0)), mode='constant')
            
        char_img_resized = resize(char_img, (32, 32))
        characters.append(char_img_resized)
    
    # Fill remaining with empty if needed
    while len(characters) < num_chars:
        characters.append(np.zeros((32, 32)))
        
    return characters

# ============================
# CNN Model Definition
# ============================

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
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
        self.rnn = nn.Sequential(
            BidirectionalLSTM(1024, 256, 256),
            BidirectionalLSTM(256, 256, num_classes)
        )
        
    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w) 
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return torch.nn.functional.log_softmax(output, dim=2)

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.embedding = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output

# ============================
# Prediction Logic
# ============================

def predict_captcha(image_path: str, use_cnn=True):
    try:
        if use_cnn:
            return predict_captcha_cnn(image_path)
        
        # Fallback to SVM
        return predict_captcha_svm(image_path)
    except Exception as e:
        return f"Error: {str(e)}"

def predict_captcha_cnn(image_path: str):
    # Load model and config
    model_path = Path(__file__).parent.parent / "models" / "crnn_v5_final.pth"
    # CRNN uses a fixed vocab
    vocab = "2345678bcdefgmnpwxy"
    int_to_char = {i + 1: char for i, char in enumerate(vocab)}
    num_classes = len(vocab) + 1
    
    if not model_path.exists():
        return "Error: CRNN model file not found."
    
    model = CRNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Preprocess
    img = imread(image_path)
    if img.ndim == 3:
        if img.shape[2] == 4: img = img[..., :3]
        img = rgb2gray(img)
    img = resize(img, (50, 200), anti_aliasing=True)
    img = (img - 0.5) / 0.5
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        log_probs = model(img_tensor)
        # Decode CTC (Greedy)
        log_probs = log_probs.permute(1, 0, 2)
        _, max_indices = torch.max(log_probs, 2)
        row = max_indices[0]
        
        predicted_text = ""
        prev = 0
        for char_idx in row:
            char_idx = char_idx.item()
            if char_idx != 0 and char_idx != prev:
                predicted_text += int_to_char.get(char_idx, "")
            prev = char_idx
            
    return predicted_text

def predict_captcha_svm(image_path: str):
    # Load models
    model_path = Path(__file__).parent.parent / "models" / "captcha_svm_model.pkl"
    encoder_path = Path(__file__).parent.parent / "models" / "label_encoder.pkl"
    scaler_path = Path(__file__).parent.parent / "models" / "scaler.pkl"
    
    if not model_path.exists():
        return "Error: SVM model files not found."
    
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    
    img = imread(image_path)
    img_cleaned = preprocess_image_v3(img)
    char_images = segment_characters_v2(img_cleaned, num_chars=5)
    
    predicted_text = ""
    for char_img in char_images:
        feat = extract_hog_features(char_img)
        # Apply scaling
        feat_scaled = scaler.transform(feat.reshape(1, -1))
        pred = model.predict(feat_scaled)
        char = label_encoder.inverse_transform(pred)[0]
        predicted_text += char
        
    return predicted_text

# ============================
# Training Logic (Async friendly)
# ============================

async def train_model_async(images_dir, progress_callback):
    """
    Simulates training with progress updates.
    In a real scenario, this would loop through image_files.
    """
    image_files = list(Path(images_dir).glob("*.png")) + list(Path(images_dir).glob("*.jpg"))
    total = len(image_files)
    
    if total == 0:
        await progress_callback(0, "No images found.")
        return 0.0

    X_char_features = []
    y_char_labels = []

    # Simulation of feature extraction
    for i, path in enumerate(image_files):
        # Report progress every 5%
        if i % (max(1, total // 20)) == 0:
            percentage = int((i / total) * 100)
            await progress_callback(percentage, f"Extracting features: {i}/{total}")

        try:
            img = imread(str(path))
            img_cleaned = preprocess_image_v3(img)
            char_images = segment_characters_v2(img_cleaned, num_chars=5)
            label_text = path.stem
            
            if len(char_images) == len(label_text):
                for char_img, char_label in zip(char_images, label_text):
                    feat = extract_hog_features(char_img)
                    X_char_features.append(feat)
                    y_char_labels.append(char_label)
        except:
            continue
            
    await progress_callback(90, "Training model...")
    
    X_char_features = np.array(X_char_features)
    y_char_labels = np.array(y_char_labels)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_char_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_char_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # RBF Kernel SVM for better non-linear separation
    model = SVC(kernel='rbf', C=2.481, gamma='scale', class_weight='balanced', random_state=42) # C=2.481 gamma='scale'
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save results
    model_path = Path(__file__).parent.parent / "models" / "captcha_svm_model.pkl"
    encoder_path = Path(__file__).parent.parent / "models" / "label_encoder.pkl"
    scaler_path = Path(__file__).parent.parent / "models" / "scaler.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(scaler, scaler_path)
    
    await progress_callback(100, f"Training complete! Accuracy: {accuracy:.2%}")
    return accuracy
