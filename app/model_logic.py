
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
from skimage.morphology import opening, footprint_rectangle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ============================
# Core Logic
# ============================

def preprocess_image_v2(img):
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    img_gray = rgb2gray(img)
    img_resized = resize(img_gray, (50, 200))
    thresh = threshold_otsu(img_resized)
    img_bin = img_resized > thresh
    if np.mean(img_bin[0, :]) < 0.5:
        img_bin = ~img_bin
    
    # Text is currently 0 (black), Background is 1 (white)
    binary_inv = 1 - img_bin
    
    # 1. Specialized Noise Line Removal using Morphological Kernels
    # Remove horizontal lines
    hor_kernel = footprint_rectangle((1, 5))
    hor_noise = opening(binary_inv, hor_kernel)
    
    # Remove vertical lines
    ver_kernel = footprint_rectangle((5, 1))
    ver_noise = opening(binary_inv, ver_kernel)
    
    # Subtract noise but keep characters
    # (Simplified: just generic opening with a small square is often safer for characters)
    img_cleaned = opening(binary_inv, footprint_rectangle((2, 2))) 
    
    return 1 - img_cleaned

def extract_hog_features(img_binary: np.ndarray) -> np.ndarray:
    features = hog(
        img_binary,
        orientations=9,
        pixels_per_cell=(4, 4), # Increased granularity from (8,8) to (4,4)
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True
    )
    return features

def segment_characters(img_bin, num_chars=5):
    h, w = img_bin.shape
    char_width = w // num_chars
    characters = []
    for i in range(num_chars):
        start = i * char_width
        end = (i + 1) * char_width
        char_img = img_bin[:, start:end]
        char_img_resized = resize(char_img, (32, 32))
        characters.append(char_img_resized)
    return characters

# ============================
# Prediction Logic
# ============================

def predict_captcha(image_path: str):
    try:
        # Load models
        model_path = Path(__file__).parent.parent / "models" / "captcha_svm_model.pkl"
        encoder_path = Path(__file__).parent.parent / "models" / "label_encoder.pkl"
        scaler_path = Path(__file__).parent.parent / "models" / "scaler.pkl"
        
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
        
        img = imread(image_path)
        img_bin = preprocess_image_v2(img)
        char_images = segment_characters(img_bin, num_chars=5)
        
        predicted_text = ""
        for char_img in char_images:
            feat = extract_hog_features(char_img)
            # Apply scaling
            feat_scaled = scaler.transform(feat.reshape(1, -1))
            pred = model.predict(feat_scaled)
            char = label_encoder.inverse_transform(pred)[0]
            predicted_text += char
            
        return predicted_text
    except Exception as e:
        return f"Error: {str(e)}"

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
            img_bin = preprocess_image_v2(img)
            char_images = segment_characters(img_bin, num_chars=5)
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
    model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
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
