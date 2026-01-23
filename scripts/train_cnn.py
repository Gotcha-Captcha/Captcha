import os
# Force CPU usage to avoid Metal/GPU initialization hangs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import resize
from skimage.morphology import opening, footprint_rectangle, remove_small_objects
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import kagglehub

import tensorflow as tf
# Ensure TF sees only CPU
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

from tensorflow.keras import layers, models, callbacks

# 1. Preprocessing Logic (Same as updated app/model_logic.py)
def preprocess_image_v3(img):
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        img_gray = rgb2gray(img)
    else:
        img_gray = img
        
    img_resized = resize(img_gray, (50, 200))
    thresh = threshold_otsu(img_resized)
    img_bin = img_resized > thresh
    
    corners = [img_bin[0,0], img_bin[0,-1], img_bin[-1,0], img_bin[-1,-1]]
    if np.mean(corners) < 0.5: 
        img_inv = img_bin.astype(np.uint8)
    else:
        img_inv = (1 - img_bin).astype(np.uint8)
    
    img_cleaned = opening(img_inv, footprint_rectangle((2, 2)))
    img_cleaned = remove_small_objects(img_cleaned.astype(bool), min_size=20).astype(np.uint8)
    
    return img_cleaned

def segment_characters_v2(img_cleaned, num_chars=5):
    projection = np.sum(img_cleaned, axis=0)
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
        
        diff = abs(h - w)
        p1, p2 = diff // 2, diff - (diff // 2)
        if h > w:
            char_img = np.pad(char_img, ((0, 0), (p1, p2)), mode='constant')
        else:
            char_img = np.pad(char_img, ((p1, p2), (0, 0)), mode='constant')
            
        char_img_resized = resize(char_img, (32, 32))
        characters.append(char_img_resized)
    
    while len(characters) < num_chars:
        characters.append(np.zeros((32, 32)))
        
    return characters

# 2. Build Model
def create_cnn_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(32, 32, 1)),
        layers.Conv2D(16, (3, 3), activation='relu'),  # Reduced from 32
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),  # Reduced from 64
        layers.Flatten(),
        layers.Dense(32, activation='relu'),  # Reduced from 64
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("fournierp/captcha-version-2-images")
    images_dir = Path(path) / "samples"
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    
    # Limit to 200 images for faster training
    image_files = image_files[:200]
    
    X = []
    y = []

    print(f"Using {len(image_files)} images for training. Preprocessing...")
    
    for path in tqdm(image_files):
        try:
            img = imread(str(path))
            label_text = path.stem
            
            img_cleaned = preprocess_image_v3(img)
            char_images = segment_characters_v2(img_cleaned, num_chars=5)
            
            if len(char_images) == len(label_text):
                for char_img, char_label in zip(char_images, label_text):
                    X.append(char_img)
                    y.append(char_label)
        except Exception:
            continue

    X = np.array(X)
    X = X.reshape(-1, 32, 32, 1)
    y = np.array(y)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    print(f"Data Shape: {X.shape}")
    print(f"Classes: {le.classes_}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    
    model = create_cnn_model(len(le.classes_))
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    print("Starting Training...")
    model.fit(
        X_train, y_train,
        epochs=10,  # Reduced from 30
        batch_size=16,  # Reduced from 32
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("\nEvaluating...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    # Save Artifacts
    if not os.path.exists('models'):
        os.makedirs('models')
        
    model.save('models/captcha_cnn_model.h5')
    joblib.dump(le, 'models/label_encoder.joblib')
    print("Model saved to models/captcha_cnn_model.h5")

if __name__ == "__main__":
    main()
