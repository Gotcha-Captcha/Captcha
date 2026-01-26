import numpy as np
import joblib
from pathlib import Path
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.feature import hog
from skimage.transform import resize
from skimage.morphology import opening, footprint_rectangle, remove_small_objects
from ..core.config import MODELS_DIR

def preprocess_image_v3(img):
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    img_gray = rgb2gray(img)
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

def extract_hog_features(img_binary, pixels_per_cell=(4, 4)):
    return hog(
        img_binary,
        orientations=9,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True
    )

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

def predict_captcha_svm(image_path: str):
    model_path = MODELS_DIR / "captcha_svm_model.pkl"
    encoder_path = MODELS_DIR / "label_encoder.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    
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
        feat_scaled = scaler.transform(feat.reshape(1, -1))
        pred = model.predict(feat_scaled)
        char = label_encoder.inverse_transform(pred)[0]
        predicted_text += char
    return predicted_text
