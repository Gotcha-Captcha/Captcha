import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from ..ml_models.crnn import CRNN
from ..core.config import BASE_DIR, MODELS_DIR
from .svm_service import predict_captcha_svm

def predict_captcha_cnn(image_path: str):
    # CRNN uses a fixed vocab based on common captcha characters
    vocab = "2345678bcdefgmnpwxy" # Default for this dataset
    int_to_char = {i + 1: char for i, char in enumerate(vocab)}
    num_classes = len(vocab) + 1
    
    model_path = MODELS_DIR / "crnn_v5_best.pth"
    if not model_path.exists():
        # Fallback to final or epoch checkpoints
        model_path = MODELS_DIR / "crnn_v5_final.pth"
        
    if not model_path.exists():
        checkpoints = sorted(list(MODELS_DIR.glob("crnn_v5_epoch_*.pth")))
        if checkpoints:
            model_path = checkpoints[-1]
        else:
            return "Error: CRNN model file not found."
    
    print(f"Loading model from {model_path}")
    model = CRNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Preprocess
    img = imread(str(image_path))
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

def predict_captcha(image_path: str, use_cnn=True):
    """Refined wrapper for prediction service."""
    try:
        if use_cnn:
            return predict_captcha_cnn(image_path)
        return predict_captcha_svm(image_path)
    except Exception as e:
        return f"Error: {str(e)}"
