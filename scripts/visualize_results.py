import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import joblib

# Import model definition from training script
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from scripts.train_crnn_ctc import CRNN, INT_TO_CHAR, CHAR_TO_INT, VOCAB, IMG_HEIGHT, IMG_WIDTH, DEVICE

def preprocess_image(img_path):
    img = imread(str(img_path))
    if img.ndim == 3:
        if img.shape[2] == 4: img = img[..., :3]
        img = rgb2gray(img)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), anti_aliasing=True)
    img = (img - 0.5) / 0.5
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
    return img_tensor

def visualize_ctc(model, img_path, save_path):
    model.eval()
    img_tensor = preprocess_image(img_path).to(DEVICE)
    
    with torch.no_grad():
        # log_probs: [T, batch, classes]
        log_probs = model(img_tensor)
        probs = torch.exp(log_probs).squeeze(1).cpu().numpy() # [T, classes]
        
    T, C = probs.shape
    
    plt.figure(figsize=(15, 6))
    
    # 1. Show Image
    plt.subplot(2, 1, 1)
    img = imread(img_path)
    plt.imshow(img)
    plt.title(f"Original CAPTCHA: {Path(img_path).stem}")
    plt.axis('off')
    
    # 2. Show Probability Heatmap
    plt.subplot(2, 1, 2)
    # We ignore the blank token (index 0) for better visualization of characters
    char_probs = probs[:, 1:]
    char_labels = [INT_TO_CHAR[i] for i in range(1, C)]
    
    sns.heatmap(char_probs.T, xticklabels=5, yticklabels=char_labels, cmap="YlGnBu")
    plt.title("CTC Character Probabilities over Time (Sequence Steps)")
    plt.xlabel("Sequence Step (T)")
    plt.ylabel("Characters")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")

def main():
    model_path = "models/crnn_v5_final.pth"
    if not os.path.exists(model_path):
        print("Model not found!")
        return
        
    # Standard alphabet used in the 200-epoch run
    vocab = "2345678bcdefgmnpwxy"
    int_to_char = {i + 1: char for i, char in enumerate(vocab)}
    num_classes = len(vocab) + 1
    
    # Update global INT_TO_CHAR for visualize_ctc
    global INT_TO_CHAR
    INT_TO_CHAR = int_to_char
    
    model = CRNN(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # Take a few samples
    import kagglehub
    data_path = kagglehub.dataset_download("fournierp/captcha-version-2-images")
    samples_dir = Path(data_path) / "samples"
    sample_images = list(samples_dir.glob("*.png"))[:3]
    
    os.makedirs("docs/visualizations", exist_ok=True)
    
    for i, img_path in enumerate(sample_images):
        save_path = f"docs/visualizations/ctc_heatmap_{i}.png"
        visualize_ctc(model, img_path, save_path)

if __name__ == "__main__":
    import os
    main()
