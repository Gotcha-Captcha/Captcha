import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image

# ============================
# 1. Configuration
# ============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')

# ============================
# 2. Grad-CAM Utils
# ============================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # Hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activation = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        # Forward pass
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        # Generate CAM
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activation[0].cpu().data.numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, class_idx, torch.softmax(output, dim=1).max().item()

# ============================
# 3. Main Logic
# ============================
def load_metadata(path="models/model_metadata_v2.json"):
    if not os.path.exists(path):
        print("Metadata not found. Plotting skipped.")
        return None
    with open(path, "r") as f:
        return json.load(f)

def plot_history(metadata):
    if not metadata or "fold_results" not in metadata:
        return
        
    folds = metadata["fold_results"]
    plt.figure(figsize=(15, 6))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    for f in folds:
        plt.plot(f['history']['train_loss'], alpha=0.3, label=f"Fold {f['fold']} Train")
        plt.plot(f['history']['val_loss'], label=f"Fold {f['fold']} Val")
    plt.title("Loss per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    for f in folds:
        plt.plot(f['history']['train_acc'], alpha=0.3, label=f"Fold {f['fold']} Train")
        plt.plot(f['history']['val_acc'], label=f"Fold {f['fold']} Val")
    plt.title("Accuracy per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    os.makedirs("docs/visualizations", exist_ok=True)
    plt.savefig("docs/visualizations/v2_training_history.png")
    print("Saved training history to docs/visualizations/v2_training_history.png")

def visualize_heatmaps(model, classes, data_root, num_samples=3):
    img_paths = []
    # Collect random images
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.lower().endswith(('.png', '.jpg')):
                img_paths.append(os.path.join(root, f))
                
    if not img_paths:
        return
        
    selected = np.random.choice(img_paths, num_samples, replace=False)
    
    # Target Layer: Last Conv layer of EfficientNet features
    # EfficientNet parts: features (Sequential) -> avgpool -> classifier
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i, img_path in enumerate(selected):
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
        
        # Run Grad-CAM
        cam, pred_idx, conf = grad_cam(img_tensor)
        
        # Prepare visualization
        img_np = np.array(img_pil.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        
        true_label = Path(img_path).parent.name
        pred_label = classes[pred_idx]
        
        # Plot
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(img_np)
        plt.title(f"Original: {true_label}")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(heatmap)
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(overlay)
        plt.title(f"Pred: {pred_label} ({conf:.2f})")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig("docs/visualizations/v2_gradcam.png")
    print("Saved Grad-CAM heatmaps to docs/visualizations/v2_gradcam.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/efficientnet_v2_best.pth")
    # Try to load path from metadata, if not default
    parser.add_argument("--data_root", default=None) 
    args = parser.parse_args()
    
    metadata = load_metadata()
    if metadata:
        plot_history(metadata)
        data_root = args.data_root if args.data_root else metadata.get("dataset_path", "data/samples_v2")
    else:
        data_root = args.data_root or "data/samples_v2"

    if not os.path.exists(args.model_path):
        print("Model not found. Train first.")
        exit()
        
    # Load Model
    classes = load_metadata()["classes"] if metadata else []
    if not classes and os.path.exists("models/v2_classes.txt"):
        with open("models/v2_classes.txt") as f:
            classes = [l.strip() for l in f]
            
    print(f"Loading model with {len(classes)} classes...")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    visualize_heatmaps(model, classes, data_root)
