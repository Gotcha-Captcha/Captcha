import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import json

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "cnn_best_model.pth"
METADATA_PATH = BASE_DIR / "models" / "model_metadata_v2.json"

class CaptchaV2Service:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CaptchaV2Service, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized: 
            return
        
        self.model = None
        self.classes = [
            "Bicycle", "Bridge", "Bus", "Car", "Chimney", 
            "Crosswalk", "Hydrant", "Motorcycle", "Palm", 
            "Stair", "Traffic Light"
        ]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.load_model()
        self._initialized = True

    def load_model(self):
        try:
            # Load classes from metadata if available
            if METADATA_PATH.exists():
                with open(METADATA_PATH, "r") as f:
                    meta = json.load(f)
                    self.classes = meta.get("classes", self.classes)
                    print(f"✅ [V2 Service] Loaded {len(self.classes)} classes from metadata")

            # Load Model: EfficientNet-B1 with specific classifier structure
            self.model = models.efficientnet_b1(weights=None)
            num_ftrs = self.model.classifier[1].in_features
            
            # Match the exact structure used during training
            self.model.classifier[1] = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_ftrs, len(self.classes))
            )
            
            if MODEL_PATH.exists():
                state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                
                # Handle 'model.' prefix if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        new_state_dict[k.replace("model.", "", 1)] = v
                    else:
                        new_state_dict[k] = v
                
                self.model.load_state_dict(new_state_dict)
                self.model.to(DEVICE)
                self.model.eval()
                print(f"✅ [V2 Service] EfficientNet-B1 Model loaded from {MODEL_PATH}")
            else:
                print(f"⚠️ [V2 Service] Model file not found at {MODEL_PATH}")
                self.model = None
                
        except Exception as e:
            print(f"❌ [V2 Service] Failed to load model: {e}")
            self.model = None

    def predict(self, img_path: Path):
        """
        Predicts the class of the image at img_path.
        Returns the class name (str) or None if prediction fails.
        """
        if not self.model: 
            self.load_model()
            if not self.model: 
                return None
        
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            idx = predicted_idx.item()
            conf_score = confidence.item() * 100
            result_class = self.classes[idx]
            
            return result_class
        except Exception as e:
            print(f"❌ [V2 Service] Prediction error for {img_path}: {e}")
            return None

# Singleton export
captcha_v2_service = CaptchaV2Service()
