import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import json

class RecaptchaPredictor:
    def __init__(self, model_path, device=None):
        """
        reCAPTCHA 모델 추론을 위한 래퍼 클래스.
        모델 로드와 전처리를 자동으로 수행합니다.
        
        Args:
            model_path (str): 학습된 .pth 모델 파일 경로
            device (str): 'cuda', 'mps', 'cpu' 중 선택 (None이면 자동 선택)
        """
        self.classes = [
            "Bicycle", "Bridge", "Bus", "Car", "Chimney", 
            "Crosswalk", "Hydrant", "Motorcycle", "Palm", 
            "Stair", "Traffic Light"
        ]
        
        # 파일 시스템의 폴더명과 모델 클래스명 간의 매핑 정의
        self.folder_to_class = {
            "TLight": "Traffic Light",
            "Traffic Light": "Traffic Light",
            # 나머지는 이름이 동일하다고 가정
        }
        
        # 장치 자동 설정
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        # 1. 모델 아키텍처 정의 (EfficientNet-B1)
        self.model = models.efficientnet_b1(weights=None)
        
        # 마지막 분류 레이어 수정 (클래스 11개)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, len(self.classes))
        )
        
        # 2. 가중치 로드
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            
            # 'model.' 접두어 제거 (학습 시 RecaptchaCNN 래퍼 사용으로 인한 키 불일치 해결)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k.replace("model.", "", 1)] = v
                else:
                    new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise e
            
        self.model.to(self.device)
        self.model.eval() # 평가 모드 필수!

        # 3. 전처리 파이프라인 (학습 설정과 동일하게 고정)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # config.yaml의 resize 값 (EfficientNet-B1)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_source):
        """
        이미지 경로 또는 PIL 이미지를 받아 예측 결과를 반환합니다.
        
        Args:
            image_source (str or PIL.Image): 이미지 경로 또는 객체
            
        Returns:
            dict: 예측 클래스, 신뢰도(Confidence), 클래스 인덱스
        """
        # 이미지 로드
        if isinstance(image_source, str) or isinstance(image_source, Path):
            image = Image.open(image_source).convert('RGB')
        else:
            image = image_source.convert('RGB')
            
        # 전처리
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 추론
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        idx = predicted_idx.item()
        conf_score = confidence.item() * 100
        result_class = self.classes[idx]
        
        return {
            "class": result_class,
            "confidence": f"{conf_score:.2f}%",
            "index": idx
        }

# --- 사용 예시 ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on a single image')
    parser.add_argument('--model', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    args = parser.parse_args()
    
    # 예측기 생성
    predictor = RecaptchaPredictor(args.model)
    
    # 예측 실행
    try:
        result = predictor.predict(args.image)
        print("\n Prediction Result:")
        print(f"------------------------")
        print(f" Class: {result['class']}")
        print(f" Confidence: {result['confidence']}")
        print(f"------------------------")
    except Exception as e:
        print(f"Error during inference: {e}")
