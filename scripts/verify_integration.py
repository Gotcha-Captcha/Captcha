import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.model_logic import predict_captcha

def test_integration():
    # Test images from the Kaggle dataset cache
    test_images = [
        "/Users/parkyoungdu/.cache/kagglehub/datasets/fournierp/captcha-version-2-images/versions/2/samples/p5g5m.png",
        "/Users/parkyoungdu/.cache/kagglehub/datasets/fournierp/captcha-version-2-images/versions/2/samples/e72cd.png",
        "/Users/parkyoungdu/.cache/kagglehub/datasets/fournierp/captcha-version-2-images/versions/2/samples/pgmn2.png",
    ]
    
    print("=" * 60)
    print("CNN Integration Verification")
    print("=" * 60)
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"Skipping: {img_path} (not found)")
            continue
            
        expected = Path(img_path).stem
        predicted = predict_captcha(img_path, use_cnn=True)
        
        print(f"Image: {img_path}")
        print(f"  Expected:  {expected}")
        print(f"  Predicted: {predicted}")
        print(f"  Result:    {'✅ PASS' if expected == predicted else '❌ FAIL'}")
        print("-" * 30)

if __name__ == "__main__":
    test_integration()
