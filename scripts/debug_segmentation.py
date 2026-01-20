import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread, imsave

# Import our logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.model_logic import preprocess_image_v3, segment_characters_v2

def debug_segmentation():
    import glob
    paths = glob.glob("/Users/parkyoungdu/.cache/kagglehub/datasets/fournierp/captcha-version-2-images/versions/*/samples")
    if not paths:
        print("Dataset not found")
        return
    
    images_dir = Path(paths[0])
    image_files = list(images_dir.glob("*.png"))[:10] # Take first 10 for debug
    
    output_dir = Path(__file__).parent.parent / "debug_output"
    output_dir.mkdir(exist_ok=True)
    
    for path in image_files:
        print(f"Processing {path.name}...")
        img = imread(str(path))
        img_cleaned = preprocess_image_v3(img)
        
        # Save original and full preprocessed image for context
        imsave(output_dir / f"{path.stem}_original.png", img)
        imsave(output_dir / f"{path.stem}_full_binary.png", ( (1 - img_cleaned) * 255).astype(np.uint8))
        
        char_images = segment_characters_v2(img_cleaned, num_chars=5)
        
        # Save segmented characters
        for i, char_img in enumerate(char_images):
            # Save as white-on-black (as the model sees it) for debugging
            char_save = (char_img * 255).astype(np.uint8)
            imsave(output_dir / f"{path.stem}_char_{i}.png", char_save, check_contrast=False)
            
        print(f"Saved {len(char_images)} characters for {path.name}")

if __name__ == "__main__":
    debug_segmentation()
