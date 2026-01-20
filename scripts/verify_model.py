import sys
import os
import asyncio
import glob

# Add parent directory to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.model_logic import train_model_async

async def main():
    # SEARCH for the dataset directory
    paths = glob.glob("/Users/parkyoungdu/.cache/kagglehub/datasets/fournierp/captcha-version-2-images/versions/*/samples")
    if paths:
        images_dir = paths[0]
        print(f"Found dataset at: {images_dir}")
    else:
        # Fallback
        images_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
        print(f"Dataset not found in Kaggle cache, checking project: {images_dir}")

    async def dummy_callback(progress, status):
        print(f"[{progress}%] {status}")

    print("Starting verification training run...")
    try:
        accuracy = await train_model_async(images_dir, dummy_callback)
        print(f"\n✅ Final Verified Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
