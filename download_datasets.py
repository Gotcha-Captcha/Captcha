import kagglehub
import shutil
from pathlib import Path
import os
import time

def download_with_retry(dataset_handle, max_retries=3, delay=5):
    for i in range(max_retries):
        try:
            return kagglehub.dataset_download(dataset_handle)
        except Exception as e:
            if i < max_retries - 1:
                print(f"⚠️ Error downloading {dataset_handle}: {e}. Retrying in {delay}s... ({i+1}/{max_retries})")
                time.sleep(delay)
            else:
                raise e

def download_all():
    print("📦 Preparing datasets for Docker build...")
    
    # Define destination
    data_dir = Path("/app/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. v1 Dataset
    v1_dest = data_dir / "samples_v1"
    if v1_dest.exists() and any(v1_dest.iterdir()):
        print(f"⏭️ v1 Dataset already exists at {v1_dest}, skipping download.")
    else:
        print("Downloading v1: captcha-version-2-images...")
        v1_src = download_with_retry("fournierp/captcha-version-2-images")
        if v1_dest.exists():
            shutil.rmtree(v1_dest)
        shutil.copytree(Path(v1_src) / "samples", v1_dest)
        print(f"✅ v1 Dataset copied to {v1_dest}")
    
    # 2. v2 Dataset
    v2_dest = data_dir / "samples_v2" / "images"
    # Check if images directory has subdirectories (categories)
    if v2_dest.exists() and any(d.is_dir() for d in v2_dest.iterdir()):
         print(f"⏭️ v2 Dataset already exists at {v2_dest}, skipping download.")
    else:
        print("Downloading v2: test-dataset...")
        v2_src = download_with_retry("mikhailma/test-dataset")
        if v2_dest.exists():
            shutil.rmtree(v2_dest)
        shutil.copytree(v2_src, v2_dest)
        print(f"✅ v2 Dataset copied to {v2_dest}")

if __name__ == "__main__":
    download_all()
