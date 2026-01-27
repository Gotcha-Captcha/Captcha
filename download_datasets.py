import kagglehub
import shutil
from pathlib import Path
import os

def download_all():
    print("📦 Downloading datasets for Docker build...")
    
    # Define destination
    data_dir = Path("/app/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. v1 Dataset
    print("Downloading v1: captcha-version-2-images...")
    v1_src = kagglehub.dataset_download("fournierp/captcha-version-2-images")
    v1_dest = data_dir / "samples_v1"
    if v1_dest.exists():
        shutil.rmtree(v1_dest)
    shutil.copytree(Path(v1_src) / "samples", v1_dest)
    print(f"✅ v1 Dataset copied to {v1_dest}")
    
    # 2. v2 Dataset
    print("Downloading v2: test-dataset...")
    v2_src = kagglehub.dataset_download("mikhailma/test-dataset")
    v2_dest = data_dir / "samples_v2" / "images"
    if v2_dest.exists():
        shutil.rmtree(v2_dest)
    shutil.copytree(v2_src, v2_dest)
    print(f"✅ v2 Dataset copied to {v2_dest}")

if __name__ == "__main__":
    download_all()
