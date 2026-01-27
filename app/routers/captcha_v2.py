import random
import base64
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from ..core.config import get_v2_dataset_path

router = APIRouter(prefix="/api/v2")

CATEGORIES = {
    "car": ["sedan", "suv", "coupe", "convertible"],
    "truck": ["pickup", "semi", "lorry"],
    "bus": ["bus", "shuttle"],
    "motorcycle": ["motorcycle", "bike"],
    "traffic light": ["traffic light", "signal"]
}

def get_base64_img(path: Path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

@router.get("/challenge")
async def get_challenge():
    path_str = get_v2_dataset_path()
    if not path_str:
        return JSONResponse({"error": "Dataset path configuration missing"}, status_code=500)
        
    dataset_root = Path(path_str)
    if not dataset_root.exists():
        return JSONResponse({"error": f"Dataset not found at {dataset_root}"}, status_code=500)
    
    # In 'samples_v2/images', subfolders are labels
    subdirs = [d for d in dataset_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    print(f"DEBUG: Found subdirs in {dataset_root}: {[d.name for d in subdirs]}")
    
    # Filter out "Other", "labels", "images" from potential target categories to avoid ambiguous prompts or structural folders
    excluded_names = ["other", "labels", "images"]
    target_subdirs = [d for d in subdirs if d.name.lower() not in excluded_names]
    
    if not target_subdirs:
        # If no identifiable subfolders, use any including Other (backup)
        target_subdirs = subdirs

    if not subdirs:
        # If no subfolders, try reading images directly from root (backup)
        target_files = list(dataset_root.glob("*.jpg")) + list(dataset_root.glob("*.png"))
        if not target_files:
            return JSONResponse({"error": "No categories or images found in dataset"}, status_code=404)
        target_name = "image"
        num_target = 9
        grid_images = [{"image": get_base64_img(f), "is_correct": True, "id": f.name} for f in random.sample(target_files, min(9, len(target_files)))]
    else:
        target_category_dir = random.choice(target_subdirs)
        target_name = target_category_dir.name
        target_files = list(target_category_dir.glob("*.jpg")) + list(target_category_dir.glob("*.png"))
        
        while not target_files and len(subdirs) > 1:
            subdirs.remove(target_category_dir)
            target_category_dir = random.choice(subdirs)
            target_name = target_category_dir.name
            target_files = list(target_category_dir.glob("*.jpg")) + list(target_category_dir.glob("*.png"))

        grid_images = []
        num_target = random.randint(3, 5)
        selected_targets = random.sample(target_files, min(num_target, len(target_files)))
        for f in selected_targets:
            grid_images.append({"image": get_base64_img(f), "is_correct": True, "id": f.name})
            
        # distractor images
        other_dirs = [d for d in subdirs if d != target_category_dir]
        num_distractors = 9 - len(grid_images)
        
        if other_dirs:
            for _ in range(num_distractors):
                d_dir = random.choice(other_dirs)
                d_files = list(d_dir.glob("*.jpg")) + list(d_dir.glob("*.png"))
                if d_files:
                    d_file = random.choice(d_files)
                    grid_images.append({"image": get_base64_img(d_file), "is_correct": False, "id": d_file.name})
    
    # Fill remaining slots if any
    while len(grid_images) < 9 and 'target_files' in locals() and target_files:
        f = random.choice(target_files)
        grid_images.append({"image": get_base64_img(f), "is_correct": True, "id": f.name})
        
    random.shuffle(grid_images)
    
    return {
        "target": target_name,
        "grid": [{"image": g["image"], "id": g["id"]} for g in grid_images],
    }

@router.post("/solve")
async def solve_v2(request: Request):
    """Simulate AI solving the 3x3 grid."""
    data = await request.json()
    grid_ids = data.get("grid_ids", []) # IDs of images currently in the grid
    
    dataset_root = Path(get_v2_dataset_path())
    target_name = data.get("target")
    
    correct_indices = []
    for idx, img_id in enumerate(grid_ids):
        # In this dataset, the image is in the folder named after the category
        img_path = dataset_root / target_name / img_id
        if img_path.exists():
            correct_indices.append(idx)
            
    return {"correct_indices": correct_indices}

@router.post("/verify")
async def verify_v2(request: Request):
    data = await request.json()
    target = data.get("target")
    selected_ids = data.get("selected_ids", [])
    grid_ids = data.get("grid_ids", [])
    
    dataset_root = Path(get_v2_dataset_path())
    
    # Calculate truth
    actual_correct_ids = []
    for img_id in grid_ids:
        if (dataset_root / target / img_id).exists():
            actual_correct_ids.append(img_id)
            
    is_correct = sorted(selected_ids) == sorted(actual_correct_ids)
    
    return {
        "is_correct": is_correct,
        "actual_ids": actual_correct_ids
    }
