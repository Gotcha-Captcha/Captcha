import base64
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from ..services.dataset_service import dataset_cache
from ..services.captcha_v2_service import captcha_v2_service

router = APIRouter(prefix="/api/v2")

def get_base64_img(path: Path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

@router.get("/challenge")
async def get_challenge():
    result = dataset_cache.get_v2_challenge()
    if not result:
        return JSONResponse({"error": "Dataset not found or empty"}, status_code=500)
    
    target_name, grid_items = result
    
    # Base64 encode images
    grid_with_images = []
    for item in grid_items:
        img_path = dataset_cache.v2_root / item["category"] / item["id"]
        grid_with_images.append({
            "image": get_base64_img(img_path),
            "id": item["id"]
        })
        
    return {
        "target": target_name,
        "grid": grid_with_images
    }

# ============================
# Model Integration
# ============================
from ..services.captcha_v2_service import captcha_v2_service

@router.on_event("startup")
async def load_v2_model():
    captcha_v2_service.load_model()

@router.post("/solve")
async def solve_v2(request: Request):
    """Real AI solving the 3x3 grid using EfficientNet."""
    data = await request.json()
    grid_ids = data.get("grid_ids", []) 
    target_name = data.get("target")
    
    correct_indices = []
    
    # Reverse lookup map (ID -> Path)
    id_to_path = {}
    for cat, ids in dataset_cache.v2_data.items():
        for img_id in ids:
            id_to_path[img_id] = dataset_cache.v2_root / cat / img_id
            
    for idx, img_id in enumerate(grid_ids):
        if img_id not in id_to_path:
            continue
            
        img_path = id_to_path[img_id]
        predicted_class = captcha_v2_service.predict(img_path)
        
        # Determine match
        # Note: predicted_class is singular (e.g. 'bus'), target_name is 'bus'.
        # Exact match.
        if predicted_class and predicted_class.lower() == target_name.lower():
            correct_indices.append(idx)
            
    return {"correct_indices": correct_indices}

@router.post("/verify")
async def verify_v2(request: Request):
    data = await request.json()
    target = data.get("target")
    selected_ids = data.get("selected_ids", [])
    grid_ids = data.get("grid_ids", [])
    
    if target not in dataset_cache.v2_data:
        return {"is_correct": False, "actual_ids": []}

    target_files = set(dataset_cache.v2_data[target])
    actual_correct_ids = [img_id for img_id in grid_ids if img_id in target_files]
            
    is_correct = sorted(selected_ids) == sorted(actual_correct_ids)
    
    return {
        "is_correct": is_correct,
        "actual_ids": actual_correct_ids
    }
