from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import base64
import random
from ..core.config import UPLOAD_DIR, get_dataset_path
from ..services.captcha_service import predict_captcha

router = APIRouter()

@router.post("/predict")
async def post_predict(file: UploadFile = File(...)):
    UPLOAD_DIR.mkdir(exist_ok=True)
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    prediction = predict_captcha(file_path)
    return JSONResponse({"prediction": prediction, "filename": file.filename})

@router.get("/api/challenge")
async def get_challenge():
    """Fetch a random CAPTCHA challenge from the dataset."""
    dataset_path = get_dataset_path()
    if not dataset_path:
        return JSONResponse({"error": "Dataset not found"}, status_code=500)
            
    samples_dir = Path(dataset_path)
    all_files = list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg"))
    
    if not all_files:
        return JSONResponse({"error": "No CAPTCHAs found"}, status_code=404)
        
    random_file = random.choice(all_files)
    
    with open(random_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    return {
        "image": f"data:image/png;base64,{encoded_string}",
        "challenge_id": random_file.name,
        "true_label": random_file.stem
    }

@router.post("/api/solve")
async def solve_challenge(request: Request):
    """Solve a given challenge using the AI model."""
    data = await request.json()
    challenge_id = data.get("challenge_id")
    
    if not challenge_id:
        return JSONResponse({"error": "Missing challenge_id"}, status_code=400)
        
    dataset_path = get_dataset_path()
    file_path = Path(dataset_path) / challenge_id
    if not file_path.exists():
        return JSONResponse({"error": "Challenge image not found"}, status_code=404)
        
    prediction = predict_captcha(file_path)
    
    return {
        "prediction": prediction,
        "is_correct": prediction == file_path.stem
    }

@router.post("/api/verify")
async def verify_challenge(request: Request):
    """Verify the user's input against the actual ground truth."""
    data = await request.json()
    challenge_id = data.get("challenge_id")
    user_input = data.get("user_input", "")
    
    if not challenge_id:
        return JSONResponse({"error": "Missing challenge_id"}, status_code=400)
        
    dataset_path = get_dataset_path()
    file_path = Path(dataset_path) / challenge_id
    if not file_path.exists():
        return JSONResponse({"error": "Challenge image not found"}, status_code=404)
        
    true_label = file_path.stem
    is_correct = user_input.strip().lower() == true_label.strip().lower()
    
    return {
        "is_correct": is_correct,
        "true_label": true_label # Returning this for feedback in demo
    }
