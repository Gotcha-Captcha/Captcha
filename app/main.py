from fastapi import FastAPI, UploadFile, File, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import os
import shutil
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from .model_logic import predict_captcha, train_model_async

# Setup paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    UPLOAD_DIR.mkdir(exist_ok=True)
    TEMPLATES_DIR.mkdir(exist_ok=True)
    yield
    # Shutdown
    if UPLOAD_DIR.exists():
        for file in UPLOAD_DIR.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
        print("✅ Cleanup complete: Uploaded files deleted.")

app = FastAPI(title="Captcha Recognition Demo", lifespan=lifespan)

# Setup templates and static files
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global state to keep track of training
STATE = {
    "training_in_progress": False,
    "last_accuracy": "0.00%", 
    "last_char_accuracy": "0.00%",
    "progress": 0,
    "status": "Ready",
    "dataset_path": "",
    "model_metadata": {
        "accuracy": "N/A",
        "char_accuracy": "N/A",
        "type": "N/A",
        "loss": "N/A",
        "features": "N/A",
        "preprocessing": "N/A"
    }
}

def load_metadata():
    global STATE
    meta_path = Path(__file__).parent.parent / "models" / "model_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                data = json.load(f)
                STATE["model_metadata"] = data
                STATE["last_accuracy"] = data.get("accuracy", "0.00%")
                STATE["last_char_accuracy"] = data.get("char_accuracy", "0.00%")
                STATE["status"] = "Pre-trained model loaded"
                print(f"✅ Metadata loaded from {meta_path}")
        except Exception as e:
            print(f"❌ Failed to load metadata: {e}")
    else:
        print(f"⚠️ No metadata found at {meta_path}")

# Initial load
load_metadata()

# --- Background Task ---
async def train_task(websockets):
    global STATE
    STATE["training_in_progress"] = True
    STATE["progress"] = 0
    STATE["status"] = "Initializing dataset..."
    
    # We use a global dataset path that would be set during setup
    # For demo purposes, we'll try to find the kaggle dataset path or use a placeholder
    images_dir = STATE.get("dataset_path", "samples") 
    
    async def progress_callback(percentage, status):
        STATE["progress"] = percentage
        STATE["status"] = status
        # Broadcast to all connected websockets
        for ws in websockets:
            try:
                await ws.send_json({"progress": percentage, "status": status})
            except:
                pass

    try:
        accuracy = await train_model_async(images_dir, progress_callback)
        STATE["last_accuracy"] = f"{accuracy:.2%}"
        STATE["status"] = f"Training Finished (Acc: {STATE['last_accuracy']})"
    except Exception as e:
        STATE["status"] = f"Error: {str(e)}"
    finally:
        STATE["training_in_progress"] = False

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "metrics": STATE["model_metadata"],
        "status": STATE["status"]
    })

@app.post("/predict")
async def post_predict(file: UploadFile = File(...)):
    file_path = BASE_DIR / "uploads" / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    prediction = predict_captcha(file_path)
    return JSONResponse({"prediction": prediction, "filename": file.filename})

# Active websockets store
active_websockets = set()

@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        # Send current state immediately
        await websocket.send_json({"progress": STATE["progress"], "status": STATE["status"]})
        while True:
            await asyncio.sleep(1) # Keep connection alive
            await websocket.receive_text() # Wait for client messages (heartbeat/close)
    except WebSocketDisconnect:
        active_websockets.remove(websocket)

@app.post("/train")
async def post_train(background_tasks: BackgroundTasks):
    global STATE
    if STATE["training_in_progress"]:
        return JSONResponse({"error": "Training already in progress"}, status_code=400)
    
    if not STATE["dataset_path"]:
        import glob
        # SEARCH for the dataset directory
        paths = glob.glob("/Users/parkyoungdu/.cache/kagglehub/datasets/fournierp/captcha-version-2-images/versions/*/samples")
        if paths:
            STATE["dataset_path"] = paths[0]
            print(f"Found dataset at: {paths[0]}")
        else:
            # Fallback for the demo environment if needed
            STATE["dataset_path"] = "samples"

    background_tasks.add_task(train_task, active_websockets)
    return JSONResponse({"message": "Training started"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
