from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import shutil
import uvicorn

from .core.config import STATE, UPLOAD_DIR, TEMPLATES_DIR
from .services.metadata_service import load_metadata
from .routers import captcha, captcha_v2

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load metadata and ensure directories exist
    load_metadata()
    UPLOAD_DIR.mkdir(exist_ok=True)
    yield
    # Shutdown: Cleanup upload directory
    if UPLOAD_DIR.exists():
        for file in UPLOAD_DIR.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
        print("✅ Cleanup complete: Uploaded files deleted.")

app = FastAPI(
    title="Gotcha! Captcha Recognition",
    description="Refactored FastAPI backend for CRNN+CTC Captcha solver.",
    lifespan=lifespan
)

# Setup templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Include routers
app.include_router(captcha.router)
app.include_router(captcha_v2.router)

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "metrics": STATE["model_metadata"],
        "status": STATE["status"]
    })

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
