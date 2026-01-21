from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
from typing import List
import os
import zipfile
from video_processor import VideoProcessor
import json

app = FastAPI(title="ClipGenius AI", description="Multimodal AI for Viral Reel Generation")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
STATIC_DIR = Path("static")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store processing status
processing_status = {}


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/app", response_class=HTMLResponse)
async def app_page():
    with open("static/app.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        upload_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{upload_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse({
            "success": True,
            "upload_id": upload_id,
            "filename": file.filename,
            "message": "Video uploaded successfully"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process/{upload_id}")
async def process_video(
    upload_id: str,
    background_tasks: BackgroundTasks,
    num_clips: int = 3,
    clip_duration: int = 60,
    min_gap: int = 60,
    use_ai: bool = True
):
    try:
        upload_files = list(UPLOAD_DIR.glob(f"{upload_id}_*"))
        if not upload_files:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        video_path = str(upload_files[0])
        
        # Initialize processing status
        processing_status[upload_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting video processing..."
        }
        
        # Process video
        processor = VideoProcessor(video_path)
        
        # Extract audio
        processing_status[upload_id] = {
            "status": "processing",
            "progress": 20,
            "message": "Extracting audio from video..."
        }
        processor.extract_audio()
        
        # Detect peaks with multimodal AI
        processing_status[upload_id] = {
            "status": "processing",
            "progress": 40,
            "message": "Analyzing with Whisper + Sentiment AI (Multimodal)..."
        }
        result = processor.detect_emotional_peaks(
            num_peaks=num_clips,
            min_gap=min_gap,
            use_ai=use_ai
        )
        peaks, scores, keywords = result
        
        # Generate clips
        processing_status[upload_id] = {
            "status": "processing",
            "progress": 60,
            "message": "Creating viral clips..."
        }
        clips_info = processor.extract_clips(
            peaks=(peaks, scores),
            clip_duration=clip_duration,
            output_dir=str(OUTPUT_DIR)
        )
        
        # Add keywords to clip info
        for clip_info, keyword in zip(clips_info, keywords):
            clip_info['keyword'] = keyword
        
        # Create ZIP file
        zip_path = OUTPUT_DIR / f"{upload_id}_all_reels.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for clip_info in clips_info:
                clip_path = Path(clip_info['path'])
                zipf.write(clip_path, clip_path.name)
                # Add thumbnail
                thumb_path = Path(clip_info['thumbnail'])
                if thumb_path.exists():
                    zipf.write(thumb_path, thumb_path.name)
            # Add metadata
            metadata_path = OUTPUT_DIR / "clips_metadata.json"
            if metadata_path.exists():
                zipf.write(metadata_path, "clips_metadata.json")
        
        # Cleanup
        processor.cleanup()
        
        # Update status
        processing_status[upload_id] = {
            "status": "completed",
            "progress": 100,
            "message": f"Generated {len(clips_info)} viral reels!",
            "clips": clips_info,
            "zip_file": f"/output/{upload_id}_all_reels.zip"
        }
        
        return JSONResponse({
            "success": True,
            "upload_id": upload_id,
            "num_clips": len(clips_info),
            "clips": clips_info,
            "zip_download": f"/output/{upload_id}_all_reels.zip"
        })
    
    except Exception as e:
        processing_status[upload_id] = {
            "status": "error",
            "progress": 0,
            "message": str(e)
        }
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status/{upload_id}")
async def get_status(upload_id: str):
    if upload_id not in processing_status:
        raise HTTPException(status_code=404, detail="Upload ID not found")
    
    return JSONResponse(processing_status[upload_id])


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.delete("/api/cleanup/{upload_id}")
async def cleanup(upload_id: str):
    try:
        for file in UPLOAD_DIR.glob(f"{upload_id}_*"):
            file.unlink()
        
        # Remove from processing status
        if upload_id in processing_status:
            del processing_status[upload_id]
        
        return JSONResponse({"success": True, "message": "Cleanup completed"})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
