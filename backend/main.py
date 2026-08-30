"""HTTP boundary; inference is isolated from upload/session handling."""
from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4
try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from ml.inference import InferenceService
except ImportError:
    FastAPI = None

if FastAPI:
    app = FastAPI(title="WonderOfU", version="0.1.0")
    sessions: dict[str, dict[str, Path]] = {}
    service = InferenceService()
    @app.post("/verification/session")
    def create_session():
        session_id = str(uuid4()); sessions[session_id] = {}; return {"id": session_id}
    async def _save(session_id: str, upload: UploadFile, kind: str) -> dict:
        if session_id not in sessions: raise HTTPException(404, "Unknown verification session")
        suffix = Path(upload.filename or "upload").suffix
        if not suffix: raise HTTPException(400, "A filename extension is required")
        destination = Path(tempfile.gettempdir()) / f"wonderofu-{session_id}-{kind}{suffix}"
        with destination.open("wb") as stream: shutil.copyfileobj(upload.file, stream)
        sessions[session_id][kind] = destination
        return {"status": "uploaded"}
    @app.post("/verification/{session_id}/face")
    async def upload_face(session_id: str, file: UploadFile = File(...)): return await _save(session_id, file, "face")
    @app.post("/verification/{session_id}/document")
    async def upload_document(session_id: str, file: UploadFile = File(...)): return await _save(session_id, file, "document")
    @app.post("/verification/{session_id}/analyze")
    def analyze(session_id: str):
        if session_id not in sessions: raise HTTPException(404, "Unknown verification session")
        item = sessions[session_id]; return service.analyze(item.get("face"), item.get("document"))
