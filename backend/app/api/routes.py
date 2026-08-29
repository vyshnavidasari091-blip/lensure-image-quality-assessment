import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.core.config import settings
from app.db.database import get_db
from app.db.models import AnalysisResult
from app.api.schemas import AnalysisResponse, AnalysisListItem
from app.ml.infer import analyze_image, UnreadableImageError

router = APIRouter()


@router.get("/health")
def health():
    from app.ml.infer import _get_model
    model_loaded = _get_model() is not None
    return {"status": "ok", "model_loaded": model_loaded}


@router.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in settings.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{file.content_type}'. "
                   f"Allowed: {sorted(settings.ALLOWED_CONTENT_TYPES)}",
        )

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_MB}MB limit.")

    stored_name = f"{uuid.uuid4().hex}_{file.filename}"
    stored_path = settings.UPLOAD_DIR / stored_name

    try:
        result = analyze_image(raw)
    except UnreadableImageError as e:
        # still persist a record of the failed analysis for auditability
        rec = AnalysisResult(
            filename=file.filename, stored_path="", quality_score=0.0,
            quality_label="INVALID", issues=[], stats={}, model_version="n/a", error=str(e),
        )
        db.add(rec)
        db.commit()
        raise HTTPException(status_code=422, detail=f"Could not analyze image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # persist the file only after a successful decode
    stored_path.write_bytes(raw)

    rec = AnalysisResult(
        filename=file.filename,
        stored_path=str(stored_path),
        quality_score=result["quality_score"],
        quality_label=result["quality_label"],
        issues=result["issues"],
        stats=result["stats"],
        model_version=result["model_version"],
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


@router.get("/api/results", response_model=list[AnalysisListItem])
def list_results(
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    rows = (
        db.query(AnalysisResult)
        .order_by(desc(AnalysisResult.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )
    return rows


@router.get("/api/results/{result_id}", response_model=AnalysisResponse)
def get_result(result_id: int, db: Session = Depends(get_db)):
    rec = db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
    if rec is None:
        raise HTTPException(status_code=404, detail="Result not found.")
    return rec


@router.get("/api/results/{result_id}/image")
def get_result_image(result_id: int, db: Session = Depends(get_db)):
    from fastapi.responses import FileResponse
    rec = db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
    if rec is None or not rec.stored_path or not Path(rec.stored_path).exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(rec.stored_path)
