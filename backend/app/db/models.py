import datetime as dt
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from app.db.database import Base


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=False)
    quality_score = Column(Float, nullable=False)
    quality_label = Column(String, nullable=False)
    issues = Column(JSON, nullable=False)          # list[{type, severity, confidence}]
    stats = Column(JSON, nullable=False)            # raw feature stats (sharpness, brightness, ...)
    model_version = Column(String, nullable=False, default="v1")
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)
