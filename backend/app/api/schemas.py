import datetime as dt
from typing import Any
from pydantic import BaseModel


class Issue(BaseModel):
    type: str
    severity: str
    confidence: float


class AnalysisResponse(BaseModel):
    id: int
    filename: str
    quality_score: float
    quality_label: str
    issues: list[Issue]
    stats: dict[str, Any]
    model_version: str
    created_at: dt.datetime

    class Config:
        from_attributes = True


class AnalysisListItem(BaseModel):
    id: int
    filename: str
    quality_score: float
    quality_label: str
    created_at: dt.datetime

    class Config:
        from_attributes = True
