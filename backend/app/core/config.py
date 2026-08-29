import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/

class Settings:
    APP_NAME: str = "AI Image Quality & Defect Detection"
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'iqa.db'}"
    )
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")))
    MODEL_PATH: Path = Path(
        os.getenv("MODEL_PATH", str(BASE_DIR / "app" / "ml" / "weights" / "iqa_head.pt"))
    )
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "15"))
    ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

    def ensure_dirs(self):
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
