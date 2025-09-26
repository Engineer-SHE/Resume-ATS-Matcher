from dataclasses import dataclass
from typing import List, Dict, Any
import os
from pathlib import Path

@dataclass
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    UPLOAD_DIR = DATA_DIR / "uploads"
    OUTPUT_DIR = DATA_DIR / "output"

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    SPACY_MODEL = "en_core_web_sm"

    SECTION_WEIGHTS = {
        "skills": 0.35,
        "experience": 0.30,
        "education": 0.15,
        "summary": 0.10,
        "achievements": 0.10
    }

    SKILL_CATEGORIES = {
        "technical": ["python", "java", "sql", "aws", "docker", "kubernetes", "git",
                     "javascript", "react", "nodejs", "mongodb", "postgresql"],
        "soft": ["leadership", "communication", "teamwork", "problem-solving",
                "analytical", "creative", "organized", "detail-oriented"],
        "action_verbs": ["developed", "implemented", "designed", "optimized",
                        "managed", "led", "created", "built", "analyzed", "improved"]
    }

    MIN_MATCH_SCORE = 0.70

    ATS_KEYWORDS_THRESHOLD = 0.60

    MAX_RESUME_LENGTH = 10000

    @classmethod
    def setup_directories(cls):
        for dir_path in [cls.DATA_DIR, cls.UPLOAD_DIR, cls.OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)