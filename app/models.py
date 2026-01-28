from enum import Enum
from typing import Optional
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    model_version: str
    metadata: Optional[dict] = None

class DocumentLabel(str, Enum):
    DEFERIMENTO = "deferimento"
    INDEFERIMENTO = "indeferimento"
    PARCIAL = "parcial"
    UNKNOWN = "unknown"

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


