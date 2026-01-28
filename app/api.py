from fastapi import APIRouter

from app.schemas import ClassificationRequest, ClassificationResponse
from app.settings import settings
from app.classifier import TextClassifier


router = APIRouter()
classifier = TextClassifier()

@router.post("/classify", response_model=ClassificationResponse)
def classify(req: ClassificationRequest):
    label, score = classifier.classify(req.text)
    return ClassificationResponse(
        label=label,
        confidence=round(score, 4),
        model_version=settings.version
    )