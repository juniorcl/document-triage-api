from fastapi import FastAPI
from app.api import router
from app.settings import settings


app = FastAPI(
    title=settings.service_name,
    version=settings.version
)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(router)