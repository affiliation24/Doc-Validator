from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from extractor import extract
from validator import load_rules, validate
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_rules()
    yield

app = FastAPI(
    title="Doc Validator",
    description="RAG сервис проверки документов по регламентам",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

class ProcessResponse(BaseModel):
    status: str
    extracted: dict
    validation: dict

@app.get("/health")
def health():
    return {"status": "ok", "service": "doc-validator"}

@app.post("/process", response_model=ProcessResponse)
def process(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст документа не может быть пустым")
    try:
        extracted = extract(request.text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Ошибка извлечения данных: {e}")
    try:
        validation = validate(extracted)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Ошибка валидации: {e}")

    return ProcessResponse(
        status="APPROVED" if validation.get("approved") else "REJECTED",
        extracted=extracted,
        validation=validation
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
