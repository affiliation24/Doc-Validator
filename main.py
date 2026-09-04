import uvicorn

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from extractor import extract
from validator import load_rules, validate
from ocr import extract_text_from_file
from history.history import add_record, get_history, get_stats   



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


def run_pipeline(text: str, source: str = "text") -> ProcessResponse:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Текст документа не может быть пустым")
    try:
        extracted = extract(text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Ошибка извлечения данных: {e}")
    try:
        validation = validate(extracted)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Ошибка валидации: {e}")

    add_record(extracted, validation, source=source)

    return ProcessResponse(
        status="APPROVED" if validation.get("approved") else "REJECTED",
        extracted=extracted,
        validation=validation
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "doc-validator"}


@app.post("/process", response_model=ProcessResponse)
def process(request: TextRequest):
    return run_pipeline(request.text, source="text")


@app.post("/process-file", response_model=ProcessResponse)
async def process_file(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {e}")

    try:
        text = extract_text_from_file(file_bytes, file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))

    return run_pipeline(text, source=file.filename)    


@app.get("/history")                           
def history():
    return get_history()


@app.get("/stats")                             
def stats():
    return get_stats()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)