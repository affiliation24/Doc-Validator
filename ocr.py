import pdfplumber
import pytesseract
from PIL import Image
import io

SUPPORTED_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/tiff",
}

def extract_text_from_pdf(file_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages_text = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

    if pages_text:
        return "\n\n".join(pages_text)

    return extract_text_from_image_bytes(file_bytes)

def extract_text_from_image_bytes(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes))
    text = pytesseract.image_to_string(image, lang="rus+eng")
    return text.strip()

def extract_text_from_file(file_bytes: bytes, content_type: str) -> str:
    if content_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"Неподдерживаемый тип файла: {content_type}. "
            f"Поддерживаются: PDF, JPG, PNG, WEBP, TIFF"
        )

    if content_type == "application/pdf":
        return extract_text_from_pdf(file_bytes)

    return extract_text_from_image_bytes(file_bytes)