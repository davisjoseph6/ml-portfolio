import os
import io
import fitz                 # PyMuPDF
import pytesseract
from PIL import Image
import boto3
import json
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the 'static' directory to serve HTML/JS/CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Optional: Serve 'static/index.html' at the root URL
@app.get("/")
def read_root():
    # If you prefer to redirect, you can redirect to "/static/index.html"
    # But let's just load index.html here:
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# ----- Update these to match your actual endpoints in AWS SageMaker -----
CLASSIFICATION_ENDPOINT = "distilbert-tf-endpoint"
SUMMARIZATION_ENDPOINT  = "my-summar-endpoint"
RAG_ENDPOINT            = "rag-endpoint-minimal-v1"

REGION_NAME = "eu-west-2"  # or your region

# Create a single runtime client we can reuse
runtime = boto3.client("sagemaker-runtime", region_name=REGION_NAME)

# -----------------------------------------------------
# POST-PROCESSING: Map Hugging Face "LABEL_XX" -> friendly label
LABEL_MAP = {
    "LABEL_0":  "advertisement",
    "LABEL_1":  "budget",
    "LABEL_2":  "email",
    "LABEL_3":  "form",
    "LABEL_4":  "handwritten",
    "LABEL_5":  "inventory_report",
    "LABEL_6":  "invoice",
    "LABEL_7":  "letter",
    "LABEL_8":  "memo",
    "LABEL_9":  "news_business",
    "LABEL_10": "news_entertainment",
    "LABEL_11": "news_general",
    "LABEL_12": "news_sport",
    "LABEL_13": "presentation",
    "LABEL_14": "purchase_orders",
    "LABEL_15": "questionnaire",
    "LABEL_16": "resume",
    "LABEL_17": "scientific_publication",
    "LABEL_18": "scientific_report",
    "LABEL_19": "shipping_orders",
    "LABEL_20": "specification"
}
# -----------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file (in-memory bytes) using PyMuPDF."""
    text_content = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_content += page.get_text()
    return text_content

def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from an image using Tesseract OCR."""
    image = Image.open(io.BytesIO(file_bytes))
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_txt(file_bytes: bytes) -> str:
    """Directly read text from a .txt file (UTF-8 assumed)."""
    return file_bytes.decode("utf-8", errors="ignore")

def extract_text_from_file_upload(upload_file: UploadFile) -> str:
    """
    Determines filetype by extension and extracts text.
    For simplicity, rely on extension in 'upload_file.filename'.
    """
    filename = upload_file.filename.lower()
    file_bytes = upload_file.file.read()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
        return extract_text_from_image(file_bytes)
    elif filename.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file extension: {filename}")

@app.post("/classify")
async def classify_file(file: UploadFile = File(...)):
    """
    1) Receives a file (PDF/IMG/TXT)
    2) Extracts text locally
    3) Sends the text to the classification endpoint
    4) Returns the predicted label (post-processed to friendly name)
    """
    try:
        text = extract_text_from_file_upload(file)
        if not text.strip():
            return JSONResponse({"error": "No text extracted from file."}, status_code=400)

        # The Hugging Face inference container typically expects {"inputs": "some text"}
        payload = {"inputs": text}

        response = runtime.invoke_endpoint(
            EndpointName=CLASSIFICATION_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read().decode("utf-8"))

        # POST-PROCESS the label to replace "LABEL_XX" with our friendly name
        if isinstance(result, list) and len(result) > 0 and "label" in result[0]:
            raw_label = result[0]["label"]  # e.g. "LABEL_14"
            friendly_label = LABEL_MAP.get(raw_label, raw_label)
            result[0]["label"] = friendly_label

        return JSONResponse({"classification_result": result})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/summarize")
async def summarize_file(file: UploadFile = File(...)):
    """
    1) Receives a file (PDF/IMG/TXT)
    2) Extracts text locally
    3) Sends the text to the summarization endpoint
    4) Returns the summary
    """
    try:
        text = extract_text_from_file_upload(file)
        if not text.strip():
            return JSONResponse({"error": "No text extracted from file."}, status_code=400)

        payload = {"inputs": text}

        response = runtime.invoke_endpoint(
            EndpointName=SUMMARIZATION_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        # Summarization container often returns something like [{"summary_text": "..."}]
        if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
            return JSONResponse({"summary": result[0]["summary_text"]})
        else:
            return JSONResponse({"raw_response": result})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/rag")
async def rag_query(request: Request):
    """
    1) Receives JSON with { "question": "..." }
    2) Passes it to your RAG endpoint with any other needed parameters (e.g., top_k)
    3) Returns the final answer from the RAG endpoint
    """
    try:
        body = await request.json()
        question = body.get("question", "")
        top_k = body.get("top_k", 2)  # or default if not provided

        if not question.strip():
            return JSONResponse({"error": "Empty 'question' provided."}, status_code=400)

        payload = {
            "query": question,
            "top_k": top_k
        }

        response = runtime.invoke_endpoint(
            EndpointName=RAG_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        result = json.loads(response["Body"].read().decode("utf-8"))

        # The structure of the result depends on your RAG container.
        return JSONResponse({"rag_result": result})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

