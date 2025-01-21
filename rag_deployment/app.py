# app.py
import os
import json
import boto3
import torch
import faiss
import numpy as np

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# --------------------------------------------------
# Optional: Helper to download from S3 at container startup
# --------------------------------------------------
def download_file_from_s3(s3_uri, local_path):
    s3 = boto3.client('s3')
    assert s3_uri.startswith("s3://"), f"S3 URI must start with s3://, got {s3_uri}"
    no_prefix = s3_uri[5:]
    bucket, key = no_prefix.split("/", 1)
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded {s3_uri} to {local_path}")

print("Starting up container...")

FAISS_BIN_PATH  = "/opt/ml/model/faiss_index.bin"
METADATA_PATH   = "/opt/ml/model/index_metadata.json"

# See if user wants to download from S3
FAISS_INDEX_S3  = os.environ.get("FAISS_INDEX_S3")
METADATA_S3     = os.environ.get("METADATA_S3")

# If environment variables are set, download the FAISS index + metadata
if FAISS_INDEX_S3:
    download_file_from_s3(FAISS_INDEX_S3, FAISS_BIN_PATH)

if METADATA_S3:
    download_file_from_s3(METADATA_S3, METADATA_PATH)

# Now load the FAISS index and metadata from local disk
if not os.path.isfile(FAISS_BIN_PATH):
    raise RuntimeError(f"FAISS index file not found at {FAISS_BIN_PATH}. "
                       f"Either copy it in or set FAISS_INDEX_S3 to download.")

if not os.path.isfile(METADATA_PATH):
    raise RuntimeError(f"Metadata file not found at {METADATA_PATH}. "
                       f"Either copy it in or set METADATA_S3 to download.")

index = faiss.read_index(FAISS_BIN_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load embedding model
EMBED_MODEL = os.environ.get("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
embed_model = SentenceTransformer(EMBED_MODEL)

# Load generation model
GEN_MODEL = os.environ.get("GEN_MODEL_NAME", "my_summarization_model")
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
gen_model.eval()
if torch.cuda.is_available():
    gen_model.to("cuda")

print("Container initialization done. Model is loaded.")

# --------------------------------------------------
# Health check endpoint
# --------------------------------------------------
@app.route("/ping", methods=["GET"])
def ping():
    # SageMaker calls GET /ping for health checks
    return "OK", 200

# --------------------------------------------------
# Inference endpoint
# --------------------------------------------------
@app.route("/invocations", methods=["POST"])
def invocations():
    # The incoming data is JSON
    input_data = request.get_json(force=True)
    query_str = input_data.get("query", "")
    top_k     = input_data.get("top_k", 3)

    # 1) Embed the query and do Faiss search
    q_emb = embed_model.encode([query_str], show_progress_bar=False).astype("float32")
    D, I = index.search(q_emb, top_k)
    docs = [metadata[doc_id] for doc_id in I[0]]

    # 2) Build prompt from retrieved docs + generate answer
    context = "\n\n".join(doc["summary"] for doc in docs)
    prompt  = f"User Query:\n{query_str}\n\nRelevant Summaries:\n{context}\n\nAnswer:"
    inputs  = gen_tokenizer([prompt], return_tensors="pt", max_length=1024, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    if hasattr(gen_model.config, "forced_bos_token_id"):
        gen_model.config.forced_bos_token_id = None

    output_ids = gen_model.generate(
        inputs["input_ids"],
        num_beams=1,
        min_length=10,
        max_length=50,
        no_repeat_ngram_size=2,
        early_stopping=False
    )
    answer = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    # Local testing command:
    #   python app.py
    # which runs Flask's built-in dev server on port 8080
    app.run(host="0.0.0.0", port=8080, debug=True)

