# app.py
import os
import json
import boto3
import tarfile
import torch
import faiss
import numpy as np

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

################################################################################
# 1. Download from S3 if needed
################################################################################
def download_file_from_s3(s3_uri, local_path):
    """
    Download a single file from an S3 URI (s3://bucket/key) to the local path.
    """
    s3 = boto3.client('s3')
    assert s3_uri.startswith("s3://"), f"S3 URI must start with s3://, got {s3_uri}"
    no_prefix = s3_uri[5:]
    bucket, key = no_prefix.split("/", 1)
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded {s3_uri} to {local_path}")

print("=== Starting up container ===")

# Environment variables with S3 URIs
FAISS_INDEX_S3 = os.environ.get("FAISS_INDEX_S3")       # e.g. s3://aym-client-data-in/rag/faiss_index.bin
METADATA_S3    = os.environ.get("METADATA_S3")          # e.g. s3://aym-client-data-in/rag/index_metadata.json
SUMM_MODEL_S3  = os.environ.get("SUMM_MODEL_S3")        # e.g. s3://aym-client-data-in/my_summarization_model.tar.gz

# Where to store them locally
FAISS_LOCAL_PATH   = "/opt/ml/model/faiss_index.bin"
METADATA_LOCAL_PATH= "/opt/ml/model/index_metadata.json"
SUMM_TAR_PATH      = "/opt/ml/model/my_summarization_model.tar.gz"
SUMM_EXTRACT_DIR   = "/opt/ml/model/summarization_model"

# Download Faiss index + metadata if env vars are set
if FAISS_INDEX_S3:
    download_file_from_s3(FAISS_INDEX_S3, FAISS_LOCAL_PATH)

if METADATA_S3:
    download_file_from_s3(METADATA_S3, METADATA_LOCAL_PATH)

if SUMM_MODEL_S3:
    download_file_from_s3(SUMM_MODEL_S3, SUMM_TAR_PATH)
    print(f"Extracting {SUMM_TAR_PATH} into {SUMM_EXTRACT_DIR} ...")
    os.makedirs(SUMM_EXTRACT_DIR, exist_ok=True)
    with tarfile.open(SUMM_TAR_PATH, "r:gz") as tar:
        tar.extractall(path=SUMM_EXTRACT_DIR)
    print("Extraction complete.")

# Verify files exist locally
if not os.path.isfile(FAISS_LOCAL_PATH):
    raise RuntimeError(f"Faiss index not found at {FAISS_LOCAL_PATH}.")

if not os.path.isfile(METADATA_LOCAL_PATH):
    raise RuntimeError(f"Metadata not found at {METADATA_LOCAL_PATH}.")

# 2. Load the FAISS index & metadata
index = faiss.read_index(FAISS_LOCAL_PATH)
with open(METADATA_LOCAL_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# 3. Load embedding model (Sentence Transformers)
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# 4. Load Summarization (Generation) model
#    We'll load from the extracted directory if present
GEN_MODEL_NAME = os.environ.get("GEN_MODEL_NAME", SUMM_EXTRACT_DIR)
print(f"Loading generation model from: {GEN_MODEL_NAME}")
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model     = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
gen_model.eval()
if torch.cuda.is_available():
    gen_model.to("cuda")

print("=== Container initialization done. Model loaded successfully. ===")

################################################################################
# Health check
################################################################################
@app.route("/ping", methods=["GET"])
def ping():
    """
    SageMaker calls GET /ping to check if the container is alive.
    Return 200 => healthy.
    """
    return "OK", 200

################################################################################
# Inference
################################################################################
@app.route("/invocations", methods=["POST"])
def invocations():
    """
    SageMaker calls POST /invocations with JSON body for inference.
    """
    input_data = request.get_json(force=True)
    query_str = input_data.get("query", "")
    top_k     = input_data.get("top_k", 3)

    # 1. Embed the query & retrieve docs from Faiss
    q_emb = embed_model.encode([query_str], show_progress_bar=False).astype("float32")
    D, I = index.search(q_emb, top_k)
    retrieved_docs = [metadata[doc_id] for doc_id in I[0]]

    # 2. Construct prompt & generate answer
    context = "\n\n".join(doc["summary"] for doc in retrieved_docs)
    prompt  = f"User Query:\n{query_str}\n\nRelevant Summaries:\n{context}\n\nAnswer:"
    inputs  = gen_tokenizer([prompt], return_tensors="pt", max_length=1024, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # If forcibly set, remove forced_bos_token_id
    if hasattr(gen_model.config, "forced_bos_token_id"):
        gen_model.config.forced_bos_token_id = None

    output_ids = gen_model.generate(
        inputs["input_ids"],
        num_beams=4,
        min_length=10,
        max_length=50,
        no_repeat_ngram_size=2
    )
    answer = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    # For local testing: python app.py  (runs Flask dev server on port 8080)
    app.run(host="0.0.0.0", port=8080, debug=True)

