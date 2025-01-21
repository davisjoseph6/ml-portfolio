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

# ---------------------------------------------------------------------
# (Optional) Download model artifacts from S3 in global scope
# so that we only do it once at container startup.
# ---------------------------------------------------------------------
def download_file_from_s3(s3_uri, local_path):
    s3 = boto3.client('s3')
    assert s3_uri.startswith("s3://")
    no_prefix = s3_uri[5:]
    bucket, key = no_prefix.split("/", 1)
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded {s3_uri} to {local_path}")


print("Starting up container...")

FAISS_INDEX_S3 = os.environ.get("FAISS_INDEX_S3")
METADATA_S3    = os.environ.get("METADATA_S3")
EMBED_MODEL    = os.environ.get("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL      = os.environ.get("GEN_MODEL_NAME", "my_summarization_model")

if FAISS_INDEX_S3 and METADATA_S3:
    download_file_from_s3(FAISS_INDEX_S3, "/opt/ml/model/faiss_index.bin")
    download_file_from_s3(METADATA_S3, "/opt/ml/model/index_metadata.json")

# Now load the FAISS index and metadata
index = faiss.read_index("/opt/ml/model/faiss_index.bin")
with open("/opt/ml/model/index_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load embedding model
embed_model = SentenceTransformer(EMBED_MODEL)

# Load generation model
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
gen_model.eval()
if torch.cuda.is_available():
    gen_model.to("cuda")

print("Container initialization done. Model is loaded.")

# ---------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------
@app.route("/ping", methods=["GET"])
def ping():
    # Return 200 if the container is healthy
    return "OK", 200


# ---------------------------------------------------------------------
# Inference route
# ---------------------------------------------------------------------
@app.route("/invocations", methods=["POST"])
def invocations():
    # The incoming data is typically JSON
    input_data = request.get_json(force=True)
    query_str = input_data.get("query", "")
    top_k     = input_data.get("top_k", 3)

    # 1. Embed the query, do Faiss retrieval
    q_emb = embed_model.encode([query_str], show_progress_bar=False).astype("float32")
    D, I = index.search(q_emb, top_k)
    docs = [metadata[doc_id] for doc_id in I[0]]

    # 2. Construct prompt and generate answer
    context = "\n\n".join(doc["summary"] for doc in docs)
    prompt  = f"User Query:\n{query_str}\n\nRelevant Summaries:\n{context}\n\nAnswer:"
    inputs  = gen_tokenizer([prompt], return_tensors="pt", max_length=1024, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    if hasattr(gen_model.config, "forced_bos_token_id"):
        gen_model.config.forced_bos_token_id = None

    output_ids = gen_model.generate(
        inputs["input_ids"],
        num_beams=4,
        min_length=10,
        max_length=300,
        no_repeat_ngram_size=2,
        early_stopping=False
    )
    answer = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    # For local testing, you can run `python app.py` which
    # will launch Flask in debug mode on port 8080
    app.run(host="0.0.0.0", port=8080, debug=True)

