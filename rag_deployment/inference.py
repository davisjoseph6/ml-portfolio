#!/usr/bin/env python3

import os
import json
import boto3
import torch
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Globals
index = None
metadata = None
embed_model = None
gen_tokenizer = None
gen_model = None

def download_file_from_s3(s3_uri, local_path):
    """Utility to download a file from s3://... to local path."""
    s3 = boto3.client('s3')
    # Parse bucket and key
    # e.g. s3_uri = "s3://mybucket/path/to/file"
    assert s3_uri.startswith("s3://")
    no_prefix = s3_uri[5:]  # remove "s3://"
    bucket, key = no_prefix.split("/", 1)
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded {s3_uri} to {local_path}")

def model_load():
    """Load everything we need at container startup."""
    global index, metadata, embed_model, gen_tokenizer, gen_model

    # 1) Download FAISS index + metadata from S3
    faiss_index_s3 = os.environ.get("FAISS_INDEX_S3")  # set this in your Dockerfile or container env
    meta_s3 = os.environ.get("METADATA_S3")

    if faiss_index_s3 and meta_s3:
        download_file_from_s3(faiss_index_s3, "faiss_index.bin")
        download_file_from_s3(meta_s3, "index_metadata.json")
    else:
        raise ValueError("Missing environment variables FAISS_INDEX_S3 or METADATA_S3")

    # 2) Load FAISS index + metadata
    index = faiss.read_index("faiss_index.bin")
    with open("index_metadata.json","r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 3) Load embedding model
    # e.g. "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 4) Load summarization/generative model
    # e.g. your "my_summarization_model" tarball or local directory
    GEN_MODEL_NAME = os.environ.get("GEN_MODEL_NAME", "my_summarization_model")
    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    gen_model.eval()
    if torch.cuda.is_available():
        gen_model.to("cuda")

    print("Model load complete!")

def embed_query(query_text: str):
    emb = embed_model.encode([query_text], show_progress_bar=False)
    return emb.astype("float32")

def retrieve_top_k(query_text, top_k=3):
    q_emb = embed_query(query_text)
    D,I = index.search(q_emb, top_k)
    retrieved_docs = []
    for rank, doc_id in enumerate(I[0]):
        doc_info = metadata[doc_id]
        retrieved_docs.append(doc_info)
    return retrieved_docs

def generate_answer(prompt: str, min_len=10, max_len=300):
    inputs = gen_tokenizer([prompt], return_tensors="pt", max_length=1024, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # We disable forced bos token for BART if needed
    if hasattr(gen_model.config, "forced_bos_token_id"):
        gen_model.config.forced_bos_token_id = None

    output_ids = gen_model.generate(
        inputs["input_ids"],
        num_beams=4,
        min_length=min_len,
        max_length=max_len,
        no_repeat_ngram_size=2,
        early_stopping=False
    )
    return gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

def rag_infer(query_text, top_k=3):
    """Full RAG pipeline: embed -> retrieve -> generative answer."""
    # 1) Retrieve top-k summaries
    docs = retrieve_top_k(query_text, top_k=top_k)
    # 2) Build context
    context = "\n\n".join([doc["summary"] for doc in docs])
    # 3) Construct a prompt
    prompt = f"User Query:\n{query_text}\n\nRelevant Summaries:\n{context}\n\nAnswer:"
    # 4) Generate final answer
    answer = generate_answer(prompt)
    return answer

# SageMaker Python Inference Toolkit calls predict_fn(data, context) by default
def predict_fn(data, context):
    """
    data is a dictionary from the request payload.
    e.g. { "query": "What does the invoice say about quantity?", "top_k": 3 }
    """
    query_str = data.get("query", "")
    top_k = data.get("top_k", 3)

    # 1) Run the pipeline
    result = rag_infer(query_str, top_k=top_k)
    return {"answer": result}

# If you want to do some initialization after container starts
model_load()

# If running locally for a quick test:
if __name__ == "__main__":
    # Example usage
    test_query = "Where is the shipping_orders doc location?"
    out = rag_infer(test_query, top_k=2)
    print("Test RAG answer:", out)

