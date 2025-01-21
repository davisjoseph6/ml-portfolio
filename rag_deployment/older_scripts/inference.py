import os
import json
import boto3
import torch
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# We'll store as globals or in a dictionary
def download_file_from_s3(s3_uri, local_path):
    s3 = boto3.client('s3')
    assert s3_uri.startswith("s3://")
    no_prefix = s3_uri[5:]
    bucket, key = no_prefix.split("/", 1)
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded {s3_uri} to {local_path}")

def model_fn(model_dir):
    """
    model_fn is called by SageMaker at container startup to load your model.
    'model_dir' is typically /opt/ml/model, but we can also rely on env vars.
    """
    print("Loading model with model_fn...")
    # We can store them in a dictionary for reference
    faiss_index_s3 = os.environ.get("FAISS_INDEX_S3")
    meta_s3 = os.environ.get("METADATA_S3")

    if not faiss_index_s3 or not meta_s3:
        raise ValueError("Missing FAISS_INDEX_S3 or METADATA_S3 environment variables")

    download_file_from_s3(faiss_index_s3, "/opt/ml/model/faiss_index.bin")
    download_file_from_s3(meta_s3, "/opt/ml/model/index_metadata.json")

    # Load index
    index = faiss.read_index("/opt/ml/model/faiss_index.bin")
    with open("/opt/ml/model/index_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    EMBEDDING_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    GEN_MODEL_NAME = os.environ.get("GEN_MODEL_NAME", "my_summarization_model")
    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    gen_model.eval()

    if torch.cuda.is_available():
        gen_model.to("cuda")

    print("Model load complete (model_fn).")

    # Return a dictionary containing everything we need
    return {
        "index": index,
        "metadata": metadata,
        "embed_model": embed_model,
        "gen_tokenizer": gen_tokenizer,
        "gen_model": gen_model
    }

def _embed_query(query_text, embed_model):
    return embed_model.encode([query_text], show_progress_bar=False).astype("float32")

def _retrieve_top_k(query_text, model_dict, top_k=3):
    index = model_dict["index"]
    metadata = model_dict["metadata"]
    embed_model = model_dict["embed_model"]

    q_emb = _embed_query(query_text, embed_model)
    D, I = index.search(q_emb, top_k)
    docs = []
    for rank, doc_id in enumerate(I[0]):
        doc_info = metadata[doc_id]
        docs.append(doc_info)
    return docs

def _generate_answer(prompt, model_dict, min_len=10, max_len=300):
    tokenizer = model_dict["gen_tokenizer"]
    g_model = model_dict["gen_model"]

    inputs = tokenizer([prompt], return_tensors="pt", max_length=1024, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    if hasattr(g_model.config, "forced_bos_token_id"):
        g_model.config.forced_bos_token_id = None

    output_ids = g_model.generate(
        inputs["input_ids"],
        num_beams=4,
        min_length=min_len,
        max_length=max_len,
        no_repeat_ngram_size=2,
        early_stopping=False
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def rag_infer(model_dict, query_text, top_k=3):
    docs = _retrieve_top_k(query_text, model_dict, top_k)
    context = "\n\n".join([doc["summary"] for doc in docs])
    prompt = f"User Query:\n{query_text}\n\nRelevant Summaries:\n{context}\n\nAnswer:"
    answer = _generate_answer(prompt, model_dict)
    return answer

def predict_fn(input_data, model_dict):
    """
    input_data: JSON-deserialized dictionary from the request.
    model_dict: the object returned by model_fn (our loaded model).
    """
    print("Running predict_fn...")

    query_str = input_data.get("query", "")
    top_k = input_data.get("top_k", 3)

    result = rag_infer(model_dict, query_str, top_k)
    return {"answer": result}

