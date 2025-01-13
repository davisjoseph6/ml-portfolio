#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_faiss_index(index_path: str, metadata_path: str):
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def embed_text(texts, embed_model):
    embeddings = embed_model.encode(texts, show_progress_bar=False)
    return embeddings.astype("float32")

def generate_answer(gen_model, gen_tokenizer, prompt, max_len=300):
    inputs = gen_tokenizer(
        [prompt],
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )
    if torch.cuda.is_available():
        gen_model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Optionally adjust generation parameters
    gen_model.config.forced_bos_token_id = None  # or 0
    output_ids = gen_model.generate(
        inputs["input_ids"],
        num_beams=4,
        min_length=10,             # Force at least 10 tokens
        max_length=max_len,
        no_repeat_ngram_size=2,
        early_stopping=False       # let model continue a bit
    )
    answer = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, default="faiss_index.bin")
    parser.add_argument("--metadata_path", type=str, default="index_metadata.json")
    parser.add_argument("--embedding_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--gen_model_dir", type=str,
                        default="../my_summarization_model")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    # 1. Load Index
    index, metadata = load_faiss_index(args.index_path, args.metadata_path)
    print(f"Index size: {index.ntotal}, metadata size: {len(metadata)}")

    # 2. Embedding model
    embed_model = SentenceTransformer(args.embedding_model)

    # 3. Embed query
    q_embedding = embed_text([args.query], embed_model)
    D, I = index.search(q_embedding, args.top_k)
    retrieved_docs = []
    for rank, doc_idx in enumerate(I[0]):
        dist = D[0][rank]
        doc_info = metadata[doc_idx]
        retrieved_docs.append((doc_info, dist))

    print("\nTop retrieved docs:")
    for rank, (doc_info, dist) in enumerate(retrieved_docs):
        print(f"  {rank+1}. path={doc_info['file_path']}  dist={dist:.4f}")

    # Debug: see if summaries are empty
    print("\nRetrieved Summaries:")
    for i, (doc_info, dist) in enumerate(retrieved_docs):
        print(f"--- Summary {i+1} (dist={dist:.4f}):\n{doc_info['summary']}\n")

    # Build context
    context = "\n\n".join([doc["summary"] for doc, _ in retrieved_docs])

    # 4. Summarization model
    gen_tokenizer = AutoTokenizer.from_pretrained(args.gen_model_dir)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.gen_model_dir)

    # 5. Prompt
    prompt = (
        f"User Query:\n{args.query}\n\n"
        f"Relevant Summaries:\n{context}\n\n"
        f"Answer:"
    )

    print("\nDEBUG: Prompt being sent to the model:\n")
    print(prompt)
    print("-------")

    # 6. Generate
    answer = generate_answer(gen_model, gen_tokenizer, prompt, max_len=512)

    print("\n===== RAG ANSWER =====")
    print(answer)

if __name__ == "__main__":
    main()

