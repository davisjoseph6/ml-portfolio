#!/usr/bin/env python3

"""
rag_query.py

Usage:
  python3 rag_query.py \
    --index_path faiss_index.bin \
    --metadata_path index_metadata.json \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --gen_model_dir ../my_summarization_model \
    --query "What does the shipping invoice say about the quantity of items?" \
    --top_k 3

Steps:
  1. Load FAISS index + metadata (which you built in build_rag_index.py).
  2. Embed the user query with the same model used to embed the summaries.
  3. Retrieve top-k relevant summaries from the index.
  4. Concatenate them into a context prompt.
  5. Use a seq2seq generative model (e.g. your fine-tuned summarizer in my_summarization_model) to generate a final answer.
"""

import os
import json
import argparse
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_faiss_index(index_path: str, metadata_path: str):
    """
    Loads a FAISS index from index_path and corresponding metadata from metadata_path.
    Returns (faiss_index, metadata_list).
    """
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def embed_text(texts, embed_model):
    """
    Embed a list of strings using a sentence-transformers model.
    Returns a numpy array of shape (len(texts), embedding_dim).
    """
    embeddings = embed_model.encode(texts, show_progress_bar=False)
    return embeddings.astype("float32")


def generate_answer(gen_model, gen_tokenizer, prompt, max_len=300):
    """
    Generate a final answer using the generative seq2seq model (BART, T5, etc.).
    'prompt' is the text input, 'max_len' is the max tokens for the output.
    """
    inputs = gen_tokenizer(
        [prompt],
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )
    if torch.cuda.is_available():
        gen_model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate the final answer
    output_ids = gen_model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_len,
        early_stopping=True
    )
    answer = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, default="faiss_index.bin",
                        help="Path to the FAISS index (built by build_rag_index.py).")
    parser.add_argument("--metadata_path", type=str, default="index_metadata.json",
                        help="Path to the metadata JSON with doc references.")
    parser.add_argument("--embedding_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Model name or path for embedding the user query.")
    parser.add_argument("--gen_model_dir", type=str,
                        default="../my_summarization_model",
                        help="Path to your fine-tuned seq2seq model for final generation.")
    parser.add_argument("--query", type=str, required=True,
                        help="User query to retrieve relevant docs and produce an answer.")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of top similar docs to retrieve from the index.")
    args = parser.parse_args()

    # 1. Load FAISS index & metadata
    index, metadata = load_faiss_index(args.index_path, args.metadata_path)
    print(f"Loaded FAISS index from {args.index_path}.")
    print(f"Index size: {index.ntotal}, Metadata size: {len(metadata)}")

    # 2. Load embedding model for queries
    print(f"Loading embedding model: {args.embedding_model}")
    embed_model = SentenceTransformer(args.embedding_model)

    # 3. Embed the user query
    print(f"Embedding user query: '{args.query}'")
    q_embedding = embed_text([args.query], embed_model)

    # 4. Search the index
    D, I = index.search(q_embedding, args.top_k)
    # D = distances, I = indices
    retrieved_docs = []
    for rank, doc_idx in enumerate(I[0]):
        dist = D[0][rank]
        doc_info = metadata[doc_idx]  # e.g. {"file_path": ..., "summary": ...}
        retrieved_docs.append((doc_info, dist))

    print(f"\nTop {args.top_k} retrieved docs:")
    for rank, (doc_info, dist) in enumerate(retrieved_docs):
        print(f"  {rank+1}. path={doc_info['file_path']}  dist={dist:.4f}")

    # 5. Build a context from the retrieved doc summaries
    # You can refine how you combine them—here we simply concat with newlines.
    context = "\n\n".join([doc["summary"] for doc, _dist in retrieved_docs])

    # 6. Load your generative model (fine-tuned summarizer)
    print(f"\nLoading generative model from {args.gen_model_dir}")
    gen_tokenizer = AutoTokenizer.from_pretrained(args.gen_model_dir)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.gen_model_dir)

    # 7. Create a prompt
    # You can prompt engineer as needed:
    prompt = (
        f"User Query: {args.query}\n\n"
        f"Relevant Summaries:\n{context}\n\n"
        f"Answer:"
    )

    # 8. Generate final answer
    print("\nGenerating final RAG answer...\n")
    answer = generate_answer(gen_model, gen_tokenizer, prompt, max_len=300)

    print("===== RAG ANSWER =====")
    print(answer)


if __name__ == "__main__":
    main()

