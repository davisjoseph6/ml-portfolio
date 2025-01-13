#!/usr/bin/env python3

"""
rag_query_v4.py

Merges the best aspects of rag_query_v2.py and rag_query_v3.py:

Usage Example:
  python3 rag_query_v4.py \
    --index_path faiss_index.bin \
    --metadata_path index_metadata.json \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --gen_model_dir ../my_summarization_model \
    --query "What does the shipping invoice say about the quantity of items?" \
    --top_k 3 \
    --debug

Key Points:
  - We set generation parameters to avoid empty answers.
  - We print additional debug info if --debug is set.
  - We forcibly disable BART's forced BOS token if needed.
  - We also allow a 'min_length' so the model must produce a certain number of tokens.

Steps:
  1) Load FAISS index & metadata.
  2) Embed user query with the same model used to embed docs.
  3) Search top_k results.
  4) Build a context from their summaries.
  5) Use a seq2seq model to generate a final answer with more robust generation params.
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
    """Loads FAISS index and metadata JSON."""
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def embed_text(texts, embed_model):
    """Embed a list of strings using a sentence-transformers model. Returns float32 array."""
    embeddings = embed_model.encode(texts, show_progress_bar=False)
    return embeddings.astype("float32")


def generate_answer(
    gen_model,
    gen_tokenizer,
    prompt,
    max_len=300,
    min_len=10,
    no_repeat_ngram_size=2,
    debug=False
):
    """
    Generate a final answer using the generative seq2seq model (BART, T5, etc.).
    - min_len ensures some minimal length to avoid empty answers.
    - no_repeat_ngram_size to reduce duplication.
    """
    inputs = gen_tokenizer(
        [prompt],
        return_tensors="pt",
        max_length=1024,    # limit for input truncation
        truncation=True
    )

    if torch.cuda.is_available():
        gen_model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Often for BART, forced_bos_token_id can cause quirks, so we set it to None
    # if the model config allows it.
    if hasattr(gen_model.config, "forced_bos_token_id"):
        gen_model.config.forced_bos_token_id = None

    # Let's run generation with some robust parameters
    output_ids = gen_model.generate(
        inputs["input_ids"],
        num_beams=4,
        min_length=min_len,
        max_length=max_len,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=False  # let it generate up to max_len
    )
    answer = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if debug:
        print("\nDEBUG: Generation config used =>")
        print(f"  min_length={min_len}, max_length={max_len}, no_repeat_ngram_size={no_repeat_ngram_size}")
        print("Generated answer tokens:", output_ids[0])
        print("------")

    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, default="faiss_index.bin",
                        help="Path to the FAISS index file.")
    parser.add_argument("--metadata_path", type=str, default="index_metadata.json",
                        help="Path to the JSON metadata file.")
    parser.add_argument("--embedding_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Model name for embedding user query.")
    parser.add_argument("--gen_model_dir", type=str,
                        default="../my_summarization_model",
                        help="Path to your fine-tuned seq2seq model for final generation.")
    parser.add_argument("--query", type=str, required=True,
                        help="The user query to answer with RAG.")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of top docs to retrieve from FAISS.")
    parser.add_argument("--max_gen_length", type=int, default=300,
                        help="Max tokens in the final generated answer.")
    parser.add_argument("--min_gen_length", type=int, default=10,
                        help="Min tokens in the final generated answer.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2,
                        help="Avoid repeating n-grams for a cleaner output.")
    parser.add_argument("--debug", action="store_true",
                        help="If set, print extra debugging info.")
    args = parser.parse_args()

    # 1. Load FAISS index & metadata
    index, metadata = load_faiss_index(args.index_path, args.metadata_path)
    print(f"Index loaded. Size: {index.ntotal}, Metadata entries: {len(metadata)}")

    # 2. Load embedding model
    print(f"Loading embedding model: {args.embedding_model}")
    embed_model = SentenceTransformer(args.embedding_model)

    # 3. Embed user query
    if args.debug:
        print(f"\nUser Query: {args.query}")

    q_embedding = embed_text([args.query], embed_model)
    D, I = index.search(q_embedding, args.top_k)

    # 4. Retrieve top-k
    retrieved_docs = []
    for rank, doc_idx in enumerate(I[0]):
        dist = D[0][rank]
        doc_info = metadata[doc_idx]
        retrieved_docs.append((doc_info, dist))

    print(f"\nTop {args.top_k} retrieved docs:")
    for rank, (doc_info, dist) in enumerate(retrieved_docs):
        print(f"  {rank+1}. path={doc_info['file_path']} dist={dist:.4f}")

    # 5. Build context from doc summaries
    #    You might do more fancy formatting if you want.
    context = "\n\n".join([doc_info["summary"] for doc_info, _dist in retrieved_docs])

    # Optional debugging: show retrieved summaries
    if args.debug:
        print("\nRetrieved Summaries:\n")
        for i, (doc_info, dist) in enumerate(retrieved_docs):
            print(f"--- Summary {i+1} (dist={dist:.4f}):\n{doc_info['summary']}\n")

    # 6. Load generative model
    print(f"\nLoading generator model from {args.gen_model_dir}")
    gen_tokenizer = AutoTokenizer.from_pretrained(args.gen_model_dir)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.gen_model_dir)

    # 7. Prompt construction
    prompt = (
        f"User Query:\n{args.query}\n\n"
        f"Relevant Summaries:\n{context}\n\n"
        f"Answer:"
    )

    if args.debug:
        print("\nDEBUG: Final Prompt =>\n")
        print(prompt)
        print("-------")

    # 8. Generate final answer
    answer = generate_answer(
        gen_model,
        gen_tokenizer,
        prompt,
        max_len=args.max_gen_length,
        min_len=args.min_gen_length,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        debug=args.debug
    )

    # 9. Print result
    print("\n===== RAG ANSWER =====")
    print(answer)


if __name__ == "__main__":
    main()

