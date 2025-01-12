#!/usr/bin/env python3

"""
build_rag_index.py

Usage example:
    python3 build_rag_index.py \
      --root_dir ../data/summarized/unsupervised/train \
      --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
      --index_path faiss_index.bin \
      --metadata_path index_metadata.json

What it does:
  1. Looks for .json files in --root_dir (which presumably have a "summary" field).
  2. Embeds each summary using a SentenceTransformer (you can also embed chunk_summaries if you prefer).
  3. Builds a FAISS index and saves it to --index_path.
  4. Writes a metadata file (JSON list) referencing each doc's path and summary text.

You can later use 'rag_query.py' (another script) to load faiss_index.bin & index_metadata.json, embed queries, and retrieve top-k results for a final generative step.
"""

import os
import json
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List

def load_summarized_docs(root_dir: str) -> List[dict]:
    """
    Scans root_dir for .json files. Each file is expected to have at least:
        {
          "text": "...",
          "summary": "..."
        }
    or, if chunked, something like:
        {
          "text": "...",
          "chunk_summaries": ["summary1", "summary2", ...]
        }
    Returns a list of dictionaries with the fields "file_path" and "summary".
    """
    docs = []
    for subdir, dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(".json"):
                fpath = os.path.join(subdir, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # If you have chunk_summaries, you can store them individually
                if "chunk_summaries" in data and isinstance(data["chunk_summaries"], list):
                    for idx, chunk_sum in enumerate(data["chunk_summaries"]):
                        chunk_sum_str = chunk_sum.strip()
                        if chunk_sum_str:
                            docs.append({
                                "file_path": f"{fpath}::chunk_{idx}",
                                "summary": chunk_sum_str
                            })
                else:
                    # Otherwise, we expect a single "summary" field
                    summary = data.get("summary", "").strip()
                    if summary:
                        docs.append({
                            "file_path": fpath,
                            "summary": summary
                        })
    return docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str,
                        default="../data/summarized",
                        help="Root directory containing summarized .json files.")
    parser.add_argument("--embedding_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Model name for embedding the summaries.")
    parser.add_argument("--index_path", type=str,
                        default="faiss_index.bin",
                        help="Path to save the FAISS index.")
    parser.add_argument("--metadata_path", type=str,
                        default="index_metadata.json",
                        help="Path to save the metadata JSON.")
    args = parser.parse_args()

    print(f"Loading summarized docs from: {args.root_dir}")
    docs = load_summarized_docs(args.root_dir)
    print(f"Found {len(docs)} summarized items (files or chunks).")

    if not docs:
        print("No summarized docs found. Exiting.")
        return

    print(f"Loading embedding model: {args.embedding_model}")
    embed_model = SentenceTransformer(args.embedding_model)

    # 1. Embed each summary
    summaries = [d["summary"] for d in docs]
    print("Embedding summaries...")
    embeddings = embed_model.encode(summaries, batch_size=16, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # 2. Build a FAISS index (L2)
    dim = embeddings.shape[1]  # e.g. 384 for MiniLM
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index size: {index.ntotal}")

    # 3. Save index
    print(f"Saving FAISS index to {args.index_path}")
    faiss.write_index(index, args.index_path)

    # 4. Build & save metadata
    metadata = []
    for d in docs:
        metadata.append({
            "file_path": d["file_path"],
            "summary": d["summary"]
        })

    print(f"Saving metadata to {args.metadata_path}")
    with open(args.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("RAG index build complete!")

if __name__ == "__main__":
    main()

