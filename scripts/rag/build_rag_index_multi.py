#!/usr/bin/env python3

"""
build_rag_index_multi.py

Usage example:
    python3 build_rag_index_multi.py \
      --input_dirs "../../data/summarized/supervised/train,../../data/summarized/unsupervised/train" \
      --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
      --index_path faiss_index.bin \
      --metadata_path index_metadata.json

What it does:
  1. Allows you to specify multiple directories (comma-separated) via --input_dirs.
  2. For each directory, looks for .json files that presumably contain either:
       - "summary" field, OR
       - "chunk_summaries": [list of summary chunks].
  3. Gathers all of these into a single list, then embeds them with a SentenceTransformer model.
  4. Builds a single FAISS index (faiss_index.bin) plus a single metadata file (index_metadata.json).

Then you can run 'rag_query.py' (or your custom script) against that single combined index.
"""

import os
import json
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List

def load_summarized_docs_multi(input_dirs: List[str]) -> List[dict]:
    """
    For each directory in 'input_dirs', recursively find all .json files.
    Each file is expected to have:
      - "summary" field (if single summary), or
      - "chunk_summaries" (list of summary chunks), or
      - Possibly no summary if you want to fallback on 'text' (optional).
    Returns a combined list of dicts: {"file_path": str, "summary": str}.
    """
    docs = []
    for dir_path in input_dirs:
        dir_path = dir_path.strip()
        if not os.path.exists(dir_path):
            print(f"Warning: {dir_path} does not exist. Skipping.")
            continue

        for subdir, dirs, files in os.walk(dir_path):
            for fname in files:
                if fname.lower().endswith(".json"):
                    fpath = os.path.join(subdir, fname)
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # If chunk_summaries exist, treat them individually
                        if "chunk_summaries" in data and isinstance(data["chunk_summaries"], list):
                            for idx, chunk_sum in enumerate(data["chunk_summaries"]):
                                if chunk_sum.strip():
                                    docs.append({
                                        "file_path": f"{fpath}::chunk_{idx}",
                                        "summary": chunk_sum.strip()
                                    })
                        else:
                            # Otherwise, check for a single "summary" field
                            summary = data.get("summary", "").strip()
                            if summary:
                                docs.append({
                                    "file_path": fpath,
                                    "summary": summary
                                })
                            else:
                                # (Optional) fallback on raw text if you prefer
                                raw_text = data.get("text", "").strip()
                                if raw_text:
                                    docs.append({
                                        "file_path": fpath,
                                        "summary": raw_text
                                    })

                    except Exception as e:
                        print(f"Error reading {fpath}: {e}")
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", type=str, required=True,
                        help="Comma-separated list of directories. E.g. ../data/summarized/supervised/train,../data/summarized/unsupervised/train")
    parser.add_argument("--embedding_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Which SentenceTransformer model to use for embedding.")
    parser.add_argument("--index_path", type=str,
                        default="faiss_index.bin",
                        help="Output path for the FAISS index.")
    parser.add_argument("--metadata_path", type=str,
                        default="index_metadata.json",
                        help="Output path for the JSON metadata.")
    args = parser.parse_args()

    # 1. Parse the input_dirs argument into a list
    input_dir_list = [d.strip() for d in args.input_dirs.split(",")]
    print(f"Input directories: {input_dir_list}")

    # 2. Load all docs from these directories
    docs = load_summarized_docs_multi(input_dir_list)
    print(f"Found {len(docs)} items (summary or chunked) across all input directories.")

    if not docs:
        print("No summarized docs found. Exiting.")
        return

    # 3. Load the embedding model
    print(f"Loading embedding model: {args.embedding_model}")
    embed_model = SentenceTransformer(args.embedding_model)

    # 4. Embed each summary
    texts_to_embed = [d["summary"] for d in docs]
    print("Embedding texts...")
    embeddings = embed_model.encode(texts_to_embed, batch_size=16, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # 5. Build a FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index size: {index.ntotal}")

    # 6. Save the index
    print(f"Saving FAISS index to {args.index_path}")
    faiss.write_index(index, args.index_path)

    # 7. Build & save metadata
    metadata = []
    for d_ in docs:
        metadata.append({
            "file_path": d_["file_path"],
            "summary": d_["summary"]
        })

    print(f"Saving metadata to {args.metadata_path}")
    with open(args.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("RAG index build complete (combined)!")


if __name__ == "__main__":
    main()

