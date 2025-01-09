#!/usr/bin/env python3
"""
summarize_docs.py

- Loads preprocessed JSON files from a specified directory (e.g. ../data/preprocessed/supervised/test).
- Summarizes each file's "text" using a BART/DistilBART model (default: distilbart-cnn-12-6).
- Outputs updated JSON (with a "summary" field) to a new directory (e.g. ../data/summarized/...).
"""

import os
import json
import argparse
from typing import List
from tqdm import tqdm

# Hugging Face imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_documents(root_dir: str) -> List[str]:
    """
    Recursively load text from JSON for summarization.
    Returns a list of (file_path, text).
    """
    docs = []
    for subdir, dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(".json"):
                fpath = os.path.join(subdir, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                text = data.get("text", "").strip()
                if text:
                    docs.append((fpath, text))
    return docs

def chunk_text(text: str, chunk_size=1024, overlap=100) -> List[str]:
    """
    If text is very long, break it into overlapping chunks to avoid
    token limits. Adjust chunk_size/overlap to your needs.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(words):
            break
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="../data/preprocessed/supervised/test",
                        help="Directory of JSON files to summarize.")
    parser.add_argument("--model_name", type=str,
                        default="sshleifer/distilbart-cnn-12-6",
                        help="Hugging Face model name for summarization.")
    parser.add_argument("--output_dir", type=str,
                        default="../data/summarized/supervised/test",
                        help="Directory where summarized JSON will be saved.")
    parser.add_argument("--chunk_size", type=int, default=1024,
                        help="Max words per chunk.")
    parser.add_argument("--overlap", type=int, default=100,
                        help="Word overlap between chunks.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading summarization model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    docs = load_documents(args.data_dir)
    print(f"Found {len(docs)} documents in {args.data_dir}")

    for (fpath, text) in tqdm(docs, desc="Summarizing"):
        # Possibly split text into overlapping chunks
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)

        # Summarize each chunk separately
        summaries = []
        for chunk in chunks:
            inputs = tokenizer([chunk], max_length=1024, truncation=True, return_tensors="pt")
            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=4,
                max_length=150,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary.strip())

        # Combine chunk-level summaries
        final_summary = "\n".join(summaries)

        # Load original JSON, add 'summary' field
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data["summary"] = final_summary

        # Write to output directory, preserving folder structure
        rel_path = os.path.relpath(fpath, args.data_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as fout:
            json.dump(data, fout, ensure_ascii=False, indent=4)

    print("\nSummarization complete.")

if __name__ == "__main__":
    main()

