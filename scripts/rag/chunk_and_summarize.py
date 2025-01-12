#!/usr/bin/env python3

"""
chunk_and_summarize.py

Usage example:
    python3 chunk_and_summarize.py \
      --input_dir ../data/preprocessed/unsupervised/train \
      --output_dir ../data/summarized/unsupervised/train \
      --model_dir ../my_summarization_model \
      --chunk_size 500 \
      --overlap 50

This script:
  1. Loads .json files from --input_dir, each containing {"text": ...} (and possibly "summary": ...).
  2. Splits text into chunks of length ~chunk_size tokens, with optional overlap.
  3. Summarizes each chunk using a local seq2seq model (fine-tuned summarizer).
  4. Saves each chunk's summary in new .json files under --output_dir.

Important: 
  - Adjust chunk_size/overlap to your needs.
  - If documents are short, you can skip chunking or set chunk_size large.
  - Overwrite logic vs. saving to a separate directory is your choice.
"""

import os
import json
import argparse
from typing import List
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    """
    Splits text into overlapping chunks of approximately 'chunk_size' tokens each.
    Overlap ensures continuity between chunks if needed.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)
        chunks.append(chunk_str)
        # Overlap for next chunk
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(words):
            break

    return chunks


def summarize_chunk(model, tokenizer, chunk, max_input_length=1024, max_summary_length=150):
    """
    Summarize a single text chunk using a seq2seq model (e.g., BART).
    """
    inputs = tokenizer(
        [chunk],
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt"
    )

    # If GPU available, move model & inputs
    if torch.cuda.is_available():
        model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    else:
        pass  # CPU fallback

    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_summary_length,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing .json files with 'text' field.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save chunked + summarized .json files.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to your fine-tuned summarization model directory.")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Approx number of tokens per chunk.")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Token overlap between consecutive chunks.")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Max input tokens for the summarizer.")
    parser.add_argument("--max_summary_length", type=int, default=150,
                        help="Max output tokens for each summary.")
    args = parser.parse_args()

    # 1. Load your locally fine-tuned summarization model
    print(f"Loading tokenizer/model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    # 2. Traverse input_dir for .json files
    json_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith(".json"):
                json_files.append(os.path.join(root, f))

    if not json_files:
        print(f"No .json files found in {args.input_dir}")
        return

    print(f"Found {len(json_files)} JSON files in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Process each file: chunk + summarize
    for file_path in tqdm(json_files, desc="Chunk + Summarize"):
        try:
            with open(file_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)

            text = data.get("text", "").strip()
            if not text:
                continue  # skip if no text

            # CHUNK
            chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
            if len(chunks) <= 1:
                # If doc is small, we might just have 1 chunk
                # Summarize once
                summary = summarize_chunk(
                    model, tokenizer, text,
                    max_input_length=args.max_input_length,
                    max_summary_length=args.max_summary_length
                )
                data["summary"] = summary
                # Save to output_dir
                rel_path = os.path.relpath(file_path, args.input_dir)
                out_path = os.path.join(args.output_dir, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as of:
                    json.dump(data, of, ensure_ascii=False, indent=4)
            else:
                # If doc is large, we do per-chunk summarization
                chunk_summaries = []
                for c in chunks:
                    chunk_sum = summarize_chunk(
                        model, tokenizer, c,
                        max_input_length=args.max_input_length,
                        max_summary_length=args.max_summary_length
                    )
                    chunk_summaries.append(chunk_sum)

                # We store them in e.g. an array
                # Or optionally combine chunk_summaries into one big doc summary
                data["chunk_summaries"] = chunk_summaries

                # Save updated data
                rel_path = os.path.relpath(file_path, args.input_dir)
                out_path = os.path.join(args.output_dir, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as of:
                    json.dump(data, of, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("Chunking & Summarization complete.")


if __name__ == "__main__":
    main()

