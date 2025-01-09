#!/usr/bin/env python3

import os
import json
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_summary(model, tokenizer, text, max_input_length=1024, max_summary_length=150):
    """
    Generate a summary using a pre-trained seq2seq model.
    """
    # Tokenize
    inputs = tokenizer(
        [text],
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt"
    )
    # Generate
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_summary_length,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    # Decode
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn",
                        help="Pretrained summarization model name.")
    parser.add_argument("--data_dir", type=str, default="../data/preprocessed/",
                        help="Root directory for your preprocessed JSON files.")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Max input length for the summarizer.")
    parser.add_argument("--max_summary_length", type=int, default=150,
                        help="Max summary length to generate.")
    args = parser.parse_args()

    print(f"Loading tokenizer & model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Move to GPU if available
    if torch.cuda.is_available():
        model.to("cuda")
        print("Using GPU for summarization.")
    else:
        print("No GPU found. Summarization will be slower on CPU.")

    # Recursively find all JSON files
    json_files = []
    for root, dirs, files in os.walk(args.data_dir):
        for f in files:
            if f.lower().endswith(".json"):
                json_files.append(os.path.join(root, f))

    print(f"Found {len(json_files)} JSON files under {args.data_dir}")
    for json_path in tqdm(json_files, desc="Generating pseudo-summaries"):
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                data = json.load(jf)

            text = data.get("text", "").strip()
            if not text:
                # no text? skip
                continue

            # If there's already a summary, we can skip or overwrite
            # if "summary" in data and data["summary"]:
            #     continue  # skip if we don't want to overwrite existing

            # Generate pseudo-summary
            summary = generate_summary(
                model,
                tokenizer,
                text,
                max_input_length=args.max_input_length,
                max_summary_length=args.max_summary_length
            )

            data["summary"] = summary

            # Save updated JSON
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(data, jf, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error processing {json_path}: {e}")

if __name__ == "__main__":
    main()

