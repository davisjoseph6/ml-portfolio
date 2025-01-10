#!/usr/bin/env python3

import os
import json
import argparse
from typing import List
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def list_json_files(directory: str) -> List[str]:
    """
    Recursively list all .json files in the given directory.
    """
    results = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".json"):
                results.append(os.path.join(root, f))
    return results

def generate_summary(model, tokenizer, text: str, max_length=150) -> str:
    """
    Generate a pseudo-summary from a pre-trained summarization model.
    """
    inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors='pt')
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=max_length)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_root", type=str, required=True,
                        help="Root directory of preprocessed data, e.g. ../data/preprocessed/")
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn",
                        help="Pre-trained model to generate pseudo-summaries (BART, DistilBART, etc.)")
    parser.add_argument("--max_summary_length", type=int, default=150,
                        help="Max tokens for generated summary.")
    args = parser.parse_args()

    # 1) Build paths to the test sets you want to modify:
    # e.g. supervised/test and unsupervised/test.
    test_dirs = [
        os.path.join(args.preprocessed_root, "supervised", "test"),
        os.path.join(args.preprocessed_root, "unsupervised", "test")
    ]

    # 2) Load the summarization model
    print(f"Loading summarizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # 3) For each test directory, find all JSON files
    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            print(f"Directory not found (skipping): {test_dir}")
            continue

        print(f"\nGenerating pseudo-summaries for: {test_dir}")
        json_files = list_json_files(test_dir)
        for fpath in tqdm(json_files, desc="Summarizing", unit="file"):
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # If there's no text or empty, skip
            text = data.get("text", "").strip()
            if not text:
                continue

            # Generate a draft summary
            summary = generate_summary(model, tokenizer, text, max_length=args.max_summary_length)
            data["summary"] = summary

            # Overwrite JSON
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    print("\nAll test documents now have 'summary' fields (pseudo-summaries).")

if __name__ == "__main__":
    main()

