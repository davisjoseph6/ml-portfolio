#!/usr/bin/env python3

import os
import json
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def chunkify(lst, batch_size):
    """
    Yield successive batch_size-sized chunks from lst.
    """
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="sshleifer/distilbart-cnn-12-6",  # 1) Use DistilBART by default
        help="Pretrained summarization model name."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/preprocessed/",
        help="Root directory for your preprocessed JSON files."
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=1024,
        help="Max input length for the summarizer."
    )
    parser.add_argument(
        "--max_summary_length",
        type=int,
        default=150,
        help="Max summary length to generate."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of documents to summarize in a single batch."
    )
    args = parser.parse_args()

    print(f"Loading tokenizer & model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        print("Using GPU for summarization.")
    else:
        device = torch.device("cpu")
        print("No GPU found. Summarization will be slower on CPU.")

    # Recursively find all JSON files
    json_files = []
    for root, dirs, files in os.walk(args.data_dir):
        for f in files:
            if f.lower().endswith(".json"):
                json_files.append(os.path.join(root, f))

    print(f"Found {len(json_files)} JSON files under {args.data_dir}")
    pbar = tqdm(total=len(json_files), desc="Generating pseudo-summaries")

    # We'll process the files in batches
    for file_chunk in chunkify(json_files, args.batch_size):
        # Gather texts and store (file_path, data) for each
        texts = []
        data_objs = []  # Will hold tuples of (json_path, data)
        for json_path in file_chunk:
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                text = data.get("text", "").strip()
                if not text:
                    # No text? skip this file
                    pbar.update(1)
                    continue

                texts.append(text)
                data_objs.append((json_path, data))
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                pbar.update(1)

        # If this chunk has no valid texts, move on
        if not texts:
            continue

        # Tokenize the entire batch at once
        inputs = tokenizer(
            texts,
            max_length=args.max_input_length,
            truncation=True,
            return_tensors="pt",
            padding=True  # pad to the longest sequence
        )
        # Move inputs to same device as model
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # Generate summaries in a single forward pass
        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                num_beams=4,  # can reduce for faster generation
                max_length=args.max_summary_length,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        # Decode each summary and save back to its JSON
        for i, s_ids in enumerate(summary_ids):
            summary = tokenizer.decode(s_ids, skip_special_tokens=True)
            json_path, file_data = data_objs[i]
            file_data["summary"] = summary

            # Write updated JSON
            try:
                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump(file_data, jf, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"Error saving {json_path}: {e}")

        # Mark how many we've processed in the progress bar
        pbar.update(len(data_objs))

    pbar.close()
    print("Done generating pseudo-summaries.")

if __name__ == "__main__":
    main()

