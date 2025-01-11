#!/usr/bin/env python3

import os
import argparse
import json

os.environ["WANDB_DISABLED"] = "true"

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from typing import Dict, List

from tqdm import tqdm  # <-- Import tqdm for progress bars

def load_summarization_data(root_dir: str):
    """
    Recursively load text and summary from JSON in `root_dir`.
    Returns a list of dicts: [{"text": ..., "summary": ...}, ...]
    Provides a progress bar for the files it loads.
    """
    samples = []
    all_json_files = []

    # First, gather a list of all .json files
    for subdir, dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(".json"):
                all_json_files.append(os.path.join(subdir, fname))

    # Now iterate with a progress bar
    for fpath in tqdm(all_json_files, desc=f"Loading data from {root_dir}"):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text = data.get("text", "").strip()
            summary = data.get("summary", "").strip()
            if text and summary:
                samples.append({"text": text, "summary": summary})
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    return samples

def preprocess_function(examples, tokenizer, max_input_length=1024, max_target_length=150):
    """
    Tokenize the input text and summary (labels).
    """
    inputs = examples["text"]
    targets = examples["summary"]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True
    )

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-base",
                        help="Base model for summarization.")
    parser.add_argument("--supervised_dir", type=str,
                        default="../data/preprocessed/supervised/train",
                        help="Directory with supervised data having text+summary.")
    parser.add_argument("--unsupervised_dir", type=str,
                        default="../data/preprocessed/unsupervised/train",
                        help="Directory with unsupervised data if it has pseudo-summaries.")
    parser.add_argument("--output_dir", type=str,
                        default="../my_summarization_model",
                        help="Where to save the fine-tuned model.")
    parser.add_argument("--train_batch_size", type=int, default=2,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=2,
                        help="Batch size for evaluation.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Max tokens for the document input.")
    parser.add_argument("--max_target_length", type=int, default=150,
                        help="Max tokens for the summary.")
    args = parser.parse_args()

    print("Loading base model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # 1. Load supervised data (train set)
    print(f"Loading supervised data from: {args.supervised_dir}")
    sup_data = load_summarization_data(args.supervised_dir)
    print(f"Found {len(sup_data)} supervised samples with reference summaries.")

    # 2. Optionally load unsupervised data if they have a summary (silver-standard)
    unsup_data = []
    if os.path.exists(args.unsupervised_dir):
        print(f"Loading unsupervised data from: {args.unsupervised_dir}")
        unsup_data = load_summarization_data(args.unsupervised_dir)
        print(f"Found {len(unsup_data)} unsupervised samples with reference summaries.")
    else:
        print(f"No unsupervised directory found at {args.unsupervised_dir}")

    # Combine the two
    train_data = sup_data + unsup_data
    if not train_data:
        print("No training data found! Exiting.")
        return

    # Convert to Hugging Face dataset
    train_dataset = Dataset.from_list(train_data)

    # If you have a validation set, load it similarly or do a train/val split
    # For demonstration, we do a quick 90/10 train/val split:
    split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 3. Prepare tokenization function with progress bars
    def tokenize_fn(examples):
        return preprocess_function(
            examples,
            tokenizer,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length
        )

    # Tokenize with progress bar for train & val sets
    print("Tokenizing train dataset...")
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        desc="Tokenizing train data"
    )

    print("Tokenizing validation dataset...")
    val_dataset = val_dataset.map(
        tokenize_fn,
        batched=True,
        desc="Tokenizing val data"
    )

    # 4. Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 5. Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # use FP16 if GPU supports it
    )

    # 6. Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 7. Train
    print("Starting fine-tuning!")
    trainer.train()

    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")

if __name__ == "__main__":
    main()

