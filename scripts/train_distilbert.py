#!/usr/bin/env python3

import os
import argparse
import tensorflow as tf
from transformers import DistilBertTokenizerFast
from data_prep import load_data_from_directory, build_label_map, create_tf_dataset
from model import create_distilbert_model
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="../data/preprocessed/supervised/train", type=str,
                        help="Path to the training data directory.")
    parser.add_argument("--val_dir", default="../data/preprocessed/supervised/val", type=str,
                        help="Path to the validation data directory.")
    parser.add_argument("--output_dir", default="../distilbert_model", type=str,
                        help="Directory to save the trained model.")
    parser.add_argument("--max_length", default=256, type=int,
                        help="Max sequence length for tokenization.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training.")
    parser.add_argument("--epochs", default=3, type=int,
                        help="Number of epochs to train.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="Learning rate for Adam.")
    # NEW FLAG:
    parser.add_argument("--no_train", action="store_true",
                        help="If set, skip the actual training and only build/save label map.")

    args = parser.parse_args()

    print("Loading training data...")
    X_train, y_train = load_data_from_directory(args.train_dir)
    print(f"Train samples: {len(X_train)}")

    print("Loading validation data...")
    X_val, y_val = load_data_from_directory(args.val_dir)
    print(f"Validation samples: {len(X_val)}")

    print("Building label map...")
    all_labels = y_train + y_val
    label2id, id2label = build_label_map(all_labels)
    num_labels = len(label2id)
    print(f"Number of labels: {num_labels}")

    # Create output_dir if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    # SAVE THE LABEL MAP IMMEDIATELY
    map_json_path = os.path.join(args.output_dir, "label_map.json")
    with open(map_json_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id}, f, indent=2)
    print(f"Saved label map to {map_json_path}")

    # If user sets --no_train, we exit here, skipping training
    if args.no_train:
        print("Exiting now (--no_train specified). No training will occur.")
        return

    print("Initializing tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    print("Creating tf.data.Dataset for train/val...")
    train_dataset = create_tf_dataset(
        texts=X_train,
        labels=y_train,
        label2id=label2id,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_dataset = create_tf_dataset(
        texts=X_val,
        labels=y_val,
        label2id=label2id,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=False
    )

    print("Creating DistilBERT model...")
    model = create_distilbert_model(num_labels=num_labels, learning_rate=args.learning_rate)
    
    print("Starting training...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        verbose=1  # Keras progress bar
    )

    # Create output_dir if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    print("Training complete.")

if __name__ == "__main__":
    main()

