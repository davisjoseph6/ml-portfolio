#!/usr/bin/env python3

import os
import json
from typing import List, Tuple, Dict
from tqdm import tqdm

import tensorflow as tf
from transformers import DistilBertTokenizerFast

def load_data_from_directory(root_dir: str) -> Tuple[List[str], List[str]]:
    """
    Recursively load data from the given directory,
    returning a tuple (list_of_texts, list_of_labels).
    The label is inferred from the folder name.
    """
    texts = []
    labels = []

    # Walk through all subdirectories
    for subdir, dirs, files in os.walk(root_dir):
        label_name = os.path.basename(subdir)
        # Skip if it's the root_dir itself
        if subdir == root_dir:
            continue

        # If no files in this subdir, skip
        if not files:
            continue

        for file_name in tqdm(files, desc=f"Loading data from {subdir}", unit="file"):
            if file_name.lower().endswith(".json"):
                file_path = os.path.join(subdir, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    text = data.get("text", "").strip()
                    if not text:
                        continue  # skip empty text
                    texts.append(text)
                    labels.append(label_name)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return texts, labels

def build_label_map(labels_list: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds a mapping from label string to integer ID, and the reverse.
    """
    unique_labels = sorted(list(set(labels_list)))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label

def create_tf_dataset(
    texts: List[str],
    labels: List[str],
    label2id: Dict[str, int],
    tokenizer: DistilBertTokenizerFast,
    max_length: int = 256,
    batch_size: int = 16,
    shuffle: bool = True
) -> tf.data.Dataset:
    """
    Tokenize text data and create a tf.data.Dataset suitable for model training.
    """
    # Convert labels (strings) to label IDs
    y_ids = [label2id[label] for label in labels]

    # Tokenize all texts
    # If the dataset is large, you might prefer to batch tokenization
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length
    )

    def gen():
        for i in range(len(texts)):
            yield (
                {
                    "input_ids": encodings["input_ids"][i],
                    "attention_mask": encodings["attention_mask"][i]
                },
                y_ids[i]
            )

    # Define output signature for the generator
    output_signature = (
        {
            "input_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "attention_mask": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(texts))

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    # Simple test to verify functionality
    train_dir = "../data/preprocessed/supervised/train"
    X_train, y_train = load_data_from_directory(train_dir)
    print(f"Loaded {len(X_train)} samples from {train_dir}")
    
    label2id, id2label = build_label_map(y_train)
    print(f"label2id: {label2id}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    ds = create_tf_dataset(X_train, y_train, label2id, tokenizer, max_length=128, batch_size=8)
    for batch in ds.take(1):
        print("Example batch:", batch)
        break

