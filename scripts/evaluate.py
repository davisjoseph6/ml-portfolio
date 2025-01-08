#!/usr/bin/env python3

import os
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification

# Import your data loading & dataset creation utilities
from data_prep import load_data_from_directory, create_tf_dataset

import json  # <-- We'll load the label map JSON

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="../data/preprocessed/supervised/test", type=str,
                        help="Path to the test data directory.")
    parser.add_argument("--model_dir", default="../distilbert_model", type=str,
                        help="Path to the trained model directory.")
    parser.add_argument("--max_length", default=256, type=int,
                        help="Max sequence length for tokenization.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for inference.")
    args = parser.parse_args()

    print("Loading test data...")
    X_test, y_test = load_data_from_directory(args.test_dir)
    print(f"Test samples: {len(X_test)}")

    # ------------------------------------------------------------------------
    # REMOVE the old label map build from test data:
    #     label2id, id2label = build_label_map(y_test)
    #
    # INSTEAD, load the label map JSON that you saved in train_distilbert.py
    # ------------------------------------------------------------------------
    label_map_path = os.path.join(args.model_dir, "label_map.json")
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(
            f"Could not find label_map.json at {label_map_path}. "
            "Make sure you've run train_distilbert.py --no_train or a full training to generate it."
        )

    with open(label_map_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    label2id = saved_data["label2id"]

    # Reverse map: from index -> label string
    # e.g., if label2id = {"advertisement":0, "invoice":1, ...}
    id2label = {int(v): k for k, v in label2id.items()}

    print("Loading tokenizer and model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = TFDistilBertForSequenceClassification.from_pretrained(args.model_dir)

    print("Creating test dataset...")
    # Use the label2id from training, not from test data
    test_dataset = create_tf_dataset(
        texts=X_test,
        labels=y_test,
        label2id=label2id,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=False
    )

    print("Performing inference on test data...")
    y_true = []
    y_pred = []

    # Batch inference
    for batch in tqdm(test_dataset, desc="Evaluating", unit="batch"):
        inputs, labels = batch
        logits = model(inputs, training=False).logits
        preds = tf.argmax(logits, axis=-1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.numpy().tolist())

    print("\nCalculating metrics...")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # ------------------------------------------------------------------------
    # Build a consistent label ordering. We'll get the label names from the
    # label2id dictionary keys, sorted by alphabetical order or by ID:
    # ------------------------------------------------------------------------
    sorted_label_names = sorted(label2id.keys())  
    # Or if you want them in ascending ID order, you can do:
    #   sorted_label_names = [id2label[i] for i in range(len(id2label))]

    # classification_report requires we specify 'labels' as the numeric IDs
    # and 'target_names' as the string labels in that same order.
    label_indices = range(len(sorted_label_names))

    print("Classification Report:")
    report = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=sorted_label_names
    )
    print(report)

    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()

