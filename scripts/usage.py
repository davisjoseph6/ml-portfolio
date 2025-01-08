#!/usr/bin/env python3

import argparse
import os
import json
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../distilbert_model", type=str,
                        help="Path to the trained model directory.")
    parser.add_argument("--text", required=True, type=str,
                        help="Text to classify.")
    args = parser.parse_args()

    # 1. Load the tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = TFDistilBertForSequenceClassification.from_pretrained(args.model_dir)

    # 2. Optionally load the label map from JSON
    label_map_path = os.path.join(args.model_dir, "label_map.json")
    if not os.path.exists(label_map_path):
        print("Warning: No label_map.json found. Will only print numeric ID.")
        id2label = None
    else:
        with open(label_map_path, "r", encoding="utf-8") as f:
            saved_map = json.load(f)
        label2id = saved_map["label2id"]
        # Build reverse map: e.g., if label2id = {"invoice": 0, "memo": 1, ...}
        # then id2label = {0: "invoice", 1: "memo", ...}
        id2label = {int(v): k for k, v in label2id.items()}

    # 3. Tokenize the input text
    inputs = tokenizer(
        [args.text],
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors='tf'
    )

    # 4. Get model outputs
    outputs = model(inputs)
    logits = outputs.logits
    predicted_class_id = tf.argmax(logits, axis=-1).numpy()[0]

    # 5. Print results
    print(f"Predicted class ID: {predicted_class_id}")

    # If we have a label map, we can also show the label name
    if id2label is not None:
        label_name = id2label.get(predicted_class_id, "Unknown")
        print(f"Predicted label: {label_name}")

if __name__ == "__main__":
    main()

