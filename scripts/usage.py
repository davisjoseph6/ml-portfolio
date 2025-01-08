#!/usr/bin/env python3

import argparse
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../distilbert_model", type=str,
                        help="Path to the trained model directory.")
    parser.add_argument("--text", required=True, type=str,
                        help="Text to classify.")
    args = parser.parse_args()

    # Load the tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = TFDistilBertForSequenceClassification.from_pretrained(args.model_dir)

    # Tokenize the input text
    inputs = tokenizer(
        [args.text],
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors='tf'
    )

    # Get model outputs
    outputs = model(inputs)
    logits = outputs.logits
    predicted_class_id = tf.argmax(logits, axis=-1).numpy()[0]

    # In reality, you'd want to map predicted_class_id -> label name
    # But you need the label map used in training. For demonstration:
    print(f"Predicted class ID: {predicted_class_id}")
    # If you saved id2label as a local JSON, you could load and do:
    # print("Predicted label:", id2label[predicted_class_id])

if __name__ == "__main__":
    main()

