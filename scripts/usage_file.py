#!/usr/bin/env python3

import os
import argparse
import fitz            # PyMuPDF for PDF
import pytesseract     # for OCR
from PIL import Image  # for image reading
import json
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification

def extract_text_from_file(file_path):
    file_ext = file_path.lower().split('.')[-1]
    text = ""

    if file_ext == "pdf":
        text = extract_text_from_pdf(file_path)
    elif file_ext in ["jpg", "jpeg", "png"]:
        text = extract_text_from_image(file_path)
    elif file_ext == "txt":
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

    return text.strip()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using fitz (PyMuPDF)."""
    text_content = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_content += page.get_text()
    return text_content

def extract_text_from_image(img_path):
    """Extract text from an image using Tesseract OCR."""
    image = Image.open(img_path)
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_txt(txt_path):
    """Read plain text from a .txt file."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../distilbert_model", type=str,
                        help="Path to the trained model directory.")
    parser.add_argument("--file_path", required=True, type=str,
                        help="Path to the file (PDF, image, or txt) to classify.")
    args = parser.parse_args()

    # 1. Extract text from the single file
    text = extract_text_from_file(args.file_path)
    if not text:
        print("No text extracted. Cannot classify an empty string.")
        return

    # 2. Load the DistilBERT model
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = TFDistilBertForSequenceClassification.from_pretrained(args.model_dir)

    # 3. (Optional) Load label_map.json for printing label names
    label_map_path = os.path.join(args.model_dir, "label_map.json")
    id2label = None
    if os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            saved_map = json.load(f)
        label2id = saved_map["label2id"]
        id2label = {int(v): k for k, v in label2id.items()}

    # 4. Tokenize extracted text
    inputs = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors='tf'
    )

    # 5. Inference
    outputs = model(inputs)
    logits = outputs.logits
    predicted_class_id = tf.argmax(logits, axis=-1).numpy()[0]

    print(f"\nFile: {args.file_path}")
    print(f"Predicted class ID: {predicted_class_id}")

    if id2label is not None:
        label_name = id2label.get(predicted_class_id, "Unknown")
        print(f"Predicted label: {label_name}")
    else:
        print("No label_map.json found, showing numeric ID only.")

if __name__ == "__main__":
    main()

