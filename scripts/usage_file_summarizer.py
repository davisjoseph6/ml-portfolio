#!/usr/bin/env python3

import os
import argparse
import fitz            # PyMuPDF for PDF reading
import pytesseract     # for OCR
from PIL import Image  # for image reading
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF (fitz)."""
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

def extract_text_from_file(file_path):
    """Determine file type by extension and extract text accordingly."""
    file_ext = file_path.lower().split('.')[-1]
    if file_ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_ext in ["jpg", "jpeg", "png"]:
        return extract_text_from_image(file_path)
    elif file_ext == "txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

def summarize_text(model, tokenizer, text, max_input_length=1024, max_summary_length=150):
    """Generate a summary from text using the loaded seq2seq model."""
    inputs = tokenizer(
        [text],
        max_length=max_input_length,
        truncation=True,
        return_tensors='pt'
    )
    if torch.cuda.is_available():
        model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        print("Using GPU for summarization.")
    else:
        print("No GPU found. Summarization on CPU may be slow...")

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_summary_length,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str,
                        help="Path to your fine-tuned summarization model directory.")
    parser.add_argument("--file_path", required=True, type=str,
                        help="Path to the file (PDF, image, or txt) to summarize.")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Max input sequence length for the model.")
    parser.add_argument("--max_summary_length", type=int, default=150,
                        help="Max tokens for the summary output.")
    args = parser.parse_args()

    # 1. Extract text from the single file
    text = extract_text_from_file(args.file_path)
    if not text.strip():
        print("No text extracted from file—cannot summarize an empty string.")
        return

    # 2. Load Summarization Model and Tokenizer
    print(f"Loading summarization model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    # 3. Summarize the extracted text
    print(f"\nSummarizing file: {args.file_path}")
    summary = summarize_text(
        model,
        tokenizer,
        text,
        max_input_length=args.max_input_length,
        max_summary_length=args.max_summary_length
    )

    # 4. Print or save the summary
    print("\nGenerated Summary:")
    print(summary)

if __name__ == "__main__":
    main()

