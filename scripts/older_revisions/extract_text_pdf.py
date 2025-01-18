#!/usr/bin/env python3

import fitz  # PyMuPDF
import json
import os

def extract_text_from_pdf(file_path):
    """Extract text and metadata from a PDF file."""
    with fitz.open(file_path) as doc:
        metadata = doc.metadata
        text = ""
        for page in doc:
            text += page.get_text()
    return metadata, text

def process_pdfs(input_dir, output_dir):
    """Process all PDF files in the input directory and save extracted data as JSON."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(input_dir, filename)
            metadata, text = extract_text_from_pdf(file_path)
            
            # Prepare JSON data
            data = {
                "filename": filename,
                "metadata": metadata,
                "text": text
            }
            
            # Save JSON
            json_filename = filename.replace('.pdf', '.json')
            with open(os.path.join(output_dir, json_filename), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Processed and saved: {json_filename}")

if __name__ == "__main__":
    # Define input and output directories
    INPUT_DIR = "/home/davis/ml-portfolio/data/preprocessed/pdfs/"
    OUTPUT_DIR = "/home/davis/ml-portfolio/data/preprocessed/pdfs_json/"
    
    process_pdfs(INPUT_DIR, OUTPUT_DIR)
    print("PDF preprocessing completed.")

