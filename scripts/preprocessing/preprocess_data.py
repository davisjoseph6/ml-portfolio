#!/usr/bin/env python3

import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import json
import shutil

# For the progress bar
from tqdm import tqdm

# For text augmentation
import nlpaug.augmenter.word as naw

def extract_text_from_pdf(file_path):
    """Extract text and metadata from a PDF file."""
    try:
        with fitz.open(file_path) as doc:
            metadata = doc.metadata
            text = ""
            for page in doc:
                text += page.get_text()
        return metadata, text
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return {}, ""

def extract_text_from_image(file_path):
    """Extract text from an image using OCR."""
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error processing Image {file_path}: {e}")
        return ""

def process_txt_file(file_path):
    """Read text from a TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception as e:
        print(f"Error processing TXT {file_path}: {e}")
        return ""

def save_json(data, output_path):
    """Save data as a JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved JSON: {output_path}")
    except Exception as e:
        print(f"Error saving JSON {output_path}: {e}")

def augment_text(original_text, num_augments=2, aug_p=0.1):
    """
    Apply text augmentation to a single text using a synonym replacement strategy.
    - num_augments: how many augmented versions to produce
    - aug_p: proportion of words to be augmented
    """
    try:
        aug = naw.SynonymAug(aug_p=aug_p)
        # Generate multiple augmented versions if needed
        augmented_texts = aug.augment(original_text, n=num_augments)
        return augmented_texts if isinstance(augmented_texts, list) else [augmented_texts]
    except Exception as e:
        print(f"Error augmenting text: {e}")
        return []

def preprocess_file(file_path, raw_root, preprocessed_root, do_augmentation=True):
    """
    Determine file type and preprocess accordingly.
    Optionally apply text augmentation in a single pass.
    """
    relative_path = os.path.relpath(file_path, raw_root)
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    ext = ext.lower()

    # Define output JSON path
    output_dir = os.path.join(preprocessed_root, os.path.dirname(relative_path))
    output_filename = f"{name}.json"
    output_path = os.path.join(output_dir, output_filename)

    data = {
        "filename": filename,
        "metadata": {},
        "text": "",
        "augmented_texts": []  # We'll store augmented texts here
    }

    if ext == '.pdf':
        metadata, text = extract_text_from_pdf(file_path)
        data['metadata'] = metadata
        data['text'] = text
    elif ext == '.txt':
        text = process_txt_file(file_path)
        data['text'] = text
    elif ext in ['.jpeg', '.jpg', '.png']:
        text = extract_text_from_image(file_path)
        data['text'] = text
    else:
        print(f"Unsupported file type: {file_path}")
        return  # Skip unsupported file types

    # Apply text augmentation if requested
    if do_augmentation and data['text'].strip():
        data['augmented_texts'] = augment_text(data['text'], num_augments=2, aug_p=0.1)

    # Save result as JSON
    save_json(data, output_path)

def list_all_files(root_dir):
    """
    Recursively list all files under root_dir.
    Returns a list of absolute file paths.
    """
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_paths.append(os.path.join(dirpath, filename))
    return file_paths

def preprocess_directory(raw_root, preprocessed_root, do_augmentation=True):
    """Recursively preprocess all files in the raw data directory with a progress bar."""
    # Get the list of all files first
    all_files = list_all_files(raw_root)

    # Initialize a progress bar with the total number of files
    with tqdm(total=len(all_files), desc=f"Preprocessing {raw_root}") as pbar:
        for file_path in all_files:
            preprocess_file(file_path, raw_root, preprocessed_root, do_augmentation=do_augmentation)
            pbar.update(1)

def main():
    # Define raw and preprocessed directories
    base_dir = "/home/davis/ml-portfolio/data/raw/"
    preprocessed_base = "/home/davis/ml-portfolio/data/preprocessed/"

    # If you only want to run once, set do_augmentation=True
    do_augmentation = True

    # Define supervised and unsupervised directories
    categories = ['supervised', 'unsupervised']

    for category in categories:
        for split in ['train', 'val', 'test']:
            # Handle test_split1 and test_split2 as part of 'test' to maintain your structure
            for test_split in ['test', 'test_split1', 'test_split2']:
                raw_dir = os.path.join(base_dir, category, split)
                if split == 'test':
                    raw_dir = os.path.join(base_dir, category, 'test_split1')
                elif split == 'test_split1':
                    raw_dir = os.path.join(base_dir, category, 'test_split1')
                elif split == 'test_split2':
                    raw_dir = os.path.join(base_dir, category, 'test_split2')

                preprocessed_dir = os.path.join(preprocessed_base, category, split)

                # Only process if the directory exists
                if os.path.exists(raw_dir):
                    print(f"Preprocessing {raw_dir}...")
                    preprocess_directory(raw_dir, preprocessed_dir, do_augmentation)
                else:
                    print(f"Directory does not exist: {raw_dir}")

if __name__ == "__main__":
    main()
    print("Data preprocessing completed.")

