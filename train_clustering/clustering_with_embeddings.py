#!/usr/bin/env python3

import os
import json
import shutil
import argparse
from typing import List
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_unsupervised_data(root_dir: str):
    """
    Recursively scans 'root_dir' for JSON files, returning two lists:
     - file_paths[]: Absolute paths to each JSON file
     - documents[]: Text extracted from each JSON
    """
    file_paths = []
    documents = []
    for subdir, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith(".json"):
                file_path = os.path.join(subdir, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    text = data.get("text", "").strip()
                    if text:
                        file_paths.append(file_path)
                        documents.append(text)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return file_paths, documents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unsupervised_dir",
        type=str,
        default="/opt/ml/input/data/training",
        help="Path to the preprocessed unsupervised data directory (JSON files)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model name for generating sentence embeddings."
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Number of clusters for K-Means."
    )
    parser.add_argument(
        "--cluster_map",
        type=str,
        default="",
        help="Optional JSON string mapping cluster_id to known label. E.g. '{\"0\":\"shipping_orders\"}'."
    )
    args = parser.parse_args()

    # 1. Load all JSON documents from the specified directory
    file_paths, documents = load_unsupervised_data(args.unsupervised_dir)
    print(f"Loaded {len(documents)} documents from {args.unsupervised_dir}")
    if not documents:
        print("No documents found. Exiting.")
        return

    # 2. Load the sentence-transformers model for embeddings
    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    # 3. Embed the documents
    print("Embedding documents...")
    embeddings = model.encode(documents, batch_size=16, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # 4. Perform K-Means clustering
    print(f"Clustering with K-Means into {args.n_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    # 5. Compute silhouette score (optional)
    try:
        sil = silhouette_score(embeddings, labels)
        print(f"Silhouette score: {sil:.4f}")
    except ValueError:
        print("Could not compute silhouette score (possibly only one cluster or not enough samples).")

    # 6. Print cluster distribution
    from collections import Counter
    dist = Counter(labels)
    print("Cluster distribution:", dist)

    # 7. Print a snippet for each cluster
    sample_doc = {}
    for i, label in enumerate(labels):
        if label not in sample_doc:
            # Show up to 200 chars of text
            sample_doc[label] = documents[i][:200]
    print("\nSample snippet per cluster:")
    for c, snippet in sample_doc.items():
        print(f"Cluster {c}: {snippet}...")

    # 8. Optionally map numeric cluster IDs to labels
    cluster_map = {}
    if args.cluster_map:
        # E.g.: --cluster_map '{"0": "shipping_orders", "1": "memo"}'
        try:
            cluster_map = json.loads(args.cluster_map)
            print("Loaded cluster map:", cluster_map)
        except json.JSONDecodeError as e:
            print(f"Error parsing cluster_map JSON: {e}")

    # 9. Write back "predicted_cluster" and optional "predicted_label"
    #    to each JSON file in-place
    for i, fp in enumerate(file_paths):
        cluster_id = labels[i]
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["predicted_cluster"] = int(cluster_id)
        if str(cluster_id) in cluster_map:
            data["predicted_label"] = cluster_map[str(cluster_id)]
        else:
            data["predicted_label"] = f"cluster_{cluster_id}"

        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    print("Re-labeling complete. Updated JSON files in-place.")

    # 10. (Optional) Copy updated files into /opt/ml/model/ so they become part
    #     of the final SageMaker training job artifacts (model.tar.gz).
    output_dir = "/opt/ml/model/final_clustered_data"
    os.makedirs(output_dir, exist_ok=True)

    for fp in file_paths:
        filename = os.path.basename(fp)
        shutil.copy(fp, os.path.join(output_dir, filename))

    print(f"Copied updated JSON files to {output_dir} for final artifact upload.")

if __name__ == "__main__":
    main()

