#!/usr/bin/env python3

import os
import json
import argparse
from typing import List
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_unsupervised_data(root_dir: str):
    """
    Returns two lists: file_paths[], documents[]
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
    parser.add_argument("--unsupervised_dir", type=str,
                        default="../data/preprocessed/unsupervised/train",
                        help="Path to preprocessed unsupervised data directory.")
    parser.add_argument("--model_name", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Model name for embeddings.")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Number of clusters.")
    parser.add_argument("--cluster_map", type=str, default="",
                        help="Optional JSON file mapping cluster_id to known label, e.g. '{\"0\":\"shipping_orders\"}'")
    args = parser.parse_args()

    file_paths, documents = load_unsupervised_data(args.unsupervised_dir)
    print(f"Loaded {len(documents)} documents from {args.unsupervised_dir}")
    if not documents:
        print("No documents found. Exiting.")
        return

    # Load model
    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    # Embed
    print("Embedding documents...")
    embeddings = model.encode(documents, batch_size=16, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # K-Means
    print(f"Clustering with K-Means into {args.n_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    # Silhouette
    try:
        sil = silhouette_score(embeddings, labels)
        print(f"Silhouette score: {sil:.4f}")
    except ValueError:
        print("Could not compute silhouette score.")

    # Show distribution
    from collections import Counter
    dist = Counter(labels)
    print("Cluster distribution:", dist)

    # Show snippet for each cluster
    sample_doc = {}
    for i, label in enumerate(labels):
        if label not in sample_doc:
            sample_doc[label] = documents[i][:200]
    print("\nSample snippet per cluster:")
    for c, snippet in sample_doc.items():
        print(f"Cluster {c}: {snippet}...")

    # Optional: load cluster->label map if provided
    cluster_map = {}
    if args.cluster_map:
        # e.g.  --cluster_map '{"0": "shipping_orders", "1": "memo"}'
        # Or you can use a file
        cluster_map = json.loads(args.cluster_map)
        print("Loaded cluster map:", cluster_map)

    # Re-label in-place
    for i, fp in enumerate(file_paths):
        cluster_id = labels[i]
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["predicted_cluster"] = int(cluster_id)
        if cluster_id in cluster_map:
            data["predicted_label"] = cluster_map[str(cluster_id)]
        else:
            data["predicted_label"] = f"cluster_{cluster_id}"

        # Overwrite file
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    print("Re-labeling complete. Updated JSON files in-place.")

if __name__ == "__main__":
    main()

