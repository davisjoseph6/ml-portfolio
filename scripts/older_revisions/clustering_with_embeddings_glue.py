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
import boto3
from io import BytesIO
import gzip

def load_unsupervised_data(s3_client, bucket: str, prefix: str):
    """
    Load JSON files from S3 and return file paths and documents.
    """
    file_paths = []
    documents = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.lower().endswith(".json"):
                try:
                    response = s3_client.get_object(Bucket=bucket, Key=key)
                    content = response['Body'].read().decode('utf-8')
                    data = json.loads(content)
                    text = data.get("text", "").strip()
                    if text:
                        file_paths.append(key)
                        documents.append(text)
                except Exception as e:
                    print(f"Error reading {key}: {e}")
    return file_paths, documents

def save_relabelled_data(s3_client, bucket: str, prefix: str, file_key: str, data: dict):
    """
    Save the updated JSON back to S3.
    """
    try:
        json_str = json.dumps(data, ensure_ascii=False, indent=4)
        s3_client.put_object(Bucket=bucket, Key=file_key, Body=json_str.encode('utf-8'))
        print(f"Updated and saved: {file_key}")
    except Exception as e:
        print(f"Error saving {file_key}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsupervised_bucket", type=str, required=True,
                        help="S3 bucket containing preprocessed unsupervised data.")
    parser.add_argument("--unsupervised_prefix", type=str, required=True,
                        help="S3 prefix for the unsupervised data, e.g., 'preprocessed/unsupervised/train/'.")
    parser.add_argument("--model_name", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Model name for embeddings.")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Number of clusters.")
    parser.add_argument("--cluster_map", type=str, default="",
                        help="Optional JSON string mapping cluster_id to known label, e.g., '{\"0\":\"shipping_orders\"}'")
    args = parser.parse_args()

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Load data from S3
    file_keys, documents = load_unsupervised_data(
        s3_client,
        bucket=args.unsupervised_bucket,
        prefix=args.unsupervised_prefix
    )
    print(f"Loaded {len(documents)} documents from s3://{args.unsupervised_bucket}/{args.unsupervised_prefix}")

    if not documents:
        print("No documents found. Exiting.")
        return

    # Load SentenceTransformer model
    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    # Embed documents
    print("Embedding documents...")
    embeddings = model.encode(documents, batch_size=16, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # K-Means Clustering
    print(f"Clustering with K-Means into {args.n_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    # Silhouette Score
    try:
        sil = silhouette_score(embeddings, labels)
        print(f"Silhouette score: {sil:.4f}")
    except ValueError:
        print("Could not compute silhouette score.")

    # Cluster Distribution
    from collections import Counter
    dist = Counter(labels)
    print("Cluster distribution:", dist)

    # Sample Snippets per Cluster
    sample_doc = {}
    for i, label in enumerate(labels):
        if label not in sample_doc:
            sample_doc[label] = documents[i][:200]
    print("\nSample snippet per cluster:")
    for c, snippet in sample_doc.items():
        print(f"Cluster {c}: {snippet}...")

    # Load Cluster Map if provided
    cluster_map = {}
    if args.cluster_map:
        cluster_map = json.loads(args.cluster_map)
        print("Loaded cluster map:", cluster_map)

    # Re-label and Save Back to S3
    for i, file_key in enumerate(tqdm(file_keys, desc="Re-labeling and saving")):
        cluster_id = labels[i]
        try:
            # Retrieve the existing data
            response = s3_client.get_object(Bucket=args.unsupervised_bucket, Key=file_key)
            content = response['Body'].read().decode('utf-8')
            data = json.loads(content)

            # Update with cluster info
            data["predicted_cluster"] = int(cluster_id)
            if str(cluster_id) in cluster_map:
                data["predicted_label"] = cluster_map[str(cluster_id)]
            else:
                data["predicted_label"] = f"cluster_{cluster_id}"

            # Save back to S3
            save_relabelled_data(s3_client, args.unsupervised_bucket, args.unsupervised_prefix, file_key, data)
        except Exception as e:
            print(f"Error processing {file_key}: {e}")

    print("Re-labeling complete. Updated JSON files in S3.")

if __name__ == "__main__":
    main()

