#!/usr/bin/env python3

import os
import json
import argparse
import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# --------------------------
# 1. Helper: Load JSON & Extract text
# --------------------------
def load_random_subset(data_dir: str, sample_size: int):
    """
    Loads a random subset of JSON files from 'data_dir'. 
    Returns:
       texts[] : list of text from the chosen JSON files
    """
    # Collect all .json files
    all_json_files = []
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.lower().endswith(".json"):
                all_json_files.append(os.path.join(root, fname))

    # If fewer files than sample_size, just take them all
    if sample_size > len(all_json_files):
        sample_size = len(all_json_files)

    # Randomly pick 'sample_size' files
    random.shuffle(all_json_files)
    chosen_files = all_json_files[:sample_size]

    texts = []
    for fp in chosen_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            txt = data.get("text", "").strip()
            if txt:
                texts.append(txt)
        except Exception as e:
            print(f"Error reading {fp}: {e}")

    print(f"Loaded {len(texts)} documents from a sample of {sample_size} files.")
    return texts

# --------------------------
# 2. Manual K-Means to capture intermediate states
# --------------------------
def initialize_centroids(X, k, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(len(X), size=k, replace=False)
    return X[indices].copy()

def assign_clusters(X, centroids):
    # Compute distance of each point to each centroid
    # X: (N, 2) or (N, d)  centroids: (k, d)
    # Returns cluster labels (N,)
    dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)  # shape (N, k)
    labels = np.argmin(dists, axis=1)  # best centroid for each point
    return labels

def update_centroids(X, labels, k):
    # Recompute each centroid as mean of its cluster
    new_cents = []
    for c_id in range(k):
        cluster_points = X[labels == c_id]
        if len(cluster_points) > 0:
            new_cents.append(cluster_points.mean(axis=0))
        else:
            # If a cluster has no points assigned, pick a random point
            new_cents.append(X[random.randint(0, len(X)-1)])
    return np.vstack(new_cents)

def custom_kmeans(X, k=3, max_iters=10):
    """
    Manual K-Means capturing centroid movement each iteration.
    Returns frames: list of (centroids, labels) at each iteration
    """
    centroids = initialize_centroids(X, k)
    frames = []
    for it in range(max_iters):
        labels = assign_clusters(X, centroids)
        frames.append((centroids.copy(), labels.copy()))  # store state
        new_centroids = update_centroids(X, labels, k)

        # If centroids stop moving, we can stop
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    # Final assignment after last iteration
    labels = assign_clusters(X, centroids)
    frames.append((centroids.copy(), labels.copy()))
    return frames

# --------------------------
# 3. Main Script
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="my_local_data",
                        help="Path to the directory with JSON files.")
    parser.add_argument("--sample_size", type=int, default=200,
                        help="How many JSON files to sample from data_dir.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of clusters for K-Means.")
    parser.add_argument("--max_iters", type=int, default=10,
                        help="Max iterations for manual K-Means.")
    parser.add_argument("--output_gif", type=str, default="kmeans_animation.gif",
                        help="Output GIF filename.")
    args = parser.parse_args()

    # --- Step A: Load random subset of data ---
    texts = load_random_subset(args.data_dir, args.sample_size)
    if not texts:
        print("No text found. Exiting.")
        return

    # --- Step B: Encode with SentenceTransformer ---
    print("Loading SentenceTransformer (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding texts...")
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True)
    print("Embeddings shape:", embeddings.shape)

    # --- Step C: Reduce from e.g. 384D -> 2D with PCA ---
    print("Reducing to 2D for visualization...")
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(embeddings)  # shape (n_samples, 2)

    # --- Step D: Run manual K-Means capturing frames ---
    frames = custom_kmeans(X_2d, k=args.k, max_iters=args.max_iters)
    print(f"Collected {len(frames)} frames of K-Means iteration.")

    # --- Step E: Animate with matplotlib ---
    fig, ax = plt.subplots(figsize=(6,6))
    colors = ['red','green','blue','purple','orange','brown','pink','gray','olive','cyan']  # up to 10
    k = args.k

    def init():
        ax.set_xlim(X_2d[:,0].min()-1, X_2d[:,0].max()+1)
        ax.set_ylim(X_2d[:,1].min()-1, X_2d[:,1].max()+1)
        ax.set_title("K-Means on Embeddings (Iteration 0)")
        return []

    def update(frame_idx):
        ax.clear()
        centroids, labels = frames[frame_idx]
        # Plot each cluster
        for c_id in range(k):
            pts = X_2d[labels == c_id]
            ax.scatter(pts[:,0], pts[:,1], color=colors[c_id], s=10, alpha=0.7, label=f"Cluster {c_id}")

        # Plot centroids
        ax.scatter(centroids[:,0], centroids[:,1], color='black', s=80,
                   marker='X', edgecolors='white', linewidth=1, zorder=10)
        ax.set_title(f"K-Means Iteration: {frame_idx}")
        ax.set_xlim(X_2d[:,0].min()-1, X_2d[:,0].max()+1)
        ax.set_ylim(X_2d[:,1].min()-1, X_2d[:,1].max()+1)
        return []

    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, interval=1000, blit=False)

    # Save as GIF
    ani.save(args.output_gif, writer='pillow', fps=1)
    print(f"Animation saved to {args.output_gif}")

    # Alternatively, to save MP4 (requires ffmpeg installed), do:
    # ani.save("kmeans_animation.mp4", writer="ffmpeg", fps=1)

if __name__ == "__main__":
    main()

