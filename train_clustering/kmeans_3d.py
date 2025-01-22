#!/usr/bin/env python3

import os
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# For 3D axes
from mpl_toolkits.mplot3d import Axes3D

def load_random_subset(data_dir: str, sample_size: int):
    """Pick random JSON files from data_dir and extract 'text'."""
    all_json_files = []
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.lower().endswith(".json"):
                all_json_files.append(os.path.join(root, fname))

    if not all_json_files:
        return []

    if sample_size > len(all_json_files):
        sample_size = len(all_json_files)

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

def initialize_centroids(X, k, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices].copy()

def assign_clusters(X, centroids):
    dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)  # shape: (N, k)
    labels = np.argmin(dists, axis=1)
    return labels

def update_centroids(X, labels, k):
    new_centroids = []
    for c_id in range(k):
        cluster_points = X[labels == c_id]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            # if no points assigned, pick random
            new_centroids.append(X[random.randint(0, len(X)-1)])
    return np.vstack(new_centroids)

def custom_kmeans(X, k=5, max_iters=10):
    """Manual K-Means loop capturing centroid movement each iteration."""
    centroids = initialize_centroids(X, k)
    frames = []

    for it in range(max_iters):
        labels = assign_clusters(X, centroids)
        frames.append((centroids.copy(), labels.copy()))
        new_centroids = update_centroids(X, labels, k)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    # final assignment
    labels = assign_clusters(X, centroids)
    frames.append((centroids.copy(), labels.copy()))
    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="my_local_data",
                        help="Directory with JSON files.")
    parser.add_argument("--sample_size", type=int, default=200,
                        help="How many JSON files to load.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of clusters.")
    parser.add_argument("--max_iters", type=int, default=10,
                        help="Max K-Means iterations.")
    parser.add_argument("--output_gif", type=str, default="kmeans_3d.gif",
                        help="Output GIF filename.")
    parser.add_argument("--rotate", action="store_true",
                        help="If set, rotate camera angle each frame.")
    args = parser.parse_args()

    # 1) Load texts
    texts = load_random_subset(args.data_dir, args.sample_size)
    if not texts:
        print("No text found. Exiting.")
        return

    # 2) Encode with sentence-transformers
    print("Loading model: all-MiniLM-L6-v2...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding texts...")
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True)
    print("Embeddings shape:", embeddings.shape)

    # 3) Reduce to 3D with PCA
    print("Reducing to 3D for visualization...")
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(embeddings)
    print("Reduced shape:", X_3d.shape)

    # 4) K-Means capturing frames
    frames = custom_kmeans(X_3d, k=args.k, max_iters=args.max_iters)
    print(f"Collected {len(frames)} frames of K-Means iteration.")

    # 5) Animate in 3D
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection='3d')

    colors = ['red','green','blue','purple','orange','brown','pink','gray','olive','cyan']
    k = args.k

    # Precompute axis limits
    x_min, x_max = X_3d[:,0].min()-1, X_3d[:,0].max()+1
    y_min, y_max = X_3d[:,1].min()-1, X_3d[:,1].max()+1
    z_min, z_max = X_3d[:,2].min()-1, X_3d[:,2].max()+1

    def init():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_title("K-Means in 3D (Iteration 0)")
        return []

    def update(frame_idx):
        ax.clear()
        centroids, labels = frames[frame_idx]

        # Plot each cluster
        for c_id in range(k):
            pts = X_3d[labels == c_id]
            ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                       color=colors[c_id], s=10, alpha=0.7)
        # Plot centroids
        ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2],
                   color='black', marker='X', s=80, edgecolors='white', linewidth=1)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_title(f"K-Means in 3D (Iteration: {frame_idx})")

        if args.rotate:
            # slowly rotate camera
            # e.g. rotate azim by 5 degrees each frame
            ax.view_init(elev=30, azim=frame_idx * 5)

        return []

    ani = FuncAnimation(fig, update, frames=len(frames),
                        init_func=init, interval=1200, blit=False)

    ani.save(args.output_gif, writer='pillow', fps=1)
    print(f"3D animation saved to {args.output_gif}")

if __name__ == "__main__":
    main()

