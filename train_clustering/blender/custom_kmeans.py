import numpy as np
import json
import random

def random_init_centroids(X, k, seed=42):
    """
    Pick k distinct points from X as centroids (randomly).
    """
    np.random.seed(seed)
    indices = np.random.choice(len(X), size=k, replace=False)
    centroids = X[indices].copy()
    return centroids

def assign_clusters(X, centroids):
    """
    For each point in X, find the nearest centroid.
    Return an array of cluster indices with shape (N,).
    """
    # X shape: (N, d); centroids shape: (k, d)
    # distances shape: (N, k)
    distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

def update_centroids(X, labels, k):
    """
    Compute the new centroid for each cluster (mean of all points assigned).
    If a cluster has no points, randomly re-initialize it.
    Return shape: (k, d)
    """
    d = X.shape[1]
    new_cents = np.zeros((k, d), dtype=X.dtype)
    for cluster_id in range(k):
        pts_in_cluster = X[labels == cluster_id]
        if len(pts_in_cluster) > 0:
            new_cents[cluster_id] = pts_in_cluster.mean(axis=0)
        else:
            # If no points assigned, pick a random point
            new_cents[cluster_id] = X[random.randint(0, len(X)-1)]
    return new_cents

def kmeans_with_logging(X, k=5, max_iter=10, tol=1e-4, seed=42):
    """
    Run a custom K-Means, capturing each iteration's cluster assignments
    and centroid positions. Return a list of frames for animation.

    frames = [
      {
        "iteration": 0,
        "points": [
          {"id": i, "x": X[i,0], "y": X[i,1], "cluster": labels[i], ... },
          ...
        ],
        "centroids": [
          {"id": c_id, "x": cx, "y": cy, ... },
          ...
        ]
      },
      ...
    ]
    """
    # 1) Initialize centroids
    centroids = random_init_centroids(X, k, seed=seed)

    frames = []
    for iteration in range(max_iter):
        # 2) Assign clusters
        labels = assign_clusters(X, centroids)

        # 3) Store the iteration info
        iteration_data = {
            "iteration": iteration,
            "points": [],
            "centroids": []
        }
        for i in range(len(X)):
            iteration_data["points"].append({
                "id": int(i),
                "x": float(X[i,0]),
                "y": float(X[i,1]),
                "cluster": int(labels[i])
            })
        for c_id in range(k):
            iteration_data["centroids"].append({
                "id": c_id,
                "x": float(centroids[c_id,0]),
                "y": float(centroids[c_id,1])
            })

        frames.append(iteration_data)

        # 4) Update centroids
        new_centroids = update_centroids(X, labels, k)

        # 5) Check for convergence
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            # No significant movement => done
            break

        centroids = new_centroids

    return frames

# ------------- MAIN DEMO -------------
if __name__ == "__main__":
    # Suppose you have X in 2D. For a real scenario, load or generate data:
    # Here we create random points for demonstration:
    np.random.seed(42)
    X_cluster0 = np.random.normal(loc=(0,0), scale=0.3, size=(20,2))
    X_cluster1 = np.random.normal(loc=(2,2), scale=0.3, size=(20,2))
    X_cluster2 = np.random.normal(loc=(1,4), scale=0.3, size=(20,2))
    X = np.vstack([X_cluster0, X_cluster1, X_cluster2])

    # Let's do k=3 for the demonstration
    frames_data = kmeans_with_logging(X, k=3, max_iter=10)

    # Write out as JSON
    output_json = "kmeans_data.json"
    with open(output_json, "w") as f:
        json.dump(frames_data, f, indent=2)

    print(f"Saved {len(frames_data)} frames to {output_json}")

