#!/usr/bin/env python3
"""
transformer_embeddings_workflow.py

End-to-end example of:
1) Creating a Transformer-based embedding function in TensorFlow
2) Generating embeddings for supervised/unsupervised documents
3) Classification using scikit-learn
4) Clustering using scikit-learn
5) Tips & Best Practices (as comments)

Includes a progress bar (via tqdm) when encoding documents.
"""

import os
import glob
import json

import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from tqdm import tqdm  # NEW: for progress bars


# =============================================================================
# 3. Creating a Sentence Embedding Function
# =============================================================================

class TFEmbeddingExtractor:
    """
    A helper class to generate sentence/document embeddings in TensorFlow
    using Hugging Face transformer models. This approach mimics "Sentence-BERT"
    style embeddings by mean-pooling the last hidden states.

    Args:
        model_name (str): Hugging Face model identifier, e.g. 'distilbert-base-uncased'.
        max_length (int): Max token length for each document during tokenization.
        show_progress (bool): Whether to display a progress bar during encoding.
    """

    def __init__(self, model_name="distilbert-base-uncased", max_length=256, show_progress=True):
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # If model has no official TF weights, pass `from_pt=True` to convert from PyTorch
        self.model = TFAutoModel.from_pretrained(model_name, from_pt=True)
        self.max_length = max_length
        self.show_progress = show_progress

    def mean_pooling(self, last_hidden_states, attention_mask):
        """
        Performs mean pooling on the token embeddings, ignoring padded tokens.

        Args:
            last_hidden_states (tf.Tensor): shape (batch_size, seq_len, hidden_dim)
            attention_mask (tf.Tensor): shape (batch_size, seq_len)

        Returns:
            tf.Tensor of shape (batch_size, hidden_dim)
        """
        # Expand attention mask for broadcasting: (batch_size, seq_len, 1)
        mask = tf.cast(tf.expand_dims(attention_mask, axis=-1), dtype=tf.float32)

        # Multiply hidden states by the mask (0 for padded tokens, 1 for real tokens)
        token_embeddings = last_hidden_states * mask

        # Sum the embeddings across the sequence
        sum_embeddings = tf.reduce_sum(token_embeddings, axis=1)
        # Count how many tokens are actually real
        sum_mask = tf.reduce_sum(mask, axis=1)

        # Avoid division by zero by clipping the denominator
        embeddings = sum_embeddings / tf.clip_by_value(sum_mask, 1e-9, 1e9)
        return embeddings

    def encode(self, texts, batch_size=16):
        """
        Encode a list of texts and return sentence-level embeddings.

        Args:
            texts (List[str]): List of documents/sentences
            batch_size (int): Batch size during inference

        Returns:
            tf.Tensor of shape (len(texts), hidden_dim)
        """
        all_embeddings = []

        # We'll iterate over the data in batches, showing a progress bar if show_progress=True
        num_batches = (len(texts) + batch_size - 1) // batch_size
        if self.show_progress:
            print(f"Encoding {len(texts)} documents in {num_batches} batches...")
            batch_iter = tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch")
        else:
            batch_iter = range(0, len(texts), batch_size)

        for start_idx in batch_iter:
            batch_texts = texts[start_idx : start_idx + batch_size]
            # Tokenize
            encoding = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="tf"
            )
            outputs = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"]
            )
            # Last hidden states: (batch_size, seq_len, hidden_dim)
            last_hidden_states = outputs.last_hidden_state
            # Mean pooling
            embeddings = self.mean_pooling(last_hidden_states, encoding["attention_mask"])
            all_embeddings.append(embeddings)

        # Concatenate all batch embeddings into one tensor
        return tf.concat(all_embeddings, axis=0)


# =============================================================================
# 4. Generating Embeddings for Your Documents
# =============================================================================

def load_supervised_data(json_dir):
    """
    Loads text data from a directory with subfolders representing labels.
    Each subfolder has multiple .json files, each containing "text" and optional metadata.

    Directory structure example:
        json_dir/
            label1/
                file1.json
                file2.json
            label2/
                file3.json
                ...

    Args:
        json_dir (str): Path to the directory containing labeled subfolders.

    Returns:
        (List[str], List[str]): (texts, labels)
    """
    texts = []
    labels = []

    if not os.path.exists(json_dir):
        print(f"Directory does not exist: {json_dir}")
        return texts, labels

    for label_name in os.listdir(json_dir):
        full_label_dir = os.path.join(json_dir, label_name)
        if not os.path.isdir(full_label_dir):
            continue

        # label is the name of the subfolder
        for file_path in glob.glob(os.path.join(full_label_dir, "*.json")):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = data.get("text", "")
            texts.append(text)
            labels.append(label_name)
    return texts, labels


def load_unsupervised_data(json_dir):
    """
    Loads text data from a directory of .json files, no labels.

    Args:
        json_dir (str): Path to directory containing unlabeled .json files.

    Returns:
        List[str]: List of document texts.
    """
    texts = []

    if not os.path.exists(json_dir):
        print(f"Directory does not exist: {json_dir}")
        return texts

    for file_path in glob.glob(os.path.join(json_dir, "*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = data.get("text", "")
        texts.append(text)
    return texts


def generate_embeddings_example():
    """
    Demonstrates generating embeddings for supervised train/val sets
    and unsupervised data. Returns embeddings and labels as needed.
    """

    # Example paths -- change to your actual directories
    train_dir = "/home/davis/ml-portfolio/data/preprocessed/supervised/train"
    val_dir   = "/home/davis/ml-portfolio/data/preprocessed/supervised/val"
    unsup_dir = "/home/davis/ml-portfolio/data/preprocessed/unsupervised/train"

    # 1. Load supervised train data
    X_train_texts, y_train = load_supervised_data(train_dir)
    # 2. Load supervised val data
    X_val_texts, y_val = load_supervised_data(val_dir)
    # 3. Load unsupervised data
    X_unsup_texts = load_unsupervised_data(unsup_dir)

    # 4. Create embedding extractor (with progress bars enabled)
    embed_extractor = TFEmbeddingExtractor(
        model_name="distilbert-base-uncased",
        max_length=256,
        show_progress=True  # set to False if you want to hide progress bars
    )

    # 5. Generate embeddings
    print("Generating train embeddings...")
    X_train_embeddings = embed_extractor.encode(X_train_texts, batch_size=16)

    print("Generating val embeddings...")
    X_val_embeddings   = embed_extractor.encode(X_val_texts, batch_size=16)

    print("Generating unsupervised embeddings...")
    X_unsup_embeddings = embed_extractor.encode(X_unsup_texts, batch_size=16)

    print(f"Train embeddings shape: {X_train_embeddings.shape}")
    print(f"Val embeddings shape:   {X_val_embeddings.shape}")
    print(f"Unsupervised embeddings shape: {X_unsup_embeddings.shape}")

    return X_train_embeddings, y_train, X_val_embeddings, y_val, X_unsup_embeddings


# =============================================================================
# 5. Classification with Scikit-Learn
# =============================================================================

def classification_example(X_train_embeddings, y_train, X_val_embeddings, y_val):
    """
    Demonstrates classification using scikit-learn (RandomForest) on the embeddings.
    Prints out a classification report.
    """
    # Convert from TensorFlow tensors to NumPy arrays
    X_train_np = X_train_embeddings.numpy()
    X_val_np   = X_val_embeddings.numpy()

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_np, y_train)

    y_val_pred = clf.predict(X_val_np)
    print("\n=== Validation Classification Report ===")
    print(classification_report(y_val, y_val_pred))


# =============================================================================
# 6. Clustering with Scikit-Learn
# =============================================================================

def clustering_example(X_unsup_embeddings, n_clusters=10):
    """
    Demonstrates K-Means clustering with scikit-learn on unlabeled embeddings.
    Prints out silhouette score.

    Args:
        X_unsup_embeddings (tf.Tensor): Unsupervised embeddings
        n_clusters (int): Number of clusters for K-Means
    """
    X_unsup_np = X_unsup_embeddings.numpy()

    print(f"\nRunning K-Means with n_clusters={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_unsup_np)

    cluster_labels = kmeans.labels_
    sil_score = silhouette_score(X_unsup_np, cluster_labels)
    print(f"Silhouette Score: {sil_score:.4f}")


# =============================================================================
# 7. Tips & Best Practices
# =============================================================================

def tips_and_best_practices():
    """
    Prints or logs tips and best practices as a reminder.
    """
    print("\n=== Tips & Best Practices ===\n")
    print("1. For large datasets, embed in small batches and optionally save embeddings to disk.")
    print("2. Consider alternative pooling strategies (e.g., CLS token, max pooling) for embeddings.")
    print("3. Hyperparameter-tune your classifiers/clustering. Try different n_clusters, n_estimators, etc.")
    print("4. Evaluate carefully with confusion matrices, silhouette scores, or domain-specific metrics.")
    print("5. If your dataset is domain-specific, consider fine-tuning the Transformer on your domain texts.")
    print("6. For real production, push these steps to AWS Glue or SageMaker for automation and scalability.")
    print("7. Watch memory usage for large text corpora. Use chunking or streaming as needed.")
    print("8. Use GPU (e.g., tensorflow-gpu) if available for faster embedding inference.")
    print("========================================")


# =============================================================================
# Main function to tie everything together
# =============================================================================

def main():
    # Step 4: Generate embeddings
    X_train_emb, y_train, X_val_emb, y_val, X_unsup_emb = generate_embeddings_example()

    # Step 5: Classification
    classification_example(X_train_emb, y_train, X_val_emb, y_val)

    # Step 6: Clustering
    clustering_example(X_unsup_emb, n_clusters=10)

    # Step 7: Tips & Best Practices
    tips_and_best_practices()


if __name__ == "__main__":
    main()

