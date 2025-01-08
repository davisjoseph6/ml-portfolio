#!/usr/bin/env python3

import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertConfig

def create_distilbert_model(num_labels: int, learning_rate: float = 3e-5):
    """
    Create a DistilBERT classification model with the specified number of labels.
    Compiles it with SparseCategoricalCrossentropy loss and Adam optimizer.
    """
    # Create config
    config = DistilBertConfig.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )
    # Load pre-trained DistilBERT with classification head
    model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        config=config
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

if __name__ == "__main__":
    # Quick test
    model = create_distilbert_model(num_labels=3)
    model.summary()

