#!/usr/bin/env python3

import sagemaker
from sagemaker.huggingface import HuggingFace
import boto3
import time

def main():
    # ========== 1. SageMaker Setup ==========
    region = "eu-west-2"
    # Update with your actual SageMaker execution role ARN:
    role_arn = "arn:aws:iam::637423166046:role/service-role/AmazonSageMaker-ExecutionRole-20241216T172309"

    # Create a SageMaker session (handles interactions with S3, etc.)
    sagemaker_session = sagemaker.Session(
        boto3.session.Session(region_name=region)
    )

    # ========== 2. Define Hugging Face Estimator ==========
    # We pick a CPU-compatible version of PyTorch + Transformers
    huggingface_estimator = HuggingFace(
        entry_point="clustering_with_embeddings.py",
        source_dir=".",  # current folder that has clustering_with_embeddings.py
        dependencies=["min_requirements.txt"],
        base_job_name="embedding-clustering-job",
        role=role_arn,
        instance_count=1,
        instance_type="ml.p3.2xlarge",  # example GPU instance; switch to ml.m5.large for CPU
        py_version="py38",

        # This combination has CPU/GPU images available:
        pytorch_version="1.10",
        transformers_version="4.17",

        hyperparameters={
            "unsupervised_dir": "/opt/ml/input/data/training",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "n_clusters": 5
            # Optionally: "cluster_map": '{"0": "labelA", "1": "labelB"}'
        },

        # We specify a custom output path for final model.tar.gz artifacts
        output_path="s3://aym-client-data-out/clustered-output/"
    )

    # ========== 3. Specify S3 Input Data ==========
    # Where your JSON files are stored:
    train_data_uri = "s3://aym-client-data-in/data/preprocessed/unsupervised/train/"

    train_input = sagemaker.inputs.TrainingInput(
        s3_data=train_data_uri,
        distribution="FullyReplicated",
        content_type="application/json",  # optional
        input_mode="File"
    )

    # ========== 4. Launch the Training Job ==========
    job_name = f"embedding-clustering-{int(time.time())}"
    print(f"Launching training job: {job_name}")

    # Provide the input channel named "training"
    huggingface_estimator.fit({"training": train_input}, job_name=job_name, wait=True)

    print(f"Training job {job_name} completed!")

    # ========== 5. Check or Download Artifacts ==========
    # The final artifacts (model.tar.gz) will appear in:
    #   huggingface_estimator.model_data
    #
    # Specifically at:
    #   s3://aym-client-data-out/clustered-output/<job-name>/output/model.tar.gz
    #
    # Inside that tarball, you'll find /final_clustered_data with your updated JSON.
    print("Model data stored at:", huggingface_estimator.model_data)

if __name__ == "__main__":
    main()

