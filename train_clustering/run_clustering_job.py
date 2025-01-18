#!/usr/bin/env python3

import sagemaker
from sagemaker.huggingface import HuggingFace
import boto3
import time

def main():
    # ========== 1. SageMaker Setup ==========
    # Replace <ACCOUNT_ID> and <YourSageMakerExecutionRole> with your actual values.
    region = "eu-west-2"
    role_arn = "arn:aws:iam::637423166046:role/service-role/AmazonSageMaker-ExecutionRole-20241216T172309"

    # Create a SageMaker session (handles interactions with S3, etc.)
    sagemaker_session = sagemaker.Session(boto3.session.Session(region_name=region))

    # ========== 2. Define Hugging Face Estimator ==========
    # We pick a known CPU-supported version of PyTorch + Transformers
    huggingface_estimator = HuggingFace(
        entry_point="clustering_with_embeddings.py",
        source_dir=".",  # current folder that has clustering_with_embeddings.py
        base_job_name="embedding-clustering-job",
        role=role_arn,
        instance_count=1,
        instance_type="ml.p3.2xlarge",  # CPU
        py_version="py38",
        
        # This combination has CPU images available in the HF DLC:
        pytorch_version="1.10",
        transformers_version="4.17",
        
        hyperparameters={
            "unsupervised_dir": "/opt/ml/input/data/training", 
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "n_clusters": 5
            # Optionally: "cluster_map": '{"0": "labelA", "1": "labelB"}'
        }
    )

    # ========== 3. Specify S3 Input Data ==========
    # Adjust this S3 path to point to your unsupervised data directory
    # containing JSON files.
    train_data_uri = "s3://aym-client-data-in/unsupervised/train/"

    train_input = sagemaker.inputs.TrainingInput(
        s3_data=train_data_uri,
        distribution="FullyReplicated",
        content_type="application/json",  # optional
        input_mode="File"
    )

    # ========== 4. Launch the Training Job ==========
    job_name = f"embedding-clustering-{int(time.time())}"
    print(f"Launching training job: {job_name}")

    huggingface_estimator.fit({"training": train_input}, job_name=job_name, wait=True)

    print(f"Training job {job_name} completed!")

    # ========== 5. Check or Download Artifacts ==========
    # The final artifacts (model.tar.gz) will appear in:
    #   huggingface_estimator.model_data
    # or typically: s3://<default-bucket>/<job-name>/output/model.tar.gz
    #
    # Inside that tarball, you'll find /final_clustered_data with your updated JSON.

if __name__ == "__main__":
    main()

