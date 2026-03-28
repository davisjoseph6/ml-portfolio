# Chat-Secure / ML Portfolio

Document understanding pipeline combining:

- **document classification**
- **document summarization**
- **retrieval-augmented generation (RAG)**
- **unsupervised clustering**
- **local demo UI**
- **AWS SageMaker deployment**

This repository contains experiments and deployment scripts for processing documents such as **PDF**, **TXT**, and **images** (`.jpg`, `.jpeg`, `.png`) using OCR, transformer models, FAISS-based retrieval, and SageMaker endpoints.

## Project status

This repository contains:

- training and evaluation scripts for a **DistilBERT-based document classifier**
- preprocessing scripts for **text extraction + OCR + text augmentation**
- scripts for **summarization model training and inference**
- a **RAG pipeline** built with **SentenceTransformers + FAISS + seq2seq generation**
- **unsupervised clustering** workflows using document embeddings and K-Means
- **local FastAPI demo server** with HTML pages for classification, summarization, RAG, and clustering demos
- **AWS SageMaker deployment** utilities for classification, summarization, clustering, and RAG endpoints

## Notes

- Large artifacts such as trained models, datasets, and generated outputs are intentionally not fully tracked in Git.
- Several scripts use **hardcoded local paths**, **bucket names**, **endpoint names**, and **AWS account-specific values**. These should be updated before reuse.
- Some folders contain **older experiments / legacy revisions** kept for reference.

## Demo / presentation links

- Presentation: https://1drv.ms/p/s!Ahdv8SmoLX9Im9Vy9x9SF9qsmu7iGg?e=GNdrUM
- Unsupervised learning on AWS: https://www.youtube.com/watch?v=CL0MTOYx5Hk
- Full presentation: https://www.youtube.com/watch?v=HhLJSgnuF1w

---

## Main capabilities

### 1. Document preprocessing
The preprocessing pipeline supports:

- **PDF text extraction** with **PyMuPDF**
- **OCR on images** with **Tesseract**
- direct reading of **TXT** files
- export to structured **JSON**
- optional **text augmentation** using synonym replacement via `nlpaug`

Main script:
- `scripts/preprocessing/preprocess_data.py`

NLTK resources for augmentation:
- `scripts/preprocessing/nltk_script.py`

### 2. Document classification
The classification workflow is built around a DistilBERT-based model and includes:

- training
- evaluation
- inference examples
- SageMaker deployment utilities

Main scripts:
- `scripts/data_prep.py`
- `scripts/model.py`
- `scripts/train_distilbert.py`
- `scripts/evaluate.py`
- `scripts/evaluate_gpu.py`
- `scripts/usage.py`
- `scripts/usage_file.py`
- `scripts/deploy_model.py`
- `scripts/delete_endpoint.py`

### 3. Summarization
The summarization workflow includes:

- pseudo-summary generation
- summarizer training
- endpoint registration and deployment
- local and SageMaker inference tests

Main scripts:
- `scripts/generate_pseudo_summaries.py`
- `scripts/train_summarizer.py`
- `scripts/usage_file_summarizer.py`
- `scripts/register_summarization_model.py`
- `scripts/deploy_from_registry.py`
- `scripts/create_endpoint.py`
- `scripts/test_summar_endpoint.py`

### 4. Retrieval-Augmented Generation (RAG)
The RAG pipeline includes:

- chunking long documents
- summarizing chunks
- building a **FAISS** index from summarized data
- querying the index with **SentenceTransformers**
- generating answers using a seq2seq model

Main scripts:
- `scripts/rag/chunk_and_summarize.py`
- `scripts/rag/build_rag_index_multi.py`
- `scripts/rag/rag_query.py`

### 5. Unsupervised clustering
The unsupervised workflow uses document embeddings and **K-Means** to group unlabeled documents.

It includes:

- local clustering scripts
- SageMaker training job launch scripts
- 2D / 3D clustering visualizations
- Blender-related experimental animation files

Main scripts:
- `scripts/clustering_with_embeddings.py`
- `train_clustering/clustering_with_embeddings.py`
- `train_clustering/run_clustering_job.py`
- `train_clustering/kmeans_animation_local.py`
- `train_clustering/kmeans_3d.py`

### 6. Local web demo
A local FastAPI server exposes a small UI for:

- classification
- summarization
- RAG querying
- unsupervised demo pages

Main files:
- `my_local_server/server.py`
- `my_local_server/server_2.py`
- `my_local_server/static/`

### 7. RAG deployment on SageMaker
The repository also contains a containerized RAG inference service for SageMaker.

Main files:
- `rag_deployment/app.py`
- `rag_deployment/Dockerfile`
- `rag_deployment/Dockerfile.minimal`
- `rag_deployment/serve.sh`
- `rag_deployment/deploy_rag_model.py`
- `rag_deployment/deploy_rag_endpoint.py`
- `rag_deployment/test_rag_invoke.py`

---

## Repository structure

```bash
ml-portfolio/
├── README.md
├── buildspec.yml
├── requirements.txt
├── requirements2.txt
├── dependencies.zip
├── model.tar.gz
├── final_clustered_data/
├── aws_config/
├── policies/
│   ├── client-data-in-policy.json
│   ├── client-data-out-policy.json
│   ├── client-data-test-1-policy.json
│   ├── client-data-test-2-policy.json
│   ├── my_s3_custom_policy.json
│   └── retrieved-client-data-in-policy.json
├── scripts/
│   ├── clustering_with_embeddings.py
│   ├── create_endpoint.py
│   ├── data_prep.py
│   ├── delete_endpoint.py
│   ├── deploy_from_registry.py
│   ├── deploy_model.py
│   ├── evaluate.py
│   ├── evaluate_gpu.py
│   ├── generate_pseudo_summaries.py
│   ├── model.py
│   ├── register_summarization_model.py
│   ├── test_summar_endpoint.py
│   ├── train_distilbert.py
│   ├── train_summarizer.py
│   ├── usage.py
│   ├── usage_file.py
│   ├── usage_file_summarizer.py
│   ├── preprocessing/
│   │   ├── nltk_script.py
│   │   └── preprocess_data.py
│   ├── rag/
│   │   ├── build_rag_index_multi.py
│   │   ├── chunk_and_summarize.py
│   │   ├── faiss_index.bin
│   │   ├── index_metadata.json
│   │   └── rag_query.py
│   ├── older_revisions/
│   └── other_scripts/
├── train_clustering/
│   ├── blender/
│   ├── clustering_with_embeddings.py
│   ├── kmeans_3d.py
│   ├── kmeans_animation_local.py
│   ├── my_3d_animation.gif
│   ├── my_animation.gif
│   ├── requirements.txt
│   └── run_clustering_job.py
├── rag_deployment/
│   ├── app.py
│   ├── deploy_rag_endpoint.py
│   ├── deploy_rag_model.py
│   ├── Dockerfile
│   ├── Dockerfile.minimal
│   ├── faiss_index.bin
│   ├── index_metadata.json
│   ├── older_scripts/
│   ├── requirements.txt
│   ├── serve.sh
│   └── test_rag_invoke.py
├── my_local_server/
│   ├── how_to_run.txt
│   ├── server.py
│   ├── server_2.py
│   └── static/
│       ├── classification.html
│       ├── index.html
│       ├── summarize_rag.html
│       └── unsupervised.html
├── tests/
│   ├── class_endpoint_test.py
│   └── test_tokenizer.py
└── .gitignore
