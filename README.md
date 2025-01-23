# Secure Chat

Project status as of **23rd January, 2025** :

## Project overview:

Notes:
- Some directories and files (such as model artifacts, raw and preprocessed data) could not be uploaded due to github limitations on size.
- Main scripts for preprocessing, training, and RAG are found in the directory called `scripts/`
- Models are deployed on AWS SageMaker and AWS have been created
- Please scroll to the right in the directory structure below to view clipped information. 

```bash
# Project Structure

ml-portfolio/
├── README.md
├── distilbert_model/                   # DistilBERT classification model artifacts.
│   ├── config.json
│   ├── label_map.json
│   ├── special_tokens_map.json
│   ├── tf_model.h5
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── distilbert_model.tar.gz             # Tarred model for SageMaker deployment.
├── my_summarization_model/             # Fine-tuned summarization model artifacts (BART/DistilBART).
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── training_args.bin
│   └── vocab.json
├── my_summarization_model.tar.gz       # Summarization model tarball for SageMaker deployment.
├── model.tar.gz			# Unsupervised clustering SageMaker final artifacts (output JSON/model)
├── scripts/                            # Main folder for Python scripts (deploy, train, RAG, etc.).│   
│   ├── data_prep.py                    	# Prepares data for DistilBERT classification training (tokenization, etc.). 
│   ├── model.py                        	# DistilBERT model creation (TF). 
│   ├── train_distilbert.py             	# Script to train DistilBERT for classification tasks. 
│   ├── evaluate.py                     	# Evaluates classification model performance 
│   ├── usage.py                        	# Example usage of the classification model. 
│   ├── usage_file.py                   	# Example usage of classification on a raw file
│   ├── deploy_model.py                 	# Deploy local classification model tarball to a AWS SageMaker endpoint. 
│   ├── delete_endpoint.py              	# Deletes an existing classification AWS SageMaker endpoint. 
│   ├── preprocessing/                  	# Subfolder for data preprocessing scripts.
│   │   ├── nltk_script.py              		# NLTK-based text cleaning or tokenization example.
│   │   └── preprocess_data.py          		# OCR, text extraction, augmentation pipeline.
│   ├── clustering_with_embeddings.py   	# SentenceTransformer + KMeans clustering (local version). Unsupervised Learning
│   ├── generate_pseudo_summaries.py    	# Generates rough/pseudo summaries for text data.
│   ├── train_summarizer.py             	# Script to fine-tune summarization model.
│   └── usage_file_summarizer.py        	# Summarizer usage or inference test script.
│   ├── register_summarization_model.py 	# Registers summarization model to SageMaker Model Registry.
│   ├── deploy_from_registry.py         	# Deploy summarization model from Model Registry to an endpoint.
│   ├── create_endpoint.py              	# Creates Summarization SageMaker endpoint from model config.
│   ├── test_summar_endpoint.py         	# Tests a deployed summarization endpoint with sample input.
│   ├── rag/                            	# Retrieval-Augmented Generation pipeline code.
│   │   ├── chunk_and_summarize.py                      # Splits docs into chunks, and summarizes.
│   │   ├── build_rag_index_multi.py    		# Builds FAISS index from multiple summarized data dirs.
│   │   ├── faiss_index.bin             		# Example FAISS index (binary).
│   │   ├── index_metadata.json         		# Metadata for the FAISS index.
│   │   └── rag_query.py                		# Queries the RAG index + generative model for answers.
├── tests/                              # Holds test files (unit/integration tests).
│   └── test_tokenizer.py               	# Tests tokenizer logic or text processing.
├── train_clustering/                   # Dedicated folder for SageMaker clustering job scripts.
│   ├── clustering_with_embeddings.py   	# Similar KMeans script adapted for SageMaker TrainingJob.
│   ├── requirements.txt                	# Dependencies to install in the SageMaker training container.
│   └── run_clustering_job.py           	# Python driver to launch the clustering job on SageMaker.
├── rag_deployment/ 			# Deployment of RAG as a SageMaker endpoint
│   ├── app.py
│   ├── Dockerfile
│   ├── serve.sh
│   ├── deploy_rag_model.py
│   ├── deploy_rag_endpoint.py
│   └── test_rag_invoke.py
├── policies/                           # JSON policy files for controlling S3 or IAM permissions in AWS.
│   ├── client-data-in-policy.json
│   ├── client-data-out-policy.json
│   ├── client-data-test-1-policy.json
│   ├── client-data-test-2-policy.json
│   ├── my_s3_custom_policy.json
│   └── retrieved-client-data-in-policy.json
├── python/                             # Python scripts, wheels, or environment folder.
├── requirements.txt                    # Main Python dependencies for the project.
├── requirements2.txt                   # Secondary or alternative dependency file.
├── aws_config/
├── dependencies.zip                    # A zip of extra dependencies or modules.
├── my_local_server/			# Local host server
│   ├── server.py
│   ├── static/
│   │   └──index.html
├── data/                               # Primary data directory (raw, preprocessed, summarized).
│   ├── examples/                       	# Example documents or samples for demonstration.
│   ├── preprocessed/                   	# Data that has been partially or fully preprocessed.
│   ├── raw/                            	# Original, unchanged source data.
│   │   ├── supervised/                 		# Labeled data (supervised) in raw form.
│   │   │   ├── train/                  			# Training set for supervised tasks.
│   │   │   │   ├── advertisement/      				# Each subfolder is a class label/type of document.
│   │   │   │   ├── budget/
│   │   │   │   ├── email/
│   │   │   │   ├── form/
│   │   │   │   ├── handwritten/
│   │   │   │   ├── inventory_report/
│   │   │   │   ├── invoice/
│   │   │   │   ├── letter/
│   │   │   │   ├── memo/
│   │   │   │   ├── news_articles/
│   │   │   │   │   ├── news_business/
│   │   │   │   │   ├── news_entertainment/
│   │   │   │   │   ├── news_general/
│   │   │   │   │   └── news_sport/
│   │   │   │   ├── presentation/
│   │   │   │   ├── purchase_orders/
│   │   │   │   ├── questionnaire/
│   │   │   │   ├── resume/
│   │   │   │   ├── scientific_publication/
│   │   │   │   ├── scientific_report/
│   │   │   │   ├── shipping_orders/
│   │   │   │   └── specification/
│   │   │   ├── val/                    			# Validation set for supervised data.
│   │   │   ├── test/                   			# Test set for supervised data.
│   │   │   ├── test_split1/            			# Alternate or partial test set (split).
│   │   │   └── test_split2/            			# Another partial test set (split).
│   │   └── unsupervised/               		# Unlabeled data in raw form (for clustering, etc.).
│   │       ├── train/	                 			# Training set for unsupervised tasks.│   │       
│   │       ├── test/                   			# Test set (unsupervised).
│   │       ├── test_split1/
│   │       └── test_split2/
│   └── summarized/                     	# Data with generated summaries (post-processing).

```

## Author

Davis Joseph (C-21 France) for [Holberton School](https://www.holbertonschool.fr/)

- [LinkedIn](https://www.linkedin.com/in/davisjoseph767/)
- [GitHub](https://github.com/davisjoseph6)

