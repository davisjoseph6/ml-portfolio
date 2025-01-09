# For the reviewer:

As of 9th January, 2025:

- Main scripts for preprocessing, supervised, unsupervised, summarization, and RAG are found in the folder called `scripts`. 
- Raw and preprocessed data, and the AI models have not been pushed due to github size limitations. 
- Supervised and unsupervised learning have been completed on preprocessed data. They have been successfully tested. I will post their test evaluation metrics soon. 
- Summarization and RAG scripts are currently under development and once I complete them, I will push them. Client interface and endpoints have also yet to be completed, but I will complete them once I finish backend.
- Once local development is complete, I will move the whole project on AWS for final deployment.
- I have already put in place a small infrastructure and working inference endpoints on AWS with the AWS service such as Glue, S3 Bucket, SageMaker. I will complete them during my final stages.   

## Below are just some personal notes from myself at earlier stages of my project. (you can ignore them).

Business intelligence

sudo apt update && sudo apt upgrade

sudo apt install python3 python3-venv python3-pip

python3 -m venv ~/project_env
source ~/project_env/bin/activate
pip install --upgrade pip

pip install --upgrade pip

pip install boto3 flask transformers fitz numpy pandas pytesseract opencv-python

pip install faiss-cpu
pip install faiss-gpu
python3 -c "import faiss; print(faiss.__version__)"

sudo apt install tesseract-ocr

python3 --version
aws s3 ls

pip install pymupdf

.
