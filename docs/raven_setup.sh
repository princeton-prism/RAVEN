#!/usr/bin/env bash

# This is required to activate conda environment
eval "$(conda shell.bash hook)"

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    conda create -n $CONDA_ENV python=3.10 -y
    conda activate $CONDA_ENV
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

python -m pip install --upgrade pip

conda activate $CONDA_ENV


pip install torch==2.8.0 torchvision==0.23.0 transformers==4.44.2 tokenizers==0.19.1
pip install vertexai==1.71.1 google-ai-generativelanguage==0.7.0 open-clip-torch==3.2.0 volcengine-python-sdk==4.0.23 qwen-vl-utils==0.0.14
pip install langchain==0.3.27 langchain-community==0.3.30 langchain-core==0.3.77 langchain-google-genai==2.1.12 langchain-huggingface==0.3.1 langchain-milvus==0.2.1 langchain-nvidia-ai-endpoints==0.3.18 langchain-ollama==0.3.10 langchain-openai==0.3.34 langchain-text-splitters==0.3.11 langgraph==0.4.0 langgraph-checkpoint==2.1.1 langgraph-prebuilt==0.1.8 langgraph-sdk==0.2.9 langsmith==0.4.31
pip install opencv-python==4.12.0.88 pillow==10.4.0  
pip install faiss-cpu==1.12.0  faiss-gpu==1.7.2  accelerate==0.33.0 
pip install sentence-transformers
pip install -e external/qqmm-package
pip install datasets==4.3.0