# MedVet
This is the main repository of the developed prototype called MedVet. The solution represents a state of the art implementation of Large Language Models with LangChain and llama.cpp. 
The implemented RAG system is able to generate answers based on embeded documents. 
This repository contains the backend for the complete solution.

------
## Project Directory Structure

Here's an overview of the project's directory structure:

- **Project-root/**
  - **model/**: contains all available LLMs
  - **Assets/**: All assets for the project
    - **RAG Documents/**: Directory to store all documents used for the RAG system
      - **Cases/**: All cases scraped for RAG
      - **Docs/**: All other documents 
      - **Knowledge Base/**: Additional documents
  - **Databases/**: All databasses for MedVet
    - **chroma_db_rag/**: ChromaDB with embedded vectore store
    - **medvet_chat_db.sqlite**: SQLite database to store all conversations with MedVet
  - **Medvet/**: backend application of MedVet
    - **config/**: config files for application
    - **Data Scraping/**: Notebooks for the web scraping 
    - **Database/**: File to handle the database connections 
    - **LLM/**: Everything to load and infere with LLMs 
    - **MMLLM/**: Implementation of multimodal model capabilities and model service 
    - **RequestModels/**: RequestModels for endpoints  
    - **Utils/**: Utils for application 
    - `.gitignore`
    - `app.py`: backend application with FastAPI
    - `README.md`
  - `requirements.txt`
  
## Installation of MedVet

Create a new virtual environment
```bash
  # Create venv
  python3 -m venv MedVetEnv
  cd MedVetEnv
  # activate env
  source bin/activate
  # Move to directory and clone repository
  mkdir Project
  cd Project
  git clone https://github.com/RicoStaedeli/MedVet.git
  cd MedVet
```


On Mac with Apple Silicon
```bash
  # Install requirements
  pip install -r requirements.txt

  # Install torch
  pip3 uninstall torch torchvision
  pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

  # Install distutils
  python3 -m pip install setuptools

  # Install llama-cpp-python for Metal
  CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

On Linux with NVIDIA GPU
```bash
  # Install requirements
  pip install -r requirements.txt

  # Install llama-cpp-python for cuBLAS
  CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

```
  
After these installation the additional files can be downloaded from Google drive: https://drive.google.com/drive/folders/1U04872Wu4TSKD3aL4hRy9mQPxuX0dqeQ

Next: To use the multimodal model LLaVA-Med the installation setup of this repo has to be done: https://github.com/RicoStaedeli/llava-medvet.git

## Run 
Navigate to the MedVet folder 

```bash
  uvicorn app:app
```

The application start at localhost `127.0.0.1:8000`
