#MedVet
Bachelor Thesis

# Project Directory Structure

Here's an overview of the project's directory structure:

- **Project-root/**
  - **Assets/**: All assets for the project
    - **RAG Documents/**: Directory to store all documents used for the RAG system
      - **Cases/**: All cases scraped for RAG
      - **Docs/**: All other documents 
      - **Knowledge Base/**: Additional documents
  - **Database/**: Source code.
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
  
  
