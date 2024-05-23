from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


from LLM.RagDocumentLoader import RagDocumentLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import os
import sys
import shutil
#Utils
from Utils.logger import get_logger
from Utils.config import load_config
config = load_config("config/cfg.yaml")
logger = get_logger(__name__, config)

class RAGCreator:
    '''
    The RAG Creator is the heart of the RAG system. This class embedds all the created test chunks and stores them in a vector database. 
    '''
    def __init__(self):
        model_name = config.get("RAG", {}).get("EMBEDDINGS_MODEL_NAME")
        use_GPU = config.get("acceleration", {}).get("useGPU")
        self.source_directory = config.get("RAG", {}).get("source_path")
        self.target_source_chunks = config.get("RAG", {}).get("TARGET_SOURCE_CHUNKS")
        
        if(use_GPU):
            device = config.get("acceleration", {}).get("GPU")
        else:
            device = "cpu"
        
        model_kwargs = {"device": device}
        logger.info(f"Acceleration device is set to: {device}")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs) 
        self.documentLoader = RagDocumentLoader(config)
        # create the open-source embedding function
        self.embedding_function = SentenceTransformerEmbeddings(model_name=model_name)

    
    def getRetriever(self):
        if  os.path.isdir('Database/chroma_db_rag'):

            vectordb = Chroma(persist_directory="Database/chroma_db_rag", embedding_function=self.embedding_function)
            retriever = vectordb.as_retriever(search_kwargs={"k": self.target_source_chunks})
            return retriever
        else:
            logger.info(f"Start creating new vectorstore")
            texts = self.documentLoader.process_documents()
            vectordb = Chroma.from_documents(documents=texts, embedding=self.embeddings, persist_directory="Database/chroma_db_rag")
            logger.info(f"Ingestion complete")
            retriever = vectordb.as_retriever(search_kwargs={"k": self.target_source_chunks})
            return retriever