from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

import fitz
import os
from time import time

#Utils
from Utils.logger import get_logger
from Utils.config import load_config
config = load_config("config/cfg.yaml")
logger = get_logger(__name__, config)

class EmbeddingHnadler:
    def __init__(self):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        use_GPU = config.get("acceleration", {}).get("useGPU")
        if(use_GPU):
            device = config.get("acceleration", {}).get("GPU")
        else:
            device = "cpu"
        
        model_kwargs = {"device": device}
        logger.info(f"Acceleration device is set to: {device}")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        
    
    def buildChunks(self):
        pdf_directory = "../assets/Documents"
        
        logger.info(f"Generate RAG with Documents")
        #iterate all documents in Fintuning Directory
        for filename in os.listdir(pdf_directory):
            f = os.path.join(pdf_directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                pdf_name = filename.split("\\.")[0]      
            with fitz.open(f) as doc:  # open document
                text = chr(12).join([page.get_text() for page in doc])
        
        logger.info(f"Split Text")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 250, chunk_overlap=0)
        docs = text_splitter.create_documents([text])        
        
        return docs
    
    def embedDocumentsAndSaveInVectoreStore(self,docs):
        vectordb = Chroma.from_documents(documents=docs, embedding=self.embeddings, persist_directory="chroma_db")
        return vectordb
    