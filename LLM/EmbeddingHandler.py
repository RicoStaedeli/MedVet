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
        pdf_directory = "../assets/Documents/Dental PDF"
        json_directory = "../assets/Documents/Cases txt"
        
        logger.info(f"Generate RAG with Documents")
        # Initialize an empty string to hold the text
        complete_text = ""

        #iterate all documents in Directory
        for filename in os.listdir(pdf_directory):
            f = os.path.join(pdf_directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                with fitz.open(f) as doc:  # open document
                # Iterate through each page of the PDF
                    for page_num in range(len(doc)):
                        # Get the page
                        page = doc.load_page(page_num)
                        # Extract text from the page
                        text = page.get_text()
                        # Append the text of the current page to the complete text
                        complete_text += text
        
        #iterate all documents in Directory
        for filename in os.listdir(json_directory):
            f = os.path.join(json_directory, filename)
            print(f)
            # checking if it is a file
            if os.path.isfile(f):
                with open(f, 'r') as file:
                    data = file.read()
                    
        complete_text = text + data
        
        logger.info(f"Split Text with length: {len(complete_text)}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 250, chunk_overlap=0)
        docs = text_splitter.create_documents([complete_text])        
        
        ################################################################################################
        ######################################### Debug function ########################################
        ################################################################################################
        if 1 == 0: 
            with open('rag_text.txt', 'w', newline='') as file:
                number = 0
                for document in docs:
                    number = number +1
                    file.write(f"{number}: {document}\n")
            
            file.close()        
        ################################################################################################
        
        return docs
    
    def embedDocumentsAndSaveInVectoreStore(self,docs):
        vectordb = Chroma.from_documents(documents=docs, embedding=self.embeddings, persist_directory="chroma_db")
        return vectordb
    