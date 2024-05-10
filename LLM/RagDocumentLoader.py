import os
from tqdm import tqdm
import time
from typing import List
import glob
from multiprocessing import Pool
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

#########################################################################################
### This code is inspired by https://github.com/13331112522/m-rag/blob/main/m-rag.py ####
#########################################################################################

class ImgLoader:
        def __init__(self, source_directory, file_path):
            self.file_path = file_path       
            self.source_directory = source_directory
        
        def img_parse(self, img_path):
            res = "Fake it till you make it!" #ToDo: implementation of LlaVa response to the image

            with open(self.source_directory+"/"+os.path.basename(img_path)+".txt", "a") as write_file:
                write_file.write("---"*10 + "\n\n")
                write_file.write(os.path.basename(img_path) + "\n\n")
                write_file.write(res)
                write_file.flush()
            print("Proceeding "+img_path)
 
        def load(self) -> List[Document]:
            try:
                self.img_parse(self.file_path)
                loader=TextLoader(self.source_directory+"/"+os.path.basename(self.file_path)+".txt")
                doc=loader.load()
            except Exception as e:
                # Add file_path to exception message       
                raise type(e)(f"{self.file_path}: {e}") from e
            return doc

class RagDocumentLoader():
    
    def __init__(self, config: dict) -> None:
        self.source_directory = config.get("RAG", {}).get("source_path")
        self.chunk_size = config.get("RAG", {}).get("chunk_size")
        self.chunk_overlap = config.get("RAG", {}).get("chunk_overlap")
        
    
    # Map file extensions to document loaders and their arguments
    LOADER_MAPPING = {
        ".csv": (CSVLoader, {}),
        ".doc": (UnstructuredWordDocumentLoader, {}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        ".enex": (EverNoteLoader, {}),
        ".epub": (UnstructuredEPubLoader, {}),
        ".html": (UnstructuredHTMLLoader, {}),
        ".md": (UnstructuredMarkdownLoader, {}),
        ".odt": (UnstructuredODTLoader, {}),
        ".pdf": (PyMuPDFLoader, {}),
        # ".pdf": (UnstructuredPDFLoader, {}),
        ".ppt": (UnstructuredPowerPointLoader, {}),
        ".pptx": (UnstructuredPowerPointLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf8"}),
        # ".mp4": (VideoLoader,{}),
        ".jpg": (ImgLoader,{}),
        ".png": (ImgLoader,{}),
        # Add more mappings for other file extensions and loaders as needed
    }

    def load_single_document(self, file_path: str) -> List[Document]:
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in self.LOADER_MAPPING:
            loader_class, loader_args = self.LOADER_MAPPING[ext]
            if(ext == ".jpg" or ext ==".png"):
                loader = loader_class(self.source_directory, file_path, **loader_args)
            else:
                loader = loader_class(file_path, **loader_args)
            
            return loader.load()

        raise ValueError(f"Unsupported file extension '{ext}'")

    def load_documents(self, source_dir: str, ignored_files: List[str] = []) -> List[Document]:
        """
        Loads all documents from the source documents directory, ignoring specified files
        """
        all_files = []
        for ext in self.LOADER_MAPPING:
            all_files.extend(
                glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
            )
        filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
        
        results = []        
        for file in filtered_files:
            print(file)
            docs = self.load_single_document(file)
            results.extend(docs)
        
        return results

    def process_documents(self, ignored_files: List[str] = []) -> List[Document]:
        """
        Load documents and split in chunks
        """
        print(f"Loading documents from {self.source_directory}")
        documents = self.load_documents(self.source_directory, ignored_files)
        if not documents:
            print("No new documents to load")
            exit(0)
        print(f"Loaded {len(documents)} new documents from {self.source_directory}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks of text (max. {self.chunk_size} tokens each)")
        
        ################################################################################################
        ######################################### Debug function ########################################
        ################################################################################################
        if 1 == 0: 
            with open('rag_text.txt', 'w', newline='') as file:
                number = 0
                for document in texts:
                    number = number +1
                    file.write(f"{number}: {document}\n")
            
            file.close()        
        ################################################################################################
        return texts
    
   

