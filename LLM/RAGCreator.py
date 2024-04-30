from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


from LLM.RagDocumentLoader import RagDocumentLoader
from LLM.ModelLoaderLLM import LlamaForCausalRAG

#Utils
from Utils.logger import get_logger
from Utils.config import load_config
config = load_config("config/cfg.yaml")
logger = get_logger(__name__, config)

class RAGCreator:
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
    
    def getRetriever(self):
        logger.info(f"Start creating new vectorstore")
        texts = self.documentLoader.process_documents()
        vectordb = Chroma.from_documents(documents=texts, embedding=self.embeddings, persist_directory="Database/chroma_db_rag")
        logger.info(f"Ingestion complete")
        retriever = vectordb.as_retriever(search_kwargs={"k": self.target_source_chunks})
        return retriever