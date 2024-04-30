from langchain.chains import RetrievalQA
import json
from LLM.ModelLoaderLLM import LlamaForCausalRAG
from LLM.EmbeddingHandler import EmbeddingHnadler
from LLM.RAGCreator import RAGCreator

from Database.DbWriter import DbWriter

#Utils
from Utils.logger import get_logger
from Utils.config import load_config
config = load_config("config/cfg.yaml")
logger = get_logger(__name__, config)

from time import time


class LLMService:
    
    def __init__(self,config: dict) -> None:
        llamaModel = LlamaForCausalRAG(config,logger)
        self.databaseWriter = DbWriter(config)
        
        model_path = config.get("model", {}).get("path")
        llm = llamaModel.load_llm(model_path)
        
        # embeddingHandler = EmbeddingHnadler()
        # chunks = embeddingHandler.buildChunks()
        # vectordb = embeddingHandler.embedDocumentsAndSaveInVectoreStore(chunks)
        
        # retriever = vectordb.as_retriever()
        ragCreator = RAGCreator()
        retriever = ragCreator.getRetriever()

        self.qaPipeline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True

        )
                
    def generate(self, prompt: str, agent_id: int):
        logger.info(f"Query: {prompt}\n")
        time_1 = time()
        result = self.qaPipeline.invoke(prompt)
        print(result)
        answer, docs = result['result'], [] if False else result['source_documents']
        time_2 = time()
        logger.info(f"Inference time: {round(time_2-time_1, 3)} sec.")
        logger.info("\nResult: ", result)
        result = json.dumps(answer)
        documents = []
        for document in docs:
            doc = {
                "source":document.metadata["source"],
                "content":document.page_content
            }
            documents.append(doc)
        self.databaseWriter.insert_chat(prompt_user=prompt, agent_id=agent_id, answer_assistant=result)
        return result,documents