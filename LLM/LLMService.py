from langchain.chains import RetrievalQA
import json
from LLM.ModelLoaderLLM import LlamaForCausalRAG
from LLM.EmbeddingHandler import EmbeddingHnadler
from LLM.RAGCreator import RAGCreator

from Database.DbWriter import DbWriter
from langchain_core.prompts import PromptTemplate


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
        

        ragCreator = RAGCreator()
        retriever = ragCreator.getRetriever()

        self.qaPipeline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True

        )
    
    def formatPrompt(self, user_message):
        prompt_template = PromptTemplate.from_template(
            "<s>[INST] <<SYS>> {system_prompt} <</SYS>> {user_message} [/INST]"
        )
        
        system_prompt = "You are an intelligent assistant designed to support veterinarians by providing detailed and specific responses related to veterinary medicine, including diagnosis, treatment, and client communication. Tailor your answers to the specific species and context of the inquiry, offering practical advice, and remind users to verify all medical information with official sources."
        formatted = prompt_template.format(system_prompt=system_prompt, user_message=user_message)
        return formatted
        
                
    def generate(self, prompt: str, agent_id: int):
        logger.info(f"Query: {prompt}\n")
        time_1 = time()
        prompt = self.formatPrompt(prompt)
        result = self.qaPipeline.invoke(prompt)
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