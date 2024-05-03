import json
import requests

from LLM.ModelLoaderLLM import LlamaForCausalRAG
from LLM.RAGCreator import RAGCreator
from Database.DbWriter import DbWriter

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA



#Utils
from Utils.constants import MODE_ASSISTANT, MODE_DISPLAY, MODE_RAG
from Utils.conversation import (default_conversation, conv_templates)
from Utils.logger import get_logger
from Utils.config import load_config
config = load_config("config/cfg.yaml")
logger = get_logger(__name__, config)

from time import time


class MMLLMService:
    
    def __init__(self,config: dict) -> None:
        llamaModel = LlamaForCausalRAG(config,logger)
        self.databaseWriter = DbWriter(config)
        
        model_path = config.get("model", {}).get("path")
        llm = llamaModel.load_llm(model_path)
        

        ragCreator = RAGCreator()
        retriever = ragCreator.getRetriever()
        self.conversation = default_conversation.copy()

        self.qaPipeline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True

        )
        
    def clear_history(self):
        self.conversation = default_conversation.copy() 
       
    def set_assistantMode(self, assistant_mode):
        if assistant_mode == MODE_ASSISTANT.KB:
            self.conversation = conv_templates['simple_kb'].copy()
        else:
            self.conversation = default_conversation.copy() 
            
            
    
    
    def formatPromptLlaMACombined(self, user_message, img_desc):       
        prompt_template = PromptTemplate.from_template(
            "<s>[INST] <<SYS>> {system_prompt} <</SYS>> {user_message} \n Image Description: {img_desc}[/INST]"
        )
        
        system_prompt = "You are an intelligent assistant designed to support veterinarians by providing detailed and specific responses related to veterinary medicine, including diagnosis and treatment. Tailor your answers to the specific species and context of the inquiry, offering practical advice, and remind users to verify all medical information with official sources."
        formatted = prompt_template.format(system_prompt=system_prompt, user_message=user_message, img_desc= img_desc)
        return formatted
    
    
    def formatPromptLlaMA(self, user_message):
        self.conversation.append_message(role=self.conversation.roles[0],message=user_message)
        
        print(f"Conversation: {self.conversation.get_prompt()}")

        formatted = self.conversation.get_prompt()
        return formatted
    
    
    def formatPromptLlaVA(self, prompt):
        prompt_template = PromptTemplate.from_template(
            "Case Description: {prompt} \n\n Describe what you see in the image and what you interpret with this description"
        )
        
        formatted = prompt_template.format(prompt=prompt)
        return formatted
        
    
    def generateLLaVAresponse(self, ip_address_llava, prompt_llava, temperature, max_new_tokens, image):
        url = f"http://{ip_address_llava}/generate"
        data = {
                "prompt": prompt_llava,
                "image": image,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "agent_id":""
            }

        headers = {'Content-type': 'application/json', 'Accept': 'application/json'}        
        try:
            resp = requests.put(url, data=json.dumps(data), headers=headers)
            output = {
                "result":resp.json()['result'],
                "prompt":prompt_llava,
                "status":"OK"
            }
        except Exception as e:
            logger.info(f"Failed to connect to AWS:{e}")
            output = {
                "result":"",
                "status":f"Failed to connect to AWS:{e}"
            }
        
        return output
    
    def generateLLaMAresponse(self, prompt_llama, use_rag ):
        documents = []
        result = ""
        try:
            time_1 = time()
            result = self.qaPipeline.invoke(prompt_llama)
            answer, docs = result['result'], [] if False else result['source_documents']
            time_2 = time()
            logger.info(f"Inference time: {round(time_2-time_1, 3)} sec.")
            logger.info("\nResult: ", result)
            result = answer #json.dumps(answer)
            logger.info(f"found {len(docs)} Documents")
            for document in docs:
                print(document)
                doc = {
                    "source":document.metadata["source"],
                    "content":document.page_content,
                    "page":document.metadata["page"]
                }
                documents.append(doc)
            
            output = {
                "result": result,
                "prompt": prompt_llama,
                "documents": documents,
                "status":"OK"
            }
        except Exception as e:
            logger.info(f"Failed to generate answer with LlaMA")
            print(f"Failed LlaMA {e}")
            output = {
                "result":result,
                "documents":documents,
                "prompt":"",
                "status":f"Failed"
            }
        
        return output
    
    def generateDocumentsmarkdown(self, documents):
        response = ""
        for document in documents:
            response = response + f"- {document['source']}  \n  \n"
        return response
    
    
    def generateAnswer(self, 
                       agent_id, 
                       prompt_user, 
                       display_combined:str, 
                       mode_assistant:str, 
                       use_rag:str, 
                       image:str, 
                       ip_address_llava:str, 
                       max_new_tokens, 
                       temperature):
        
        #No image --> use just LlaMA pipeline
        if prompt_user != "" and image == "":
            print(f"Generate answer only with LlaMA")
            try:
                prompt_llama = self.formatPromptLlaMA(prompt_user)
                response_llama = self.generateLLaMAresponse(prompt_llama=prompt_llama,use_rag=use_rag)
                self.conversation.append_message(role=self.conversation.roles[1],message=response_llama['result'])
                self.databaseWriter.insert_chat( agent_id=agent_id, prompt_user=prompt_user, prompt_llava="", prompt_llama=prompt_llama, answer_llava="", answer_llama=response_llama["result"], answer_combined = "", image = "", mode_display="", mode_assistant=mode_assistant, mode_rag=use_rag)

                status = "OK"
                
                #format in streamlit markdown
                response = f"**Result LlaMA and RAG**\n\n{response_llama['result']}\n\n**Sources**\n\n{self.generateDocumentsmarkdown(response_llama['documents'])}"

            except Exception as e:
                logger.info(f"Failed to generate Answer {e}")
                response = "Failed to generate Answer"
                response_llama = ""
                status = "Failed"
                 
            return {"status":status, "response":response, "answer_llama": response_llama}
            
            
        #because image isn't empty --> enter multimodal inference
        elif prompt_user != "" and image != "" and display_combined == MODE_DISPLAY.SEPARATE:
            print(f"Start generating Answer with LLaMA and LLaVa, display separate and with RAG")
            try:
                prompt_llama = self.formatPromptLlaMA(prompt_user)
                response_llama = self.generateLLaMAresponse(prompt_llama=prompt_llama,use_rag=use_rag)
                self.conversation.append_message(role=self.conversation.roles[1],message=response_llama['result'])
                
                prompt_llava = self.formatPromptLlaVA(prompt_user)
                response_llava = self.generateLLaVAresponse(image=image, ip_address_llava=ip_address_llava , max_new_tokens=max_new_tokens, prompt_llava=prompt_llava, temperature=temperature)
                
                self.databaseWriter.insert_chat( agent_id=agent_id, prompt_user=prompt_user, prompt_llava=prompt_llava, prompt_llama=prompt_llama, answer_llava=response_llava["result"], answer_llama=response_llama["result"], answer_combined = "", image = image, mode_display=display_combined, mode_assistant=mode_assistant, mode_rag=use_rag)
                
                response = f"**Result LlaMA and RAG**\n\n{response_llama['result']}\n\n**Sources**\n\n{self.generateDocumentsmarkdown(response_llama['documents'])}\n\n**Result LlaVA- Med**\n\n{response_llava["result"]}"
                status = "OK"
            except: 
                response = "Failed to generate Answer"
                response_llama = ""
                response_llava = ""
                status = "Failed"
            
            return {"status":status, "response":response, "answer_llama": response_llama, "answer_llava": response_llava}
        
        elif prompt_user != "" and image != "" and display_combined == MODE_DISPLAY.COMBINED:
            print(f"Generate combined answer")
            try:
                
                prompt_llava = self.formatPromptLlaVA(prompt_user)
                response_llava = self.generateLLaVAresponse(image=image, ip_address_llava=ip_address_llava , max_new_tokens=max_new_tokens, prompt_llava=prompt_llava, temperature=temperature)
                
                prompt_llama = self.formatPromptLlaMACombined(prompt_user,img_desc=prompt_llava)
                response_llama = self.generateLLaMAresponse(prompt_llama=prompt_llama,use_rag=use_rag)
                self.conversation.append_message(role=self.conversation.roles[1],message=response_llama['result'])
                
                
                self.databaseWriter.insert_chat( agent_id=agent_id, prompt_user=prompt_user, prompt_llava=prompt_llava, prompt_llama=prompt_llama, answer_llava=response_llava["result"], answer_llama=response_llama["result"], answer_combined = response_llama["result"], image = image, mode_display=display_combined, mode_assistant=mode_assistant, mode_rag=use_rag)
                
                response = f"**Result combination of LlaVA- MEd and LlaMA with RAG**\n\n{response_llama['result']}\n\n**Sources**\n\n{self.generateDocumentsmarkdown(response_llama['documents'])}"
                status = "OK"
            
            except: 
                response = "Failed to generate Answer"
                response_llama = ""
                response_llava = ""
                status = "Failed"
            
            return {"status":status, "response":response, "answer_llama": response_llama, "answer_llava": response_llava}
        
        
        else:
            print(f"No valid configuration")        