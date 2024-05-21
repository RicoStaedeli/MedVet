import json
import requests

from LLM.ModelLoaderLLM import LlamaForCausalRAG
from LLM.ModelLoaderLLMHF import LlamaForCausalRAGHF
from LLM.RAGCreator import RAGCreator
from Database.DbWriter import DbWriter

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage


from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda, RunnablePassthrough,RunnableParallel
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate

from operator import itemgetter


#Utils
from Utils.constants import MODE_DISPLAY, MODE_RAG
from Utils.conversation import (default_conversation, conv_templates)
from Utils.SystemPromptsMedVet import (systemprompt_templates)
from Utils.LLMTemplates import (llm_templates)
from Utils.logger import get_logger
from Utils.config import load_config
config = load_config("config/cfg.yaml")
logger = get_logger(__name__, config)
#template has to be set if the model changes

from time import time

from langsmith import traceable
# export LANGCHAIN_TRACING_V2=true
# export LANGCHAIN_API_KEY="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"

class MMLLMService:
    
    def __init__(self,config: dict) -> None:
        llamaModel = LlamaForCausalRAG(config,logger)
        # llamaModel = LlamaForCausalRAGHF(config,logger)
        self.databaseWriter = DbWriter(config)
        
        model_path = config.get("model", {}).get("path")
        self.llm = llamaModel.load_llm(model_path)
        self.llm_standalone = llamaModel.load_llm(model_path)
        
        ragCreator = RAGCreator()
        self.retriever = ragCreator.getRetriever() 

   
    def clear_history(self):
        # self.conversation = default_conversation.copy() 
        try:
            #self.memory.clear()
            print("If you want to delete uncoment above")
        except:
            logger.info("memory doesn't exist")
                      
    
     
    ################################################################################################################
    ##################### Functions Chains ##############
    ################################################################################################################   
    
    def getRelevantDocuments(self,question):
        """
        This function retrieves the most relevant Documents for the question.
        """
        docs = self.retriever.get_relevant_documents(question)
        print(f"Documents: {docs}")
        return docs
    
    def formatPromptLlaVA(self, prompt):
        prompt_template = PromptTemplate.from_template(
            "Case Description: {prompt} \n\n Describe what you see in the image and what you interpret with this description"
        )
        
        formatted = prompt_template.format(prompt=prompt)
        return formatted
    
    def _combine_documents(self,docs, document_separator="\n\n"):
        document_prompt = self.DEFAULT_DOCUMENT_PROMPT
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)
     
    def _sources_documents(self,docs):
        sources = []
        for document in docs:
            if "page" in document.metadata:
                page = document.metadata["page"]
            else:
                page = "unknown"
            sources.append({
                "source": document.metadata["source"], 
                "page": page
            })
        return sources
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    ################################################################################################################
    #####################                               Chians                                          #############
    ################################################################################################################   
    
    def buildChainForPlainInference(self,llm_template_name):
        '''
        This function creates a chain based on the answer prompt template. 
        No RAG. 
        '''
        
        ANSWER_PROMPT = ChatPromptTemplate.from_template(llm_templates[llm_template_name].getTemplate())
              
        answer_chain = RunnableParallel({
            "system_prompt": lambda x: x["system_prompt"],
            "answer": ANSWER_PROMPT | self.llm | StrOutputParser(),
            "question": lambda x: x["question"],
            "img_description": lambda x: x["img_description"],
        })
        
        return answer_chain
    

    def buildChainRAG(self,llm_template_name):
        '''
        This chain uses a RAG pipeline to retrieve the most relevant documents for the submitted input. 
        '''
        #This template is used to create a standalone question  outcommented: Do the rephrase only if there is a chat history. / If the user asks about something you wrote given the following conversation and a follow up question. 
        ANSWER_PROMPT = ChatPromptTemplate.from_template(llm_templates[llm_template_name].getTemplate())
        
        self.DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
        
  
        retrieved_documents = RunnableParallel({
            "system_prompt": lambda x: x["system_prompt"],
            "docs": itemgetter("question") | self.retriever,
            "question": lambda x: x["question"],
            "chat_history":lambda x: x["chat_history"],
            "img_description":lambda x: x["img_description"],
        })

        final_inputs = {
            "system_prompt": lambda x: x["system_prompt"],
            "context": lambda x: self._combine_documents(x["docs"]),
            "sources": lambda x: self._sources_documents(x["docs"]),
            "question": lambda x: x["question"],
            "img_description": lambda x: x["img_description"],
            "chat_history": lambda x: x["chat_history"],
        }

        answer_chain = {
            "system_prompt": lambda x: x["system_prompt"],
            "answer": final_inputs | ANSWER_PROMPT | self.llm | StrOutputParser(),
            "question": lambda x: x["question"],
            "context": final_inputs["context"],
            "sources": final_inputs["sources"],
            "img_description": final_inputs["img_description"]
        }
        
        final_chain = retrieved_documents | answer_chain
             
        return final_chain
    
    #ToDo use history    Legacy
    def buildChainForConversationalRAG(self,llm_template_name):
        '''
        THis chain creates a This chain uses a RAG pipeline to retrieve the most relevant documents for the submitted input. 
        Also it uses the conversion chat history. 
        '''
        #This template is used to create a standalone question  outcommented: Do the rephrase only if there is a chat history. / If the user asks about something you wrote given the following conversation and a follow up question. 
        ANSWER_PROMPT = ChatPromptTemplate.from_template(llm_templates[llm_template_name].getTemplate())
        
        self.DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
        
        # ConversationBufferMemory is a buffer for the messages from a conversation and a template from langchain
        self.memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")

        # With this we load the memory of the chat history
        # This adds a "memory" key to the input object
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history")
        )
    
        retrieved_documents = RunnableParallel({
            "system_prompt": lambda x: x["system_prompt"],
            "docs": itemgetter("question") | self.retriever,
            "question": RunnablePassthrough(),
            "img_description":lambda x: x["img_description"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        })

        final_inputs = {
            "system_prompt": lambda x: x["system_prompt"],
            "context": lambda x: self._combine_documents(x["docs"]),
            "sources": lambda x: self._sources_documents(x["docs"]),
            "question": itemgetter("question"),
            "img_description": lambda x: x["img_description"],
        }

        answer_chain = {
            "system_prompt": lambda x: x["system_prompt"],
            "answer": final_inputs | ANSWER_PROMPT | self.llm | StrOutputParser(),
            "question": itemgetter("question"),
            "context": final_inputs["context"],
            "sources": final_inputs["sources"],
            "img_description": final_inputs["img_description"]
        }
        
        final_chain = loaded_memory | retrieved_documents | answer_chain
             
        return final_chain
  
    def generateLlama3context(history):
        """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

        What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        Bonjour! The capital of France is Paris!<|eot_id|><|start_header_id|>user<|end_header_id|>

        What can I do there?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        Paris, the City of Light, offers a romantic getaway with must-see attractions like the Eiffel Tower and Louvre Museum, romantic experiences like river cruises and charming neighborhoods, and delicious food and drink options, with helpful tips for making the most of your trip.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Give me a detailed list of the attractions I should visit, and time it takes in each one, to plan my trip accordingly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        return ""
    
    
    def buildChainForConversationalRAG_Standalone(self,llm_template_name):
        '''
        This chain creates a standalone question based on the input and the complete chat history of the conversion.
        This standalone question is afterwards used to retrieve the relevant documents and the answer of the llm. 
        It uses a RAG pipeline.
        '''
        #This template is used to create a standalone question  outcommented: Do the rephrase only if there is a chat history. / If the user asks about something you wrote given the following conversation and a follow up question. 
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(llm_templates["standalone_question"].getTemplate())
        
        ANSWER_PROMPT = ChatPromptTemplate.from_template(llm_templates[llm_template_name].getTemplate())
        
        self.DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
        
        # # ConversationBufferMemory is a buffer for the messages from a conversation and a template from langchain
        # self.memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")

        # # With this we load the memory of the chat history
        # # This adds a "memory" key to the input object
        # loaded_memory = RunnablePassthrough.assign(
        #     chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history")
        # )
        
        # It is possible that a user asks a follow up question to the previous context. Therefore a first chain is created to generate a standalone question with the complete information. 
        # This question will later be used to generate the answer
        standalone_question = RunnableParallel({
            "system_prompt": lambda x: x["system_prompt"],
            "img_description": lambda x: x["img_description"],
            "standalone_question": {
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }            
            | CONDENSE_QUESTION_PROMPT
            | self.llm_standalone
            | StrOutputParser()
        }
        )
    
        retrieved_documents = RunnableParallel({
            "system_prompt": lambda x: x["system_prompt"],
            "docs": itemgetter("standalone_question") | self.retriever,
            "question": lambda x: x["standalone_question"],
            "img_description":lambda x: x["img_description"],
        })

        final_inputs = {
            "system_prompt": lambda x: x["system_prompt"],
            "context": lambda x: self._combine_documents(x["docs"]),
            "sources": lambda x: self._sources_documents(x["docs"]),
            "question": itemgetter("question"),
            "img_description": lambda x: x["img_description"],
        }

        answer_chain = {
            "system_prompt": lambda x: x["system_prompt"],
            "answer": final_inputs | ANSWER_PROMPT | self.llm | StrOutputParser(),
            "question": itemgetter("question"),
            "context": final_inputs["context"],
            "sources": final_inputs["sources"],
            "img_description": final_inputs["img_description"]
        }
        
        final_chain = standalone_question | retrieved_documents | answer_chain
        return final_chain
    

    
    
   
    def call_Chain(self, question,image_description,chaintype,systemprompt_template_name,llm_template_name,history):
        """
        Calls a conversational RAG (Retrieval-Augmented Generation) model to generate an answer to a given question.

        This function sends a question to the RAG model, retrieves the answer, and stores the question-answer pair in memory 
        for context in future interactions.

        Parameters:
        question (str): The question to be answered by the RAG model.
        
        Returns:
        dict: A dictionary containing the generated answer from the RAG model.
        """
        print(history)
        # Prepare the input for the RAG model
        if(image_description != ""):
            img_description = f"Description of provided image: \n {image_description}"
        else:
            img_description = ""
        
        #get the prompt template of the actual set conversion type
        system_prompt = systemprompt_templates[systemprompt_template_name].getPrompt()
        #build the inputs for the chains
        
        inputs = {"question": question, "img_description": img_description, "system_prompt": system_prompt, "chat_history":history}
        
        #invoke the correct chain
        if(chaintype == "plain"):
            chain_plain = self.buildChainForPlainInference(llm_template_name)
            result = chain_plain.invoke(inputs)
            
        elif(chaintype =="rag"):
            chain_rag = self.buildChainRAG(llm_template_name)  
            result = chain_rag.invoke(inputs)
        
        # legacy 
        # elif(chaintype =="rag_history"):
        #     chain_rag_history = self.buildChainForConversationalRAG(llm_template_name)
        #     result = chain_rag_history.invoke(inputs)
            
        else:
            if("llama2" in llm_template_name):
                temp_name = "llama2_plain_without_chathistory"
            else:
                temp_name = "llama3_plain_without_chathistory"
                
            chain_rag_history_standalone = self.buildChainForConversationalRAG_Standalone(temp_name)    
            result = chain_rag_history_standalone.invoke(inputs)
            # Save the current question and its answer to memory for future context
            #self.memory.save_context(inputs, {"answer": result["answer"]})        
        
        # Return the result
        return result
    
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
    
    def generateDocumentsmarkdown(self, documents):
        response = ""
        for document in documents:
            response = response + f"- {document['source']}  \n  \n"
        return response
    
    @traceable(
        run_type="llm",
        name="generate Answer",
        project_name="Evaluation MedVet KB LLaMA 3 Finetuned"
        # project_name="MedVet Demo Traces"
    )
    def generateAnswerConversionChainRAG(self, 
                       agent_id, 
                       prompt_user, 
                       history, 
                       display_combined:str, 
                       mode_assistant:str, 
                       use_rag:str, 
                       image:str, 
                       ip_address_llava:str, 
                       chaintype:str, 
                       llm_template_name:str, 
                       max_new_tokens, 
                       temperature):
        
        #No image --> use just LlaMA pipeline
        if prompt_user != "" and image == "":
            print(f"Generate answer only with LlaMA")
            try:
                prompt_llama = prompt_user
                response_llama = self.call_Chain(question=prompt_llama,image_description="",chaintype=chaintype,llm_template_name=llm_template_name, systemprompt_template_name=mode_assistant,history=history)
                self.databaseWriter.insert_chat( agent_id=agent_id, prompt_user=prompt_user, prompt_llava="", prompt_llama=prompt_llama, answer_llava="", answer_llama=response_llama["answer"], answer_combined = "", image = "", mode_display="", mode_assistant=mode_assistant, mode_rag=use_rag)

                status = "OK"
                
                #format in streamlit markdown
                if "sources" in response_llama:
                    response = f"**Result LlaMA and RAG**\n\n{response_llama['answer']}\n\n**Sources**\n\n{self.generateDocumentsmarkdown(response_llama['sources'])}"
                else: 
                    response = f"**Result LlaMA and RAG**\n\n{response_llama['answer']}"
                    

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
                prompt_llama = prompt_user
                response_llama = self.call_Chain(question=prompt_llama,image_description="",chaintype=chaintype,llm_template_name=llm_template_name, systemprompt_template_name=mode_assistant,history=history)
                
                prompt_llava = self.formatPromptLlaVA(prompt_user)
                response_llava = self.generateLLaVAresponse(image=image, ip_address_llava=ip_address_llava , max_new_tokens=max_new_tokens, prompt_llava=prompt_llava, temperature=temperature)
                
                self.databaseWriter.insert_chat( agent_id=agent_id, prompt_user=prompt_user, prompt_llava=prompt_llava, prompt_llama=prompt_llama, answer_llava=response_llava["result"], answer_llama=response_llama["answer"], answer_combined = "", image = image, mode_display=display_combined, mode_assistant=mode_assistant, mode_rag=use_rag)
                
                if "sources" in response_llama:             
                    response = f'''**Result LlaMA and RAG**\n\n{response_llama['answer']}\n\n**Sources**\n\n{self.generateDocumentsmarkdown(response_llama['sources'])}\n\n**Result LlaVA- Med**\n\n{response_llava["result"]}'''
                else:
                    response = f'''**Result LlaMA**\n{response_llama['answer']}\n\n**Result LlaVA- Med**\n{response_llava["result"]}'''
                
                status = "OK"
            except Exception as e: 
                logger.info(f"Failed to generate Answer {e}")
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
                
                print(f'''Response LlaVA: {response_llava["result"]}''')
                
                img_desc = response_llava["result"]
                prompt_llama = prompt_user
                response_llama = self.call_Chain(question=prompt_llama,image_description=img_desc,chaintype=chaintype,llm_template_name=llm_template_name, systemprompt_template_name=mode_assistant,history=history)
                
                
                self.databaseWriter.insert_chat( agent_id=agent_id, prompt_user=prompt_user, prompt_llava=prompt_llava, prompt_llama=prompt_llama, answer_llava=response_llava["result"], answer_llama=response_llama["answer"], answer_combined = response_llama["answer"], image = image, mode_display=display_combined, mode_assistant=mode_assistant, mode_rag=use_rag)
                if "sources" in response_llama:                
                    response = f"**Result combination of LlaVA- MEd and LlaMA with RAG**\n\n{response_llama['answer']}\n\n**Sources**\n\n{self.generateDocumentsmarkdown(response_llama['sources'])}"
                else:
                    response = f"**Result combination of LlaVA- MEd and LlaMA**\n\n{response_llama['answer']}"
                    
                status = "OK"
            
            except Exception as e: 
                logger.info(f"Failed to generate Answer {e}")
                response = "Failed to generate Answer"
                response_llama = ""
                response_llava = ""
                status = "Failed"
            
            return {"status":status, "response":response, "answer_llama": response_llama, "answer_llava": response_llava}
        
        
        else:
            print(f"No valid configuration")        
    