from LLM.RagDocumentLoader import RagDocumentLoader
from LLM.ModelLoaderLLM import LlamaForCausalRAG


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

import os
import time

#Utils
from Utils.logger import get_logger
from Utils.config import load_config
config = load_config("config/cfg.yaml")
logger = get_logger(__name__, config)

source_directory = config.get("RAG", {}).get("source_path")
EMBEDDINGS_MODEL_NAME =  config.get("RAG", {}).get("EMBEDDINGS_MODEL_NAME")
TARGET_SOURCE_CHUNKS = config.get("RAG", {}).get("TARGET_SOURCE_CHUNKS")

documentLoader = RagDocumentLoader(config)
 
#embeddings = GPT4AllEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

try:
    # Create and store locally vectorstore if folder not exit
    print("Creating new vectorstore")
    texts = documentLoader.process_documents()
    print(f"Creating embeddings. May take some minutes...")
    # vectordb = FAISS.fcrarom_documents(texts, embeddings)
    # vectordb.save_local("faiss_index")
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="Database/chroma_db_rag")
    print(f"Ingestion complete! You can now query your visual documents")

except Exception as e:
    print(e)


#loading the vectorstore
#db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llamaModel = LlamaForCausalRAG(config,logger)

model_path = config.get("model", {}).get("path")
llm = llamaModel.load_llm(model_path)

    
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
# Interactive questions and answers
while True:
    query = input("\nEnter a query: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue
    
    # Get the answer from the chain
    start = time.time()
    res = qa.invoke(query)
    answer, docs = res['result'], [] if False else res['source_documents']
    end = time.time()
    # Print the result
    print("\n\n> Question:")
    print(query)
    print(f"\n> Answer (took {round(end - start, 2)} s.):")
    print(answer)
    # Print the relevant sources used for the answer
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)