from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

#Utils
from Utils.constants import MODE_DISPLAY, MODE_RAG
from Utils.conversation import (conv_templates)
from Utils.SystemPromptsMedVet import (systemprompt_templates)
from Utils.LLMTemplates import (llm_templates)
from Utils.logger import get_logger
from Utils.config import load_config
config = load_config("config/cfg.yaml")
logger = get_logger(__name__, config)

print(config)

#Requests
from RequestModels.requestMMGeneration import MMGeneration
from RequestModels.requestQuestion import Questions

#Serices
from MMLLM.ModelService import MMLLMService
mmService = MMLLMService(config)

tags_metadata = [
    {
        "name": "Text Generation",
        "description": "Generate text by sending a prompt as JSON",
    },
]

app = FastAPI(
        title="MedVet API",
        version="0.0.1",
        openapi_tags=tags_metadata,
        swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"}
    )

########################################################################################################################
################################## Enable CORS #########################################################################
########################################################################################################################

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

########################################################################################################################
##############################    API Endpoints     ####################################################################
########################################################################################################################
@app.get("/", include_in_schema=False)
async def read_root():
    logger.info(f"Redirect to /docs")
    return RedirectResponse(url='/docs')

   
@app.put("/generatelagacy",tags=["Text Generation"], include_in_schema=False)
def generate_answer(txtGen: MMGeneration):
    logger.info(f"Received request: {txtGen}")
    try:
        #Set the prmompt
        prompt = txtGen.prompt.replace('"', ' ').replace("'", ' ')
        logger.info(f"replace disturbing characters and generate prompt: {prompt}")
        agent_id = txtGen.agent_id
        ip_address_llava = txtGen.ip_address_llava
        image = txtGen.img
        max_new_tokens = txtGen.max_new_tokens
        temperature = txtGen.temperature
        
        if txtGen.display_combined:
            display_combined = MODE_DISPLAY.COMBINED
        else:
            display_combined = MODE_DISPLAY.SEPARATE
        
        if txtGen.use_rag:
            use_rag = MODE_RAG.RAG
        else:
            use_rag = MODE_RAG.NORAG
        
        mode_assistant = txtGen.mode_assistant
        
        response = mmService.generateAnswer(agent_id=agent_id, display_combined=display_combined, image=image, ip_address_llava=ip_address_llava,max_new_tokens=max_new_tokens, temperature=temperature,mode_assistant=mode_assistant, prompt_user=prompt,use_rag=use_rag )

        return response
    
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        # raise HTTPException(status_code=500, detail="Could not generate a text output. More details in the Logs.")
        return {"result":"Failed"}
    
@app.post("/clearchat",tags=["Text Generation"])
def clearchat():
    try:
        return {"result":"Succeded"}
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        # raise HTTPException(status_code=500, detail="Could not generate a text output. More details in the Logs.")
        return {"result":"Failed"}

def formatPrompt(conversation):
    formatted_conversation = ""
    last_user_message = ""

    for entry in conversation:
        if entry['role'] == 'assistant':
            if(entry['content'] != "How may I assist you?"):
                formatted_conversation += f"Assistant: {entry['content']}\n"
        elif entry['role'] == 'user':
            formatted_conversation += f"User: {entry['content']}\n"
            last_user_message = entry['content']

    return last_user_message, formatted_conversation


@app.put("/generate",tags=["Text Generation"])
def generateragconversational(txtGen: MMGeneration):
    # logger.info(f"Received request: {txtGen}")
    try:    
        #Set the prmompt
        mode_assistant = txtGen.mode_assistant
        prompt,history = formatPrompt(txtGen.prompt)
        agent_id = txtGen.agent_id
        ip_address_llava = txtGen.ip_address_llava
        image = txtGen.img
        max_new_tokens = txtGen.max_new_tokens
        temperature = txtGen.temperature
        chaintype = txtGen.chaintype
        llm_template_name = txtGen.llm_template_name
        
        if txtGen.display_combined:
            display_combined = MODE_DISPLAY.COMBINED
        else:
            display_combined = MODE_DISPLAY.SEPARATE
        
        if txtGen.use_rag:
            use_rag = MODE_RAG.RAG
        else:
            use_rag = MODE_RAG.NORAG
        
        
        response = mmService.generateAnswerConversionChainRAG(agent_id=agent_id, 
                                                              display_combined=display_combined, 
                                                              image=image, 
                                                              ip_address_llava=ip_address_llava,
                                                              max_new_tokens=max_new_tokens, 
                                                              temperature=temperature,
                                                              mode_assistant=mode_assistant, 
                                                              prompt_user=prompt,
                                                              history = history,
                                                              use_rag=use_rag,
                                                              llm_template_name = llm_template_name,
                                                              chaintype=chaintype )

        return response
    
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        # raise HTTPException(status_code=500, detail="Could not generate a text output. More details in the Logs.")
        return {"result":"Failed"}
    

@app.put("/ragdocs",tags=["Text Generation"])
def getDocuments(load: Questions):
    try:
        response = mmService.getRelevantDocuments(load.prompt)
        return {"Response":response}
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        # raise HTTPException(status_code=500, detail="Could not generate a text output. More details in the Logs.")
        return {"result":"Failed"}    


@app.get("/convtemplates",tags=["Text Generation"])
def ragconversational():
    try:
        templates = []
        for key in conv_templates.keys():
            if conv_templates[key].model_type == "langchain":
                templates.append(key)
        return {"Response":templates}
        
    except Exception as e:
        logger.error(f"Error during retrieving conversation templates {e}")
        # raise HTTPException(status_code=500, detail="Could not generate a text output. More details in the Logs.")
        return {"result":"Failed"}

@app.get("/systemprompttemplates",tags=["Text Generation"])
def systemprompttemplates():
    try:
        templates = []
        for key in systemprompt_templates.keys():
            templates.append(key)
        return {"Response":templates}
        
    except Exception as e:
        logger.error(f"Error during retrieving conversation templates {e}")
        # raise HTTPException(status_code=500, detail="Could not generate a text output. More details in the Logs.")
        return {"result":"Failed"}
    
@app.get("/llmtemplates",tags=["Text Generation"])
def llmtemplates():
    try:
        templates = []
        for key in llm_templates.keys():
            if llm_templates[key].show_for_user == 1:
                templates.append(key)
        return {"Response":templates}
        
    except Exception as e:
        logger.error(f"Error during retrieving conversation templates {e}")
        # raise HTTPException(status_code=500, detail="Could not generate a text output. More details in the Logs.")
        return {"result":"Failed"}

@app.get("/chaintypes",tags=["Text Generation"])
def chaintypes():
    try:
        chaintypes = ['rag','plain','rag_history_standalone']
        return {"Response":chaintypes}
        
    except Exception as e:
        logger.error(f"Error during retrieving conversation templates {e}")
        # raise HTTPException(status_code=500, detail="Could not generate a text output. More details in the Logs.")
        return {"result":"Failed"}

##########################################################################################
##############################   HTTP Exception             ##############################
##########################################################################################

class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: UnicornException):
    logger.error(f"404 error encountered: {request.url.path}")
    return JSONResponse(
        status_code=404,
        content={
            "message": f"Oops! The endpoint '{request.url.path}' does not exist. Please read the docs '/docs'."
        }
    )

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )