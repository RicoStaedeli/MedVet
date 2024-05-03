from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

#Utils
from Utils.constants import MODE_ASSISTANT, MODE_DISPLAY, MODE_RAG

from Utils.logger import get_logger
from Utils.config import load_config
config = load_config("config/cfg.yaml")
logger = get_logger(__name__, config)

print(config)

#Requests
from RequestModels.requestMMGeneration import MMGeneration

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

##############################
#### Enable CORS 
##############################
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##############################
#### API Endpoints
##############################
@app.get("/", include_in_schema=False)
async def read_root():
    logger.info(f"Redirect to /docs")
    return RedirectResponse(url='/docs')

   
@app.put("/generate",tags=["Text Generation"])
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
        mmService.set_assistantMode(mode_assistant)
        
        response = mmService.generateAnswer(agent_id=agent_id, display_combined=display_combined, image=image, ip_address_llava=ip_address_llava,max_new_tokens=max_new_tokens, temperature=temperature,mode_assistant=mode_assistant, prompt_user=prompt,use_rag=use_rag )

        return response
    
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        # raise HTTPException(status_code=500, detail="Could not generate a text output. More details in the Logs.")
        return {"result":"Failed"}
   
@app.post("/clearchat",tags=["Text Generation"])
def clearchat():
    try:
        mmService.clear_history()
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        # raise HTTPException(status_code=500, detail="Could not generate a text output. More details in the Logs.")
        return {"result":"Failed"}
    
##############################
#### HTTP Exception
##############################

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