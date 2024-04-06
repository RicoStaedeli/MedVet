from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

#Utils
from Utils.logger import get_logger
from Utils.config import load_config
config = load_config("config/cfg.yaml")
logger = get_logger(__name__, config)

print(config)

#Requests
from RequestModels.requestTextGeneration import TextGeneration

#Serices
from LLM.LLMService import LLMService
LlmService = LLMService(config)

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
def generate_text(txtGen: TextGeneration):
    logger.info(f"Received request: {txtGen}")
    try:
        #Set the prmompt
        prompt = txtGen.prompt.replace('"', ' ').replace("'", ' ')
        logger.info(f"replace disturbing characters and generate prompt: {prompt}")
        agent_id = txtGen.agent_id
        result = LlmService.generate(prompt,agent_id)
       
        return {"id":1,"prompt": prompt,"result":result}
    
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        raise HTTPException(status_code=500, detail="Could not generate a text output. More details in the Logs.")

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