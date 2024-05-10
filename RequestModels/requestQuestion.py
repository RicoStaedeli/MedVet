from pydantic import BaseModel, Field
from Utils.constants import MODE_ASSISTANT, MODE_DISPLAY, MODE_RAG
class Questions(BaseModel):
    prompt: str

    
    