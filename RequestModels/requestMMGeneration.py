from pydantic import BaseModel, Field
from Utils.constants import MODE_ASSISTANT, MODE_DISPLAY, MODE_RAG
class MMGeneration(BaseModel):
    prompt: str
    agent_id: str
    ip_address_llava: str
    img: str
    display_combined: bool  | None = Field(default= 0)
    mode_assistant:str | None = Field(default= MODE_ASSISTANT.CASE, examples=[f"{MODE_ASSISTANT.CASE},{MODE_ASSISTANT.KB}"])
    use_rag: bool| None = Field(default= 1)
    temperature: float | None = 0.7
    max_new_tokens: int | None = 1024
    
    