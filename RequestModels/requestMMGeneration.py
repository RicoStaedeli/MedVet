from pydantic import BaseModel, Field
from Utils.constants import MODE_ASSISTANT, MODE_DISPLAY, MODE_RAG
class MMGeneration(BaseModel):
    prompt: str
    #negative_prompt: str | None = """bad anatomy, fused fingers, deformed, weird, bad resolution, weird, worst quality, worst resolution,too blurry, not relevant,unreal"""
    agent_id: str
    ip_address_llava: str
    img: str
    display_combined: bool  | None = Field(default= 0, examples=[f"0 = {MODE_DISPLAY.SEPARATE} 1 = {MODE_DISPLAY.COMBINED}"])
    mode_assistant:str | None = Field(default= MODE_ASSISTANT.CASE, examples=[f"{MODE_ASSISTANT.CASE},{MODE_ASSISTANT.KB}"])
    use_rag: bool| None = Field(default= 1, examples=[f"0 = {MODE_RAG.NORAG} 1 = {MODE_RAG.RAG}"])
    temperature: float | None = 0.7
    max_new_tokens: int | None = 1024
    
    