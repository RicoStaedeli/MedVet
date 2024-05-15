from pydantic import BaseModel, Field
class MMGeneration(BaseModel):
    prompt: str
    agent_id: str
    ip_address_llava: str
    img: str
    chaintype: str | None = Field(default= "rag")
    display_combined: bool  | None = Field(default= 0)
    mode_assistant:str | None = Field(default= "simple_langchain_kb")
    use_rag: bool| None = Field(default= 1)
    temperature: float | None = 0.7
    max_new_tokens: int | None = 1024
    
    