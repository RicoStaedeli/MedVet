from pydantic import BaseModel, Field
class MMGeneration(BaseModel):
    prompt: list
    agent_id: str
    ip_address_llava: str
    img: str
    chaintype: str | None = Field(default= "rag")
    display_combined: bool  | None = Field(default= 0)
    mode_assistant:str | None = Field(default= "simple_kb")
    llm_template_name:str | None = Field(default= "llama2_plain")
    use_rag: bool| None = Field(default= 1)
    temperature: float | None = 0.7
    max_new_tokens: int | None = 1024
    
    