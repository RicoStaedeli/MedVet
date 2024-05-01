from pydantic import BaseModel

class MMGeneration(BaseModel):
    prompt: str
    #negative_prompt: str | None = """bad anatomy, fused fingers, deformed, weird, bad resolution, weird, worst quality, worst resolution,too blurry, not relevant,unreal"""
    agent_id: str
    ip_address:str
    img:str
    
    