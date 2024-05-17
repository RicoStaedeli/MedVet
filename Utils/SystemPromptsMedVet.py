import dataclasses
from typing import List


@dataclasses.dataclass
class SystemPromptsMedVet:
    """A class that keeps all prompt templates"""
    system: str
    model_type: str = "llama"
    
    def getPrompt(self):
        return self.system


    def copy(self):
        return SystemPromptsMedVet(
            system=self.system,
            model_type = self.model_type)

    def dict(self):
        return {
            "system": self.system,
            "model_type": self.model_type
        }


simple_kb= SystemPromptsMedVet(
    system="""You are an intelligent assistant designed to be a knowledge base for veterinarians. You provide detailed and specific responses related to veterinary medicine. Use your complete knowledge to explain the aked questions very specific. Follow the instructions carefully and explain your answers in detail. If you don't know the answer, just say that you don't know, don't try to make up an answer.""",
    model_type = "langchain"
)


simple_case= SystemPromptsMedVet(
    system="""You are an intelligent assistant designed to support veterinarians by providing detailed and specific responses related to veterinary medicine, including diagnosis and treatment. You will see a description of a case. analyse the provided case and tailor your answers to the specific species and context of the inquiry. Follow the instructions carefully and explain your answers in detail. If you don't know the answer, just say that you don't know, don't try to make up an answer.""",
    model_type = "langchain"
)


# default_conversation = simple_conv_Llama_casesolver
default_template = simple_kb

systemprompt_templates = {
    "default": simple_kb,
    "simple_case": simple_case,
    "simple_kb": simple_kb,
}

if __name__ == "__main__":
    print(default_template.getPrompt())