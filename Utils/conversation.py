import dataclasses
from typing import List


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        
        ret = "<s>[INST] <<SYS>>" + self.system + "<</SYS>>"
        for role, message in self.messages:
            if message:
                if role =="Human":
                    ret += role + ": " + message + "[/INST]"
                else:
                    ret += role + ": " + message + " </s><s>[INST]"
            else:
                ret += role + ":"
        return ret

    
    def append_message(self, role, message):
        self.messages.append([role, message])


    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


simple_conv_Llama_Knowledgebase = Conversation(
    system="You are an intelligent assistant designed to be a knowledge base for veterinarians. You provide detailed and specific responses related to veterinary medicine."
           "Use your complete knowledge to explain the aked questions very specific.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!"),
        ("Assistant", "Hi there!  I am a knowledge base for veterinarians. How can I help you?")
    ),
    offset=2,
)

simple_conv_Llama_casesolver = Conversation(
    system="You are an intelligent assistant designed to support veterinarians by providing detailed and specific responses related to veterinary medicine, including diagnosis and treatment."
           "You analyse the provided case and tailor your answers to the specific species and context of the inquiry."
           "Follow the instructions carefully and explain your answers in detail.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!"),
        ("Assistant", "Hi there!  How can I help you?")
    ),
    offset=2,
)


default_conversation = simple_conv_Llama_casesolver

conv_templates = {
    "default": simple_conv_Llama_casesolver,
    "simple_kb": simple_conv_Llama_Knowledgebase,
    "simple_case": simple_conv_Llama_casesolver
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())