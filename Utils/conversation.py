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


simple_conv_Knowledgebase = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!"),
        ("Assistant", "Hi there!  How can I help you?\n")
    ),
    offset=2,
)

simple_conv_Llama_casesolver = Conversation(
    system="You are an intelligent assistant designed to support veterinarians by providing detailed and specific responses related to veterinary medicine, including diagnosis and treatment."
           "Tailor your answers to the specific species and context of the inquiry, offering practical advice, and remind users to verify all medical information with official sources."
           "Follow the instructions carefully and explain your answers in detail.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!"),
        ("Assistant", "Hi there!  How can I help you?\n")
    ),
    offset=2,
)


default_conversation = simple_conv_Llama_casesolver
conv_templates = {
    "default": simple_conv_Llama_casesolver,
    "simple": simple_conv_Knowledgebase,
    "simple_Llama": simple_conv_Llama_casesolver
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())