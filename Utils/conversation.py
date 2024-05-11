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
    model_type: str = "llama"
    skip_next: bool = False

    def get_prompt(self):
        if(self.model_type == "llama"):
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
        elif(self.model_type == "falcon"):
            ret = "System: " + self.system 
            for role, message in self.messages:
                if message:
                    if role =="User":
                        ret += role + ": " + message
                    else:
                        ret += role + ": " + message
                else:
                    ret += role + ":"
            return ret
    
    def append_message(self, role, message):
        self.messages.append([role, message])
    
    def getSystemPrompt(self):
        return self.system


    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            model_type = self.model_type)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "model_type": self.model_type
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
    model_type="llama"
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
    model_type="llama"
)

simple_conv_falcon= Conversation(
    system="You are an intelligent assistant designed to support veterinarians by providing detailed and specific responses related to veterinary medicine, including diagnosis and treatment."
           "You analyse the provided case and tailor your answers to the specific species and context of the inquiry."
           "Follow the instructions carefully and explain your answers in detail.",
    roles=("User", "Assistant"),
    messages=(
        ("User", "Hi!"),
        ("Assistant", "Hi there!  How can I help you?")
    ),
    offset=2,
    model_type = "falcon"
)
"<s>[INST] <<SYS>> {system_prompt} <</SYS>> {user_message} \n Image Description: {img_desc}[/INST]"
simple_langchain_kb= Conversation(
    system="""<s>[INST] <<SYS>> You are an intelligent assistant designed to be a knowledge base for veterinarians. You provide detailed and specific responses related to veterinary medicine. \
                Use your complete knowledge to explain the aked questions very specific. \
                Follow the instructions carefully and explain your answers in detail. \
                If you don't know the answer, just say that you don't know, don't try to make up an answer. <</SYS>> \

                {context}
                {img_description}

                Question: {question}
                [/INST]""",
    roles=("human", "Assistant"),
    messages=(
        ("human", "Hi!"),
        ("Assistant", "Hi there!  How can I help you?")
    ),
    offset=2,
    model_type = "langchain"
)

simple_langchain= Conversation(
    system="""[INST] You are an intelligent assistant designed to support veterinarians by providing detailed and specific responses related to veterinary medicine, including diagnosis and treatment.\
                You analyse the provided case and tailor your answers to the specific species and context of the inquiry. \
                Follow the instructions carefully and explain your answers in detail. \
                If you don't know the answer, just say that you don't know, don't try to make up an answer." \

                {context}
                {img_description}

                Question: {question}
                [/INST]""",
    roles=("human", "Assistant"),
    messages=(
        ("human", "Hi!"),
        ("Assistant", "Hi there!  How can I help you?")
    ),
    offset=2,
    model_type = "langchain"
)

simple_langchain_llama3= Conversation(
    system="""<|begin_of_text|>
<|start_header_id|>
  system
<|end_header_id|>
You are an intelligent assistant designed to support veterinarians by providing detailed and specific responses related to veterinary medicine, including diagnosis and treatment.
You analyse the provided case and tailor your answers to the specific species and context of the inquiry. 
Follow the instructions carefully and explain your answers in detail. 
If you don't know the answer, just say that you don't know, don't try to make up an answer." 
<|eot_id|>
<|start_header_id|>
   user
<|end_header_id|>
  Answer the user question based on the context provided below
  Context :{context} {img_description}
  Question: {question}
<|eot_id|>
<|start_header_id|>
  assistant
<|end_header_id|>

[INST]""",
    roles=("human", "Assistant"),
    messages=(
        ("human", "Hi!"),
        ("Assistant", "Hi there!  How can I help you?")
    ),
    offset=2,
    model_type = "langchain"
)


# default_conversation = simple_conv_Llama_casesolver
default_conversation = simple_conv_falcon

conv_templates = {
    "default": simple_conv_Llama_casesolver,
    "simple_kb": simple_conv_Llama_Knowledgebase,
    "simple_case": simple_conv_Llama_casesolver,
    "simpe_falcon":simple_conv_falcon,
    "simple_langchain": simple_langchain,
    "simple_langchain_kb": simple_langchain_kb,
    "simple_langchain_llama3": simple_langchain_llama3,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())