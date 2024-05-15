import dataclasses
from typing import List


@dataclasses.dataclass
class LlmTemplate:
    """A class that keeps all Templays"""
    template: str
    
    
    def getTemplate(self):
        return self.template


    def copy(self):
        return LlmTemplate(
            template=self.template)

    def dict(self):
        return {
            "template": self.template
        }


llama2 = LlmTemplate(
    template="""<s>[INST] <<SYS>> {system_prompt}<</SYS>> 

            {context}
            {img_description}

            Question: {question}
            [/INST]"""
)

llama3 = LlmTemplate(
    template="""<|begin_of_text|>
                <|start_header_id|>
                system
                <|end_header_id|>
                {system_prompt}
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
                """
)

llm_generic = LlmTemplate(
    template="""{system_prompt}

            {context}
            {img_description}

            Question: {question}"""
)

standalone_question= LlmTemplate(
    template="""            
            Rephrase the Follow Up Input to be a standalone question based on the chat history. 

            Chat History:
            {chat_history}
            
            Follow Up Input: {question}
            
            Standalone question:
        """
)

default_template = llama2

templates = {
    "default": llama2,
    "llama3": llama3,
    "llama2": llama2,
    "llm_generic":llm_generic,
    "standalone_question": standalone_question
}


if __name__ == "__main__":
    print(default_template.getTemplate())