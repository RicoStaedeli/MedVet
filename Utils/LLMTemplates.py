import dataclasses
from typing import List


@dataclasses.dataclass
class LlmTemplate:
    """A class that keeps all templates"""
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


llama2_RAG = LlmTemplate(
    template="""<s>[INST] <<SYS>> {system_prompt}<</SYS>> 

            {context}
            {img_description}

            Question: {question}
            [/INST]"""
)

llama2_plain = LlmTemplate(
    template="""<s>[INST] <<SYS>> {system_prompt}<</SYS>> 
            
            {img_description}
            Question: {question}
            [/INST]"""
)


llama3_RAG = LlmTemplate(
    template="""{system_prompt}
    
                Context :{context} {img_description}
                
                Question: {question}
                """
)

llama3_plain = LlmTemplate(
    template="""{system_prompt}
                
                {img_description}
                Question: {question}
                """
)

llm_generic = LlmTemplate(
    template="""{system_prompt}
            \n ------- \n
            {context}
            {img_description}
            \n ------- \n
            Question: {question}
            """
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

default_ll_template = llama2_plain

llm_templates = {
    "default": llama2_plain,
    "llama3_RAG": llama3_RAG,
    "llama3_plain": llama3_plain,
    "llama2_plain": llama2_plain,
    "llama2_RAG": llama2_RAG,
    "llm_generic":llm_generic,
    "standalone_question": standalone_question
}


if __name__ == "__main__":
    print(default_ll_template.getTemplate())