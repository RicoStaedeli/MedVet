import dataclasses
from typing import List


@dataclasses.dataclass
class LlmTemplate:
    """A class that keeps all templates"""
    template: str
    show_for_user: bool
    model_type:str
    
    
    def getTemplate(self):
        return self.template


    def copy(self):
        return LlmTemplate(
            template=self.template,
            show_for_user=self.show_for_user,
            model_type=self.model_type
            )

    def dict(self):
        return {
            "template": self.template,
            "show_for_user":self.show_for_user,
            "model_type": self.model_type
        }


llama2_RAG = LlmTemplate(
    template="""<s>[INST] <<SYS>> {system_prompt}<</SYS>> 

            {chat_history}

            Context: {context}
            
            {img_description}

            Question: {question}
            [/INST]""",
    show_for_user = 1,
    model_type="llama2"
)

llama2_plain = LlmTemplate(
    template="""<s>[INST] <<SYS>> {system_prompt}<</SYS>> 
            
            {chat_history}
            
            {img_description}
            
            Question: {question}
            [/INST]""",
    show_for_user = 1,
    model_type="llama2"
)

llama2_plain_without_chathistory = LlmTemplate(
    template="""<s>[INST] <<SYS>> {system_prompt}<</SYS>> 
            
            {img_description}
            
            Question: {question}
            [/INST]""",
    show_for_user = 0,
    model_type="llama2"
)


llama3_RAG = LlmTemplate(
    template="""{system_prompt}
    
                {chat_history}
    
                Context :{context} {img_description}
                
                Question: {question}
                """,
    show_for_user = 1,
    model_type="llama3"
)

llama3_plain = LlmTemplate(
    template="""{system_prompt}              
    {chat_history}
    {img_description}
                
    Question: {question}""",
    show_for_user = 1,
    model_type="llama3"
)

llama3_plain_llamatemplate = LlmTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt} <|eot_id|><|start_header_id|>user<|end_header_id|>

{chat_history}{img_description}{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    show_for_user = 1,
    model_type="llama3"
)

llama3_plain_without_chathistory = LlmTemplate(
    template="""{system_prompt}
                
                {img_description}
                
                Question: {question}
                """,
    show_for_user = 0,
    model_type="llama3"
)

llm_generic = LlmTemplate(
    template="""{system_prompt}
            \n ------- \n
            {chat_history}
            \n ------- \n
            {context}
            {img_description}
            \n ------- \n
            Question: {question}
            """,
    show_for_user = 1,
    model_type="all"
)

standalone_question= LlmTemplate(
    template="""            
            Rephrase the Follow Up Input to be a standalone question based on the chat history. 

            Chat History:
            {chat_history}
            
            Follow Up Input: {question}
            
            Standalone question:
        """,
    show_for_user = 0,
    model_type="none"
)

default_llm_template = llama2_plain

llm_templates = {
    "default": llama2_plain,
    "llama2_plain": llama2_plain,
    "llama2_RAG": llama2_RAG,
    "llama3_plain": llama3_plain,
    "llama3_RAG": llama3_RAG,
    "llama3_plain_meta": llama3_plain_llamatemplate,
    "llm_generic":llm_generic,
    "standalone_question": standalone_question,
    "llama2_plain_without_chathistory": llama2_plain_without_chathistory,
    "llama3_plain_without_chathistory": llama3_plain_without_chathistory
}


if __name__ == "__main__":
    print(default_llm_template.getTemplate())