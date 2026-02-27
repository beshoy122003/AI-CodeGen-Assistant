from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def build_explain_chain(llm, memory):

    prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
You are a helpful AI tutor.

{chat_history}

Human: {question}
AI:
"""
    )

    return LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )