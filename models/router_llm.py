from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from langchain_community.llms import HuggingFacePipeline



## Model for Router and Explain Chain
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

def load_router_llm():
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                 device_map="auto",
                                                    torch_dtype=torch.float16,
                                                    load_in_4bit=True)
    return tokenizer, model


## For convert the model to langchain format to use in the router chain and explain chain 
def get_langchain_llm(model, tokenizer):

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        return_full_text=False,
        do_sample=False,
        temperature=None
    )

    return HuggingFacePipeline(pipeline=pipe)