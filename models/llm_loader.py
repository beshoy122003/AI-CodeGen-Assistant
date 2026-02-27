from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"

def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
        trust_remote_code=True
    )

    return tokenizer, model