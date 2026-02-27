## Test4: Testing the explain chain + memory
from models.router_llm import load_router_llm, get_langchain_llm
from memory.memory import build_memory
from chains.explain_chain import build_explain_chain

router_tokenizer, router_model = load_router_llm()

llm = get_langchain_llm(router_model, router_tokenizer)

memory = build_memory(llm)

explain_chain = build_explain_chain(llm, memory)



response = explain_chain.invoke(
    {"question": "What is a prime number?"}
)["text"]

print(response.strip())
print("-----")

response = explain_chain.invoke(
    {"question": "Give an example"}
)["text"]

print(response.strip())
print("-----")

response = explain_chain.invoke(
    {"question": "Why is it important?"}
)["text"]

print(response.strip())