from data.humaneval_loader import load_humaneval
from vectordb.chroma_client import index_data

# print("Loading data...")
# data = load_humaneval()

# print("Indexing data into Chroma DB...")
# index_data(data)

# print("Done ya man")




## Test1: Testing the retrival and generation 
# from models.llm_loader import load_llm
# from vectordb.chroma_client import create_vector_store
# from chains.generate_chain import generate_code

# tokenizer, model = load_llm()

# db = create_vector_store()

# query = "write a python function to check if a number is prime"

# ##  Retrival
# docs = db.similarity_search(query, k=5)

# ## Remove Dublicates
# unique_docs = []
# seen = set()

# for d in docs:
#     content = d.page_content.strip()

#     if content not in seen:
#         unique_docs.append(d)
#         seen.add(content)

# ## Generation of Code
# result = generate_code(query, unique_docs, tokenizer, model)

# print(result)



## Test2: Testing the router chain
# from models.router_llm import load_router_llm
# from chains.router_chain import SemanticRouter

# router = SemanticRouter()

# print("Router Ready")
# print("=" * 50)

# queries = [
#     "what is a prime number?",
#     "explain recursion in simple terms",
#     "write a python function to check prime",
#     "generate a function to sort a list",
# ]

# for q in queries:
#     print("Query:", q)
#     print("Intent:", router.route(q))
#     print("-" * 50)

## Test3: Testing the explain chain + memory
from models.router_llm import load_router_llm, get_langchain_llm
from memory.memory import build_memory
from chains.explain_chain import build_explain_chain

router_tokenizer, router_model = load_router_llm()

llm = get_langchain_llm(router_model, router_tokenizer)

memory = build_memory(llm)

explain_chain = build_explain_chain(llm, memory)



response = explain_chain.run("What is a prime number?")
print(response.strip())

print("-----")

response = explain_chain.run("Give an example")
print(response.strip())

print("-----")

response = explain_chain.run("Why is it important?")
print(response.strip())