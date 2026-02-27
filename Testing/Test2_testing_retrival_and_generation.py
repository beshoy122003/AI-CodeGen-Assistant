from data.humaneval_loader import load_humaneval
from vectordb.chroma_client import index_data


# Test2: Testing the retrival and generation 
from models.llm_loader import load_llm
from vectordb.chroma_client import create_vector_store
from chains.generate_chain import generate_code

tokenizer, model = load_llm()

db = create_vector_store()

query = "write a python function to check if a number is prime"

##  Retrival
docs = db.similarity_search(query, k=5)

## Remove Dublicates
unique_docs = []
seen = set()

for d in docs:
    content = d.page_content.strip()

    if content not in seen:
        unique_docs.append(d)
        seen.add(content)

## Generation of Code
result = generate_code(query, unique_docs, tokenizer, model)

print(result)