# ================================
# Imports
# ================================

from data.humaneval_loader import load_humaneval
from vectordb.chroma_client import index_data, create_vector_store

from models.router_llm import load_router_llm, get_langchain_llm
from models.llm_loader import load_llm

from memory.memory import build_memory
from chains.explain_chain import build_explain_chain
from chains.generate_chain import generate_code
from chains.router_chain import SemanticRouter

import torch
import gc


# ================================
# Toggles
# ================================

RUN_INDEXING = False
RUN_RAG_TEST = False
RUN_ROUTER_TEST = False
RUN_EXPLAIN_TEST = False
RUN_FULL_SYSTEM = True


# ================================
# Indexing (run once)
# ================================

if RUN_INDEXING:
    print("Loading data...")
    data = load_humaneval()

    print("Indexing into Chroma...")
    index_data(data)

    print("Indexing complete ✅")
    exit()


# ================================
# Load always-needed components
# ================================

print("Loading Router...")
router = SemanticRouter()

print("Loading Vector DB...")
db = create_vector_store()


# ================================
# Lazy model placeholders
# ================================

gen_tokenizer = None
gen_model = None

router_tokenizer = None
router_model = None
llm = None
memory = None
explain_chain = None


# ================================
# GPU Cleanup
# ================================

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ================================
# Lazy Loaders
# ================================

def load_explain_components():

    global router_tokenizer, router_model, llm, memory, explain_chain
    global gen_model, gen_tokenizer

    if explain_chain is None:

        if gen_model is not None:
            print("Unloading CodeGen model from GPU...")

            del gen_model
            del gen_tokenizer

            gen_model = None
            gen_tokenizer = None

            clear_gpu()

        print("Loading Explain model (Phi-3)...")

        router_tokenizer, router_model = load_router_llm()
        llm = get_langchain_llm(router_model, router_tokenizer)

        memory = build_memory(llm)
        explain_chain = build_explain_chain(llm, memory)


def load_generate_components():

    global gen_tokenizer, gen_model
    global router_tokenizer, router_model, llm, memory, explain_chain

    if gen_model is None:

        if router_model is not None:
            print("Unloading Explain model from GPU...")

            del router_model
            del router_tokenizer
            del llm
            del explain_chain
            del memory

            router_model = None
            router_tokenizer = None
            llm = None
            explain_chain = None
            memory = None

            clear_gpu()

        print("Loading Code Generation model...")

        gen_tokenizer, gen_model = load_llm()


# ================================
# Unknown Knowledge Handler 🧠
# ================================

def handle_unknown(query: str):

    global db

    print("\nI don't know the answer yet 🤖")
    print("Can you teach me? (y/n)")

    choice = input(">> ")

    if choice.lower() != "y":
        return "Okay 👍"

    description = input("\nEnter task description:\n")
    code = input("\nEnter correct Python solution:\n")

    new_doc = [{
        "task_id": f"user_{hash(query)}",
        "prompt": description,
        "solution": code
    }]

    index_data(new_doc)

    print("Updating knowledge base...")

    db = create_vector_store()

    return "Thanks! I learned something new ✅"


# ================================
# Explain
# ================================

def explain(query: str):

    load_explain_components()

    response = explain_chain.invoke(
        {"question": query}
    )["text"]

    return response.strip()


# ================================
# Generate (RAG + Unknown detection)
# ================================

def generate(query: str):

    load_generate_components()

    docs = db.similarity_search(query, k=5)

    if len(docs) == 0:
        return handle_unknown(query)

    unique_docs = []
    seen = set()

    for d in docs:
        content = d.page_content.strip()

        if content not in seen:
            unique_docs.append(d)
            seen.add(content)

    return generate_code(query, unique_docs, gen_tokenizer, gen_model)


# ================================
# Tests
# ================================

if RUN_RAG_TEST:
    query = "write a python function to check if a number is prime"
    print(generate(query))
    exit()


if RUN_ROUTER_TEST:

    queries = [
        "what is a prime number?",
        "explain recursion in simple terms",
        "write a python function to check prime",
        "generate a function to sort a list",
    ]

    for q in queries:
        print("Query:", q)
        print("Intent:", router.route(q))
        print("-" * 50)

    exit()


if RUN_EXPLAIN_TEST:

    print(explain("What is a prime number?"))
    print("-----")

    print(explain("Give an example"))
    print("-----")

    print(explain("Why is it important?"))

    exit()


# ================================
# Full System 🚀
# ================================

if __name__ == "__main__":
    if RUN_FULL_SYSTEM:

        print("\n💡 AI-CodeGen-Assistant is ready! Type 'exit' to stop.\n")

        while True:

            user_query = input("You: ")

            if user_query.lower() == "exit":
                print("Goodbye 👋")
                break

            intent = router.route(user_query)

            print(f"[Router → {intent}]")

            if intent == "explain":
                answer = explain(user_query)

            elif intent == "generate":
                answer = generate(user_query)

            else:
                answer = "I am not sure how to handle that."

            print("\nAI:\n", answer)
            print("\n" + "=" * 60 + "\n")