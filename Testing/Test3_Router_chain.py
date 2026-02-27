## Test3: Testing the router chain
from models.router_llm import load_router_llm
from chains.router_chain import SemanticRouter

router = SemanticRouter()

print("Router Ready")
print("=" * 50)

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