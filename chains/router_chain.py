# import torch

# def route_query(query, tokenizer, model):

#     prompt = f"""
# You are a strict intent classifier that outputs only one word.

# Classify the user request into one of these labels:

# explain → user asks for explanation or concept
# generate → user asks to write code or a function

# Examples:

# User: what is a prime number?
# Label: explain

# User: explain recursion in simple terms
# Label: explain

# User: write a python function to check prime
# Label: generate

# User: generate a function to sort a list
# Label: generate

# Now classify:

# User: {query}
# Label:
# """
    
# ## Due to hallucination issues, I give examples in prompt (Few shot prompting) to guide the model to give the correct response.

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=3,
#             do_sample=False
#         )

#     new_tokens = output[0][inputs["input_ids"].shape[-1]:]

#     response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

#     if "generate" in response:
#         return "generate"
#     elif "explain" in response:
#         return "explain"

#     return "explain"


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticRouter:

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.labels = {
            "explain": "explain a concept or definition",
            "generate": "write a function or generate code"
        }

        self.label_embeddings = self.model.encode(list(self.labels.values()))

    def route(self, query):

        query_embedding = self.model.encode([query])

        scores = cosine_similarity(query_embedding, self.label_embeddings)[0]

        return list(self.labels.keys())[scores.argmax()]