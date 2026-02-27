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

        self.label_embeddings = self.model.encode(list(self.labels.values()),
                                                  normalize_embeddings=True)

    def route(self, query):

        query_embedding = self.model.encode([query],
                                            normalize_embeddings=True)

        scores = cosine_similarity(query_embedding, self.label_embeddings)[0]

        return list(self.labels.keys())[scores.argmax()]
    


    ################################################################################################################################
# -----------------------------------------------------------------------------
# Why Semantic Routing instead of LLM-based intent classification?
#
# The initial implementation used a small LLM with few-shot prompting to
# classify the user intent into:
#     - explain → conceptual question
#     - generate → code generation request
#
# Although it worked, it had several practical issues:
#
# 1) Latency:
#    Running a text-generation model for every user query is expensive and slow,
#    especially in a local setup where the LLM is quantized and running on GPU.
#
# 2) Hallucination risk:
#    Even with few-shot examples, the model could sometimes output unexpected
#    text instead of a clean single-word label.
#
# 3) Overkill for a simple task:
#    Intent classification in our case is a semantic similarity problem,
#    not a generative task.
#
# To make the system faster, more stable, and lightweight, we replaced the
# LLM-based classifier with a semantic router built on sentence embeddings.
#
# How it works:
# - Both the user query and the intent descriptions are embedded using
#   a SentenceTransformer model.
# - Cosine similarity is used to select the closest intent.
#
# Advantages of this approach:
# - Runs in milliseconds (no generation step).
# - Deterministic and stable (no hallucinations).
# - Much lower GPU/CPU overhead.
# - Reuses the same embedding model used in the vector database (Chroma),
#   which keeps the system efficient and consistent.
#
# This makes the router a lightweight semantic decision layer that is
# better suited for production and real-time interaction.
#### I use all-MiniLM-L6-v2 for both Chroma embeddings and routing, which keeps the system efficient and consistent.####
# -----------------------------------------------------------------------------
################################################################################################################################