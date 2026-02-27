# AI-CodeGen-Assistant

## Overview

AI-CodeGen-Assistant is an adaptive local AI system for Python code generation and concept explanation.
The system combines semantic routing, conversational memory, retrieval-augmented generation (RAG), and online continual learning, while dynamically switching between multiple LLMs on a single GPU.

It is designed to work fully offline using open-source models and a persistent vector database.

---

## Key Features

### 1. Semantic Intent Routing

The system classifies user queries into:

* explain → conceptual questions
* generate → code generation tasks

This is implemented using sentence embeddings and cosine similarity instead of an LLM classifier to achieve:

* deterministic behavior
* low latency
* zero GPU usage
* no hallucinated labels

---

### 2. Conversational Explain Mode

Concept explanations are handled by:

microsoft/Phi-3-mini-4k-instruct

With:

ConversationSummaryBufferMemory

This enables:

* multi-turn conversations
* context-aware answers
* bounded prompt size for local inference

---

### 3. Retrieval-Augmented Code Generation (RAG)

Code generation uses:

deepseek-ai/deepseek-coder-1.3b-instruct

Pipeline:

Query → Retrieve similar tasks from Chroma → Generate correct solution

The retrieval dataset is based on HumanEval and stored in a persistent local vector database.

---

### 4. Unknown Knowledge Detection

If the retrieval similarity score is below a defined threshold:

The system does not hallucinate.

Instead it asks the user to provide:

* task description
* correct implementation

---

### 5. Online Continual Learning

New user-provided tasks are:

* embedded
* stored in Chroma
* immediately retrievable

This allows the assistant to improve its knowledge during runtime without retraining.

---

### 6. Dynamic Multi-LLM GPU Switching

Due to VRAM limitations, only one model is loaded at a time.

The system:

* unloads the current model
* clears GPU memory
* loads the required model on demand

This enables running multiple LLMs locally on a single GPU.

---

## System Architecture

User Query
→ Semantic Router

If explain:
→ Phi-3 → Conversational Memory → Answer

If generate:
→ Retrieve from Chroma
→ Similarity check

If low similarity:
→ Ask user to teach new task
→ Store in vector DB

Else:
→ DeepSeek-Coder → Generate solution

---

## Models Used

Explain / Router LLM:
microsoft/Phi-3-mini-4k-instruct

Code Generation LLM:
deepseek-ai/deepseek-coder-1.3b-instruct

Embeddings:
sentence-transformers/all-MiniLM-L6-v2

Vector Database:
Chroma (persistent local storage)

---

## Project Structure

```
AI-CodeGen-Assistant
│
├── chains/
│   ├── explain_chain.py
│   ├── generate_chain.py
│   └── router_chain.py
│
├── models/
│   ├── router_llm.py
│   └── llm_loader.py
│
├── memory/
│   └── memory.py
│
├── vectordb/
│   └── chroma_client.py
│
├── data/
│   └── humaneval_loader.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

Create environment:

```
conda create -n rag_code python=3.10
conda activate rag_code
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Indexing the Dataset

Run once to build the vector database:

Set:

```
RUN_INDEXING = True
```

Then:

```
python main.py
```

After indexing, set it back to False.

---

## Running the System

Enable:

```
RUN_FULL_SYSTEM = True
```

Then:

```
python main.py
```

Example:

Explain recursion
write a python function to reverse a string

---

## Evaluation

### Router Accuracy

All tested queries were correctly classified into explain or generate.

### Retrieval Quality

Top-k retrieved tasks were semantically aligned with user queries.

### Code Generation

Generated solutions:

* syntactically correct
* functionally valid
* aligned with reference tasks

### Memory Coherence

Follow-up questions were answered within the same conversational context.

---

## Use Cases

* Local AI coding assistant
* AI programming tutor
* Self-improving internal knowledge base
* Offline developer support tool

---

## What Makes This System Different

This is not a static RAG pipeline.

It is a self-improving adaptive AI assistant that:

* learns new programming tasks during runtime
* updates its knowledge without retraining
* orchestrates multiple LLMs under GPU constraints

---

## Future Improvements

* Gradio or web interface
* Streaming responses
* Automated evaluation pipeline
* Multi-language code support