def generate_code(query, retrieved_docs, tokenizer, model):

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
### Instruction:
You are an expert Python developer.

Use the reference tasks below to write a correct solution.

### Similar Tasks:
{context}

### New Task:
{query}

### Response:
```python
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.1,
        do_sample=False
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "```python" in decoded:
        decoded = decoded.split("```python")[-1].split("```")[0]

    return decoded.strip()