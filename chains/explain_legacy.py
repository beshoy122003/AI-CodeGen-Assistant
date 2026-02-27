## I first implemented a stateless explanation module, then upgraded it to a conversational memory-aware chain.


def explain(query, tokenizer, model):

    prompt = f"Explain in simple terms:\n{query}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=150
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)