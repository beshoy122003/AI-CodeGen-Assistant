from datasets import load_dataset

def load_humaneval():
    dataset = load_dataset("openai/openai_humaneval")

    data = []
    for i in dataset["test"]:
        data.append({
            "task_id": i["task_id"],
            "prompt": i["prompt"],
            "solution": i["canonical_solution"]
        })

    return data