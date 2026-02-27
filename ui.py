import gradio as gr
from main import explain, generate, router


def respond(message, history):

    intent = router.route(message)

    if intent == "explain":
        answer = explain(message)
    elif intent == "generate":
        answer = generate(message)
    else:
        answer = "I am not sure how to handle that."

    history.append({"role": "user", "content": message})
    history.append({
        "role": "assistant",
        "content": f"Mode: {intent}\n\n{answer}"
    })

    return history, ""


with gr.Blocks() as demo:

    gr.Markdown("## AI CodeGen Assistant")

    chatbot = gr.Chatbot(height=500)

    msg = gr.Textbox(
        placeholder="Ask something...",
        label="Your message"
    )

    clear = gr.Button("Clear")

    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: [], None, chatbot)

demo.launch()