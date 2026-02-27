from langchain.memory import ConversationSummaryBufferMemory

class CleanMemory(ConversationSummaryBufferMemory):

    def save_context(self, inputs, outputs):
        cleaned_outputs = {}

        for k, v in outputs.items():

            if isinstance(v, str):

                if "AI:" in v:
                    v = v.split("AI:")[-1].strip()

                if "You are a helpful AI tutor" in v:
                    v = v.split("You are a helpful AI tutor")[-1].strip()

            cleaned_outputs[k] = v

        super().save_context(inputs, cleaned_outputs)


## Memory for Router and Explain Chain
def build_memory(llm):
    return ConversationSummaryBufferMemory(llm = llm,
                                     memory_key = "chat_history",
                                     return_messages = False,
                                     max_token_limit=2000
                                     )