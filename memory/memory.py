from langchain.memory import ConversationSummaryBufferMemory

## Memory for Router and Explain Chain
def build_memory(llm):
    return ConversationSummaryBufferMemory(llm = llm,
                                     memory_key = "chat_history",
                                     return_messages = False,
                                     max_token_limit=1000
                                     )