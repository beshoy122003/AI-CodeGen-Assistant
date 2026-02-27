from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

import os

CHROMA_PATH = "vectordb/chroma_store"

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )


def index_data(data):
    db = create_vector_store()

    documents = []

    for i in data:
        documents.append(Document(
            page_content = i["prompt"],
            metadata = {
                "task_id": i["task_id"],
                "solution": i["solution"]
            }
        ))
    
    db.add_documents(documents)