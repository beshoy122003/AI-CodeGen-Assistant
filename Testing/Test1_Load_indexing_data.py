from data.humaneval_loader import load_humaneval
from vectordb.chroma_client import index_data

print("Loading data...")
data = load_humaneval()

print("Indexing data into Chroma DB...")
index_data(data)

print("Done ya man")