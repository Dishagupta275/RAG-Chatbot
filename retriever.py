import faiss
import pickle
import numpy as np
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def load_index(doc_name: str):
    """Load FAISS index and chunks from disk"""
    index = faiss.read_index(f"indexes/{doc_name}.index")
    with open(f"indexes/{doc_name}.chunks", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def get_query_embedding(query: str) -> list:
    """Convert user question into a vector"""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values

def search(query: str, doc_name: str, top_k: int = 5) -> list:
    """Find the most relevant chunks for a query"""
    index, chunks = load_index(doc_name)
    
    # Convert query to vector
    query_vector = get_query_embedding(query)
    query_array = np.array([query_vector]).astype("float32")
    
    # Search FAISS for top_k closest chunks
    distances, indices = index.search(query_array, top_k)
    
    # Return the actual text of those chunks
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk": chunks[idx],
            "score": float(distances[0][i]),
            "index": int(idx)
        })
    
    return results


# Test it
if __name__ == "__main__":
    query = "What is ethical hacking?"
    doc_name = "EH_Unit1"
    
    print(f"Query: {query}\n")
    print("Top 5 relevant chunks:\n")
    
    results = search(query, doc_name)
    for i, result in enumerate(results):
        print(f"--- Result {i+1} (score: {result['score']:.2f}) ---")
        print(result["chunk"])
        print()