import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from retriever import search

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def ask(query: str, doc_name: str) -> dict:
    """Takes a question, finds relevant chunks, asks Gemini to answer"""
    
    # Step 1: Get relevant chunks
    results = search(query, doc_name, top_k=5)
    
    # Step 2: Combine chunks into one context block
    context = "\n\n".join([r["chunk"] for r in results])
    
    # Step 3: Build the prompt
    prompt = f"""You are a helpful document assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I couldn't find that in the document."
Always be clear and concise.

Context:
{context}

Question: {query}

Answer:"""

    # Step 4: Send to Gemini and get answer
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,  # low = more factual, less creative
            max_output_tokens=500
        )
    )
    
    return {
        "answer": response.text,
        "sources": [r["chunk"][:150] + "..." for r in results]  # first 150 chars of each source
    }


# Test it
if __name__ == "__main__":
    query = "What is ethical hacking?"
    doc_name = "EH_Unit1"
    
    print(f"Question: {query}\n")
    result = ask(query, doc_name)
    
    print("Answer:")
    print(result["answer"])
    
    print("\nSources used:")
    for i, source in enumerate(result["sources"]):
        print(f"  [{i+1}] {source}")