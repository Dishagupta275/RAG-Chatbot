import fitz
import os
import faiss
import pickle
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text_from_pdf(file_path: str) -> str:
    """Opens a PDF and extracts all text from every page"""
    doc = fitz.open(file_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    doc.close()
    return full_text


def split_into_chunks(text: str) -> list:
    """Splits large text into smaller overlapping chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


def get_embedding(text: str) -> list:
    """Convert a text chunk into a vector using Gemini"""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    return result.embeddings[0].values

def build_faiss_index(chunks: list, doc_name: str):
    """Generate embeddings for all chunks and save to FAISS"""
    print(f"Generating embeddings for {len(chunks)} chunks...")

    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"  Embedding chunk {i+1}/{len(chunks)}...", end="\r")
        
        # Retry up to 5 times if rate limited
        for attempt in range(5):
            try:
                embedding = get_embedding(chunk)
                embeddings.append(embedding)
                time.sleep(3)  # wait 3 seconds between each call
                break  # success, move to next chunk
            except Exception as e:
                if "429" in str(e):
                    wait = (attempt + 1) * 10  # wait 10s, 20s, 30s...
                    print(f"\n  Rate limited. Waiting {wait} seconds...")
                    time.sleep(wait)
                else:
                    raise e  # different error, stop

    print(f"\nAll embeddings generated!")

    # Create FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array(embeddings).astype("float32")
    index.add(vectors)

    # Save index and chunks to disk
    os.makedirs("indexes", exist_ok=True)
    faiss.write_index(index, f"indexes/{doc_name}.index")
    with open(f"indexes/{doc_name}.chunks", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved to indexes/{doc_name}.index")
    return index, chunks
# Test it
if __name__ == "__main__":
    file_path = "uploads/EH Unit 1.pdf"
    doc_name = "EH_Unit1"

    print("Step 1: Extracting text...")
    text = extract_text_from_pdf(file_path)
    print(f"Extracted {len(text)} characters")

    print("\nStep 2: Splitting into chunks...")
    chunks = split_into_chunks(text)
    print(f"Created {len(chunks)} chunks")

    print("\nStep 3: Building FAISS index...")
    build_faiss_index(chunks, doc_name)

    print("\nDone! Your document is indexed and ready to search.")