from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os
from ingest import extract_text_from_pdf, split_into_chunks, build_faiss_index
from chain import ask
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app = FastAPI()

os.makedirs("uploads", exist_ok=True)
os.makedirs("indexes", exist_ok=True)


# --- Request model for chat ---
class ChatRequest(BaseModel):
    question: str
    doc_name: str


# --- Endpoints ---

@app.get("/")
def home():
    return {"message": "RAG Chatbot API is running!"}


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    # Save uploaded file
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract, chunk, embed and index
    print(f"Processing {file.filename}...")
    text = extract_text_from_pdf(file_path)
    chunks = split_into_chunks(text)
    doc_name = file.filename.replace(".pdf", "").replace(" ", "_")
    build_faiss_index(chunks, doc_name)

    return {
        "message": "File uploaded and indexed successfully",
        "filename": file.filename,
        "doc_name": doc_name,
        "chunks": len(chunks)
    }


@app.post("/chat")
def chat(request: ChatRequest):
    result = ask(request.question, request.doc_name)
    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }


@app.get("/documents")
def list_documents():
    files = os.listdir("indexes")
    # Get unique doc names from .index files
    docs = [f.replace(".index", "") for f in files if f.endswith(".index")]
    return {"documents": docs}