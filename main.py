from fastapi import FastAPI, UploadFile, File
import shutil
import os

app = FastAPI()

# Create a folder to save uploaded files
os.makedirs("uploads", exist_ok=True)

@app.get("/")
def home():
    return {"message": "RAG Chatbot API is running!"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    # Save the uploaded file to the uploads folder
    file_path = f"uploads/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "saved_at": file_path
    }