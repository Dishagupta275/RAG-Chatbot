import fitz  # this is pymupdf
import os

def extract_text_from_pdf(file_path: str) -> str:
    """Opens a PDF and extracts all text from every page"""
    
    doc = fitz.open(file_path)
    full_text = ""
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    
    doc.close()
    return full_text


# Test it directly
if __name__ == "__main__":
    # Change this to match your uploaded file name
    file_path = "uploads/EH Unit 1.pdf"
    
    if os.path.exists(file_path):
        text = extract_text_from_pdf(file_path)
        print(text[:2000])  # print first 2000 characters
        print(f"\n\nTotal characters extracted: {len(text)}")
    else:
        print("File not found. Make sure you uploaded it first.")