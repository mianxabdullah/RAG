import gradio as gr
import numpy as np
import groq
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import PyPDF2
import pdfplumber

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not set. Please set it as an environment variable or in Hugging Face Spaces secrets.")
    client = None
else:
    client = groq.Groq(api_key=GROQ_API_KEY)

print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

documents_data = []
chat_history = []

class Document:
    def __init__(self, filename: str, pages: List[Dict], chunks: List[Dict], embeddings: np.ndarray):
        self.filename = filename
        self.pages = pages  
        self.chunks = chunks  
        self.embeddings = embeddings 

def extract_text_from_pdf(pdf_file) -> List[Dict]:
    pages = []
    try:
        with pdfplumber.open(pdf_file.name) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({
                        'page_num': page_num,
                        'text': text.strip()
                    })
    except Exception as e:
        print(f"Error with pdfplumber: {e}, trying PyPDF2...")
        with open(pdf_file.name, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({
                        'page_num': page_num,
                        'text': text.strip()
                    })
    return pages


