import gradio as gr
import numpy as np
import groq
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import PyPDF2
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity

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

def chunk_text(pages: List[Dict], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    chunks = []
    chunk_id = 0
    
    for page_info in pages:
        text = page_info['text']
        page_num = page_info['page_num']
        
        sentences = text.split('. ')
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append({
                    'text': chunk_text,
                    'page_num': page_num,
                    'chunk_id': chunk_id
                })
                chunk_id += 1
                
                overlap_words = ' '.join(current_chunk[-overlap:]).split()[:overlap]
                current_chunk = overlap_words
                current_length = len(current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunk_text = '. '.join(current_chunk)
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'page_num': page_num,
                    'chunk_id': chunk_id
                })
                chunk_id += 1
    
    return chunks

def process_pdfs(pdf_files) -> str:
    global documents_data
    if pdf_files is None or len(pdf_files) == 0:
        return "Please upload at least one PDF file."
    
    documents_data = []
    all_text = []
    
    for pdf_file in pdf_files:
        try:
            filename = pdf_file.name.split('/')[-1] if '/' in pdf_file.name else pdf_file.name.split('\\')[-1]
            print(f"Processing {filename}...")
            
            pages = extract_text_from_pdf(pdf_file)
            if not pages:
                return f"Error: Could not extract text from {filename}. The PDF might be image-based or corrupted."
    
            chunks = chunk_text(pages)
            if not chunks:
                return f"Error: Could not create chunks from {filename}."
            
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = model.encode(chunk_texts, show_progress_bar=True)
            
            doc = Document(filename, pages, chunks, embeddings)
            documents_data.append(doc)
            
            all_text.append(f"✓ {filename}: {len(pages)} pages, {len(chunks)} chunks")
            
        except Exception as e:
            error_msg = f"Error processing {pdf_file.name}: {str(e)}"
            print(error_msg)
            return error_msg
    
    summary = f"Successfully processed {len(documents_data)} PDF file(s):\n" + "\n".join(all_text)
    return summary

def retrieve_relevant_chunks(query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
    if not documents_data:
        return []
    
    query_embedding = model.encode([query])
    
    all_matches = []
    for doc in documents_data:
        similarities = cosine_similarity(query_embedding, doc.embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        for idx in top_indices:
            chunk = doc.chunks[idx]
            similarity = float(similarities[idx])
            all_matches.append(({
                'text': chunk['text'],
                'page_num': chunk['page_num'],
                'filename': doc.filename,
                'chunk_id': chunk['chunk_id']
            }, similarity))
    
    all_matches.sort(key=lambda x: x[1], reverse=True)
    return all_matches[:top_k]

def generate_answer(query: str, relevant_chunks: List[Tuple[Dict, float]], history: List) -> Tuple[str, List]:
    if not relevant_chunks:
        error_msg = "No relevant information found. Please upload PDF files first or try rephrasing your question."
        history.append((query, error_msg))
        return error_msg, history
    
    context_parts = []
    sources = []
    for chunk_info, similarity in relevant_chunks:
        context_parts.append(f"[From {chunk_info['filename']}, Page {chunk_info['page_num']}]: {chunk_info['text']}")
        sources.append({
            'filename': chunk_info['filename'],
            'page': chunk_info['page_num'],
            'similarity': f"{similarity:.2%}"
        })
    context = "\n\n".join(context_parts)
    
    history_context = ""
    if history:
        history_context = "\n\nPrevious conversation:\n"
        exchange_count = 0
        for item in history[-6:]:  
            if isinstance(item, tuple):
                user_msg, bot_msg = item
                exchange_count += 1
                history_context += f"Q{exchange_count}: {user_msg}\nA{exchange_count}: {bot_msg}\n"
                if exchange_count >= 3:
                    break
            elif isinstance(item, dict):
                continue

    prompt = f"""You are a helpful assistant that answers questions based on the provided document context.

Context from documents:
{context}
{history_context}

Question: {query}

Please answer the question based on the provided context. If the context doesn't contain enough information, say so. At the end of your answer, mention the source documents and page numbers you used.

Answer:"""
    
    if not client:
        error_msg = "Error: Groq API key not configured. Please set GROQ_API_KEY environment variable."
        history.append((query, error_msg))
        return error_msg, history
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate answers based on document context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1024
        )

        answer = chat_completion.choices[0].message.content
        sources_text = "\n\n**Sources:**\n"
        for i, source in enumerate(sources, 1):
            sources_text += f"{i}. {source['filename']} (Page {source['page']}, Relevance: {source['similarity']})\n"
        
        final_answer = answer + sources_text
        history.append((query, final_answer))

        return final_answer, history

    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        print(error_msg)
        history.append((query, error_msg))
        return error_msg, history
    
def convert_history_to_messages(history: List) -> List:
    if not history:
        return []
    messages = []
    for item in history:
        if isinstance(item, tuple) and len(item) == 2:
            user_msg, bot_msg = item
            messages.append({"role": "user", "content": str(user_msg)})
            messages.append({"role": "assistant", "content": str(bot_msg)})
        elif isinstance(item, dict) and "role" in item and "content" in item:
            messages.append(item)
    return messages

def convert_history_from_messages(history: List) -> List:
    tuples = []
    i = 0
    while i < len(history):
        if isinstance(history[i], dict):
            if history[i].get('role') == 'user' and i + 1 < len(history):
                if isinstance(history[i + 1], dict) and history[i + 1].get('role') == 'assistant':
                    tuples.append((history[i]['content'], history[i + 1]['content']))
                    i += 2
                    continue
        elif isinstance(history[i], tuple):
            tuples.append(history[i])
        i += 1
    return tuples

def chat_function(message: str, history: List) -> List:
    global documents_data
    
    history_tuples = convert_history_from_messages(history) if history else []
    if not documents_data:
        history_tuples.append((message, "Please upload PDF files first before asking questions."))
        return convert_history_to_messages(history_tuples)

    relevant_chunks = retrieve_relevant_chunks(message, top_k=3)
    
    answer, updated_history_tuples = generate_answer(message, relevant_chunks, history_tuples)
    
    return convert_history_to_messages(updated_history_tuples)
