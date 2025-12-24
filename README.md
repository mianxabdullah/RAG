# 📚 RAG-Based Chatbot with PDF Support

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload multiple PDF documents and ask questions based on their content. Built with Gradio, Groq LLM, and sentence-transformers.

## Features

### Base Requirements ✅
- **Multiple PDF Upload**: Upload and process multiple PDF files simultaneously
- **Text Extraction**: Extracts text from all pages of uploaded PDFs
- **Semantic Chunking**: Splits content into meaningful chunks with overlap
- **Vector Similarity Search**: Retrieves relevant chunks using cosine similarity
- **Groq LLM Integration**: Uses Llama 3.1 8B model for answer generation
- **Gradio UI**: User-friendly interface for interaction

### Enhancements Implemented 🚀

1. **Sentence Transformers for Embeddings**: Uses `all-MiniLM-L6-v2` model instead of TF-IDF for better semantic understanding
2. **Conversational Memory/History**: Maintains chat history for context-aware responses
3. **Source References**: Displays source documents and page numbers for each answer with relevance scores

## How It Works

### RAG Architecture

**Retrieval-Augmented Generation (RAG)** combines information retrieval with language model generation:

1. **Document Processing**:
   - PDFs are uploaded and text is extracted from all pages
   - Text is split into semantic chunks (500 tokens with 50 token overlap)
   - Each chunk is embedded using sentence transformers

2. **Retrieval Phase**:
   - User query is embedded using the same sentence transformer model
   - Cosine similarity is calculated between query and all document chunks
   - Top-k most relevant chunks are retrieved

3. **Augmentation Phase**:
   - Retrieved chunks are formatted with source information
   - Context is combined with conversation history (last 3 exchanges)
   - Prompt is constructed with context and question

4. **Generation Phase**:
   - Groq LLM generates answer based on retrieved context
   - Source references are appended to the answer
   - Chat history is updated for future context

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <https://github.com/mianxabdullah/RAG.git>
cd RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Groq API key:
```bash
export GROQ_API_KEY="your-api-key-here"
```

4. Run the application:
```bash
python app.py
```

### Hugging Face Spaces Deployment

1. Push your code to a Hugging Face Space
2. Add `GROQ_API_KEY` as a secret in Space settings
3. The app will automatically deploy

## Usage

1. **Upload PDFs**: Use the file uploader to select one or more PDF files
2. **Process Documents**: Click "Process PDFs" to extract and index content
3. **Ask Questions**: Type your question in the chat input and press Enter or click Send
4. **View Sources**: Each answer includes source document names, page numbers, and relevance scores

## Project Structure

```
RAG/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── RAG Assignment.pdf # Assignment document
```

## Dependencies

- **gradio**: Web UI framework
- **PyPDF2**: PDF text extraction (fallback)
- **pdfplumber**: Primary PDF text extraction tool
- **sentence-transformers**: Semantic embeddings
- **scikit-learn**: Cosine similarity calculation
- **numpy**: Numerical operations
- **groq**: Groq API client for LLM

## Model Information

- **Embedding Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **LLM**: `llama-3.1-8b-instant` (Groq)

## Challenges Faced

1. **PDF Text Extraction**: Some PDFs use image-based content or complex layouts. Implemented dual extraction methods (pdfplumber + PyPDF2) for better compatibility.

2. **Chunking Strategy**: Balancing chunk size for context preservation while maintaining retrieval precision. Used sentence-based chunking with overlap.

3. **Memory Management**: Large PDFs can generate many chunks. Implemented efficient numpy array storage for embeddings.

4. **Conversation Context**: Managing chat history while keeping prompts within token limits. Limited to last 3 exchanges.

## Future Improvements

- Support for other file types (DOCX, TXT)
- Advanced chunking strategies (NLTK, LangChain splitters)
- Download chat history as JSON/CSV
- Voice input/output support
- Analytics and logging dashboard
- PDF preview functionality

## License

This project is created for educational purposes as part of an assignment.

## Author

Built as part of the RAG Assignment.

