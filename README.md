# CV Matcher RAG System - Modular Architecture

## 📁 Project Structure

```
repo/
├── chunking.py          # Hassan's module - Text chunking
├── vector_database.py   # Mariam & Radwa's module - Vector database
├── rag.py              # Mohamed & Emad's module - RAG system
├── config.py           # Configuration and prompt templates
├── utils.py            # Utility functions (PDF extraction, etc.)
├── app.py              # Streamlit web application
├── main.py             # Command-line integration script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 👥 Team Responsibilities

### Hassan - Text Chunking Module (`chunking.py`)
**Responsibility:** Split text into manageable chunks for embedding

**Key Components:**
- `TextChunker` class
- Configurable chunk size and overlap
- Handles single and multiple texts
- Adds metadata to chunks

**Input:** Raw text from CVs
**Output:** List of Document objects (chunks with metadata)

**Key Methods:**
- `chunk_single_text(text, metadata)` - Chunk one text
- `chunk_multiple_texts(texts)` - Chunk multiple texts
- `get_chunk_count(docs)` - Get total chunks

---

### Mariam & Radwa - Vector Database Module (`vector_database.py`)
**Responsibility:** Create embeddings and manage Qdrant vector store

**Key Components:**
- `VectorDatabase` class
- Azure OpenAI Embeddings
- Qdrant vector store (in-memory)
- Retriever creation

**Input:** Document chunks from chunking module
**Output:** Qdrant vector store and retriever for semantic search

**Key Methods:**
- `create_vector_store(documents)` - Build Qdrant vector database
- `create_retriever(k)` - Create retriever with k results
- `similarity_search(query, k)` - Search for similar chunks
- `get_retriever()` - Get retriever instance

---

### Mohamed & Emad - RAG Module (`rag.py`)
**Responsibility:** Query processing and LLM-based response generation

**Key Components:**
- `RAGSystem` class
- Azure OpenAI LLM (gpt-4.1-nano)
- Prompt engineering
- RAG chain construction

**Input:** Retriever from vector database module
**Output:** AI-generated answers based on CV content

**Key Methods:**
- `query(question)` - Get AI response
- `query_with_evidence(question)` - Get response with source evidence
- `rank_candidates(question)` - Rank candidates by relevance
- `update_prompt(template)` - Customize system prompt

---

## 🚀 How to-r requirements.txt
```

Or install individually:
```bash
pip install streamlit langchain langchain-google-genai langchain-community faiss-cpu python-dotenv PyPDF2
```

### 2. Set Up Environment
Create a `.env` file in the repo folder with your Azure OpenAI credentials:
```
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
```

### 3. Run the Streamlit Application (Recommended)
```bash
streamlit run app.py
```

The web interface will open in your browser where you can:
- Upload PDF CVs (2-5 files)
- Configure chunking parameters
- Ask questions about candidates
- Rank candidates based on job requirements

### 4. Run Individual Modules (For Testing)

#### Test Chunking Module (Hassan)
```bash
python chunking.py
```

#### Test Vector Database Module (Mariam & Radwa)
```bash
python vector_database.py
```

#### Test RAG Module (Mohamed & Emad)
See `main.py` for complete integration

### 5. Run Command-Line Version
```bash
python main.py
```
```

This runs the complete pipeline with sample CV data (no file uploads needed).# Test Chunking Module (Hassan)
```bash
python chunking.py
```
### Complete Pipeline:
```
1. PDF Files (Upload via Streamlit)
   ⬇
2. UTILS.PY (PDF Extraction)
   → Extracts text from PDFs
   → Extracts candidate names
   ⬇
3. CHUNKING MODULE (Hassan - chunking.py)
   → Splits text into chunks with metadata
   ⬇
4. VECTOR DATABASE MODULE (Mariam & Radwa - vector_database.py)
   → Creates embeddings using Azure OpenAI
   → Builds Qdrant vector store (in-memory)
   → Creates retriever
   ⬇
5. RAG MODULE (Mohamed & Emad - rag.py)
   → Receives query from user
   → Retrieves relevant chunks
   → Uses prompt from config.py
   → Generates AI response using LLM
   ⬇
6. STREAMLIT APP (app.py)
   → Displays results to user
   → Shows evidence and rankings

```
1. Raw CV Text
   ⬇
2. CHUNKING MODULE (Hassan)
   → Splits text into chunks with metadata
   ⬇
3. VECTOR DATABASE MODULE (Mariam & Radwa)
   → Creates embeddings with Azure OpenAI
   → Builds Qdrant vector store
   → Creates retriever
   ⬇
4. RAG MODULE (Mohamed & Emad)
   → Receives query
   → Retrieves relevant chunks
   → Generates AI response using Azure OpenAI LLM
   ⬇
5. Final Answer
```

---

## 🔧 Configuration

### Chunking Parameters
```python
chunk_size = 400        # Characters per chunk
chunk_overlap = 50      # Overlap between chunks
```

### Vector Database Parameters
```python
embedding_model = "text-embedding-ada-002"  # Azure OpenAI
retriever_k = 10        # Number of chunks to retrieve
vector_store = "Qdrant" # In-memory vector database
```

### Using Streamlit App (Easiest):
1. Run: `streamlit run app.py`
2. Upload PDF CVs through the web interface
3. Click "Process CVs"
4. Ask questions about candidates

### Using Python Code:
```python
from chunking import TextChunker
from vector_database import VectorDatabase
from rag import RAGSystem
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, HR_ANALYSIS_PROMPT

# Step 1: Chunk the CVs (Hassan)
chunker = TextChunker(chunk_size=400, chunk_overlap=50)
chunks = chunker.chunk_multiple_texts(cv_data)

# Step 2: Create vector database (Mariam & Radwa)
vdb = VectorDatabase(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY)
vdb.create_vector_store(chunks)
vdb.create_retriever(k=10)

# Step 3: Query using RAG (Mohamed & Emad)
rag = RAGSystem(vdb.get_retriever(), AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY)
rag.update_prompt(HR_ANALYSIS_PROMPT)  # Use prompt from config
response = rag.query("Find Python developers with cloud experience")
print(response)
```

---

---

## 🎯 Module Independence

Each module can be developed and tested independently:

- **Hassan** can work on chunking logic without waiting for other modules
- **Mariam & Radwa** can develop the vector database using sample chunks
- **Mohamed & Emad** can build the RAG system with a mock retriever

All modules come together in `main.py` for the complete pipeline.

---

## ✅ Testing Your Module

Each module includes a `if __name__ == "__main__"` section with test code.

Run your module directly to test:
```bash
# Hassan tests chunking
python chunking.py

# Mariam & Radwa test vector database
python vector_database.py

# Mohamed & Emad see main.py for RAG testing
python main.py
```Key Files

### config.py
Contains all configuration and the **HR_ANALYSIS_PROMPT** that the LLM uses:
- API keys and model settings
- Chunk size and retriever parameters
- **Complete prompt template with security rules**
- The RAG system uses this prompt to analyze CVs

### utils.py
Utility functions for file handling:
- `extract_pdf_text()` - Extract text from PDF files
- `extract_candidate_name()` - Extract candidate names
- `save_uploaded_file()` - Save uploaded files
- `format_docs()` - Format documents for context

### app.py (Streamlit Application)
Complete web interface featuring:
- PDF file upload (2-5 files)
- Real-time CV processing
- Query interface with evidence display
- Candidate ranking
- System status monitoring

## 📚 Dependencies by Module

### chunking.py
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
```

### vector_database.py
```🎯 Features

### Streamlit Web Application
- ✅ Upload 2-5 CV files in PDF format
- ✅ Configure chunking parameters in real-time
- ✅ Ask natural language questions about candidates
- ✅ View evidence from CVs supporting each answer
- ✅ Rank candidates based on job requirements
- ✅ System status monitoring

### Prompt Management
- ✅ Centralized prompt in `config.py` (HR_ANALYSIS_PROMPT)
- ✅ Includes security rules against prompt injection
- ✅ Question classification (relevant/irrelevant)
- ✅ Structured output format
- ✅ Can be easily modified without changing code

### Security Features
- ✅ Validates questions to prevent prompt injection
- ✅ Rejects irrelevant queries (jokes, unrelated topics)
- ✅ Sanitizes uploaded filenames
- ✅ Controlled prompt template in config file

## 📌 Notes

- All modules are designed to be modular and reusable
- Each team member can work independently on their part
- The system uses **Azure OpenAI** for embeddings and LLM (gpt-4.1-nano)
- **Qdrant** (in-memory) is used for efficient vector similarity search
- **The LLM prompt is defined in config.py for easy modification**
- Streamlit provides a user-friendly web interface

## 🔧 Troubleshooting

### Common Issues:

**"AZURE_OPENAI_ENDPOINT not found"**
- Create a `.env` file in the repo folder
- Add your Azure OpenAI credentials:
  ```
  AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
  AZURE_OPENAI_API_KEY=your_api_key
  ```

**"Module not found"**
- Run: `pip install -r requirements.txt`
- Make sure you're in the correct directory

**"Cannot open PDF file"**
- Ensure PDFs are not corrupted
- Check file permissions
- Try re-uploading the file
### rag.py
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
```

### app.py
```python
import streamlit as st
from config import HR_ANALYSIS_PROMPT  # Uses prompt from config
from utils import extract_pdf_text, save_uploaded_file
### chunking.py
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
```

### vector_database.py
```python
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import Document
```

### rag.py
```python
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
```

---

## 🎓 Learning Resources

- **LangChain Documentation:** https://python.langchain.com/
- **Qdrant Documentation:** https://qdrant.tech/documentation/
- **Azure OpenAI:** https://learn.microsoft.com/en-us/azure/ai-services/openai/
- **Azure AI Models:** https://learn.microsoft.com/en-us/azure/ai-studio/

---

## 📌 Notes

- All modules are designed to be modular and reusable
- Each team member can work independently on their part
- The system uses **Azure OpenAI** for embeddings and LLM (gpt-4.1-nano from Azure AI Models)
- **Qdrant** (in-memory mode) is used for efficient vector similarity search
- **The LLM prompt is defined in config.py for easy modification**
- Streamlit provides a user-friendly web interface

---

**Good luck with your implementation! 🚀**
