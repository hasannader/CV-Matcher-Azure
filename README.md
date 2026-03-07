# 📄 Chat with CVs

A Streamlit-based AI Recruitment Assistant that analyzes **exactly 5
uploaded CVs** using Retrieval-Augmented Generation (RAG).

The system uses **OpenAI + LangChain + Qdrant** to retrieve relevant CV
chunks and generate structured HR decisions with strict professional
experience rules.

------------------------------------------------------------------------

## 📂 Files

-   `app.py` → Streamlit UI (main app)
-   `cv_pipeline.py` → PDF processing, chunking, vector storage
-   `retrieval.py` → Multi-query retrieval + RAG pipeline
-   `prompts.py` → Guardrails + HR evaluation rules
-   `config.py` → Models, embeddings, Qdrant setup
-   `requirements.txt` → Project dependencies

------------------------------------------------------------------------

## 🧪 Setup

### 1️⃣ Create Virtual Environment

``` bash
python -m venv venv
```

### 2️⃣ Activate Environment


``` bash
venv\Scripts\activate
```

### 3️⃣ Install Requirements

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🔐 Environment Variables

Create a `.env` file in the root folder:

``` env
OPENAI_API_KEY=your_openai_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

------------------------------------------------------------------------

## ▶️ Run the App

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 🧠 How It Works

1️⃣ Upload exactly **5 CV PDFs**\
2️⃣ CVs are converted to markdown and chunked\
3️⃣ Chunks are embedded and stored in Qdrant\
4️⃣ MultiQueryRetriever generates alternative queries\
5️⃣ Relevant chunks are retrieved\
6️⃣ LLM generates a structured HR response

------------------------------------------------------------------------

