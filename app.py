"""
Streamlit Application for CV Matcher RAG System
Integrates all modules: chunking, vector_database, and rag

Team Responsibilities:
- Hassan: chunking.py
- Mariam & Radwa: vector_database.py
- Mohamed & Emad: rag.py
"""

import streamlit as st
import os
import time

# Import configuration
from config import (
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, PAGE_TITLE, PAGE_ICON, LAYOUT,
    MIN_CVS, MAX_CVS, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K,
    HR_ANALYSIS_PROMPT
)

# Import utility functions
from utils import extract_pdf_text, extract_candidate_name, save_uploaded_file, clean_uploads_directory

# Import the three modules
from chunking import TextChunker
from vector_database import VectorDatabase
from rag import RAGSystem


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

if 'cv_files' not in st.session_state:
    st.session_state.cv_files = []

if 'candidate_names' not in st.session_state:
    st.session_state.candidate_names = []

if 'total_chunks' not in st.session_state:
    st.session_state.total_chunks = 0

if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function."""
    
    # Header
    st.title("📄 CV Matcher - RAG System")
    st.markdown("Upload CVs and find the best candidates for your job requirements")
    st.divider()
    
    # Sidebar - File Upload Section
    with st.sidebar:
        st.header("📤 Upload CVs")
        st.markdown(f"Upload {MIN_CVS}-{MAX_CVS} CV files in PDF format")
        
        uploaded_files = st.file_uploader(
            "Upload CVs (PDF format)",
            type="pdf",
            accept_multiple_files=True,
            help=f"Upload {MIN_CVS}-{MAX_CVS} CVs in PDF format"
        )
        
        st.divider()
        
        # Configuration Section
        st.header("⚙️ Configuration")
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=1000,
            value=CHUNK_SIZE,
            step=50,
            help="Size of text chunks for embedding"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=200,
            value=CHUNK_OVERLAP,
            step=25,
            help="Overlap between consecutive chunks"
        )
        
        retriever_k = st.slider(
            "Retrieved Chunks (K)",
            min_value=5,
            max_value=30,
            value=RETRIEVER_K,
            step=5,
            help="Number of chunks to retrieve for each query"
        )
        
        st.divider()
        
        # Process Button
        if st.button("🚀 Process CVs", type="primary", use_container_width=True):
            if not uploaded_files:
                st.error("⚠️ Please upload at least one CV file")
            elif len(uploaded_files) < MIN_CVS:
                st.error(f"⚠️ Please upload at least {MIN_CVS} CVs")
            elif len(uploaded_files) > MAX_CVS:
                st.error(f"⚠️ Maximum {MAX_CVS} CVs allowed")
            else:
                process_cvs(uploaded_files, chunk_size, chunk_overlap, retriever_k)
        
        # Clear Button
        if st.button("🗑️ Clear All", use_container_width=True):
            clear_system()
        
        st.divider()
        
        # System Status
        st.header("📊 System Status")
        if st.session_state.system_ready:
            st.success("✅ System Ready")
            st.metric("CVs Loaded", len(st.session_state.cv_files))
            st.metric("Total Chunks", st.session_state.total_chunks)
            st.metric("Candidates", len(st.session_state.candidate_names))
            
            with st.expander("View Candidates"):
                for i, name in enumerate(st.session_state.candidate_names, 1):
                    st.write(f"{i}. {name}")
        else:
            st.info("⏳ Upload CVs to get started")
    
    # Main Content Area
    if st.session_state.system_ready:
        display_query_interface()
    else:
        display_welcome_screen()


def process_cvs(uploaded_files, chunk_size, chunk_overlap, retriever_k):
    """Process uploaded CVs and build RAG system."""
    
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    try:
        # Step 1: Save uploaded files
        status_text.text("📁 Saving uploaded files...")
        progress_bar.progress(10)
        
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(uploaded_file)
            file_paths.append(file_path)
        
        st.session_state.cv_files = file_paths
        
        # Step 2: Extract text and prepare data
        status_text.text("📝 Extracting text from PDFs...")
        progress_bar.progress(25)
        
        cv_data = []
        candidate_names = []
        
        for file_path in file_paths:
            text = extract_pdf_text(file_path)
            candidate_name = extract_candidate_name(text, os.path.basename(file_path))
            candidate_names.append(candidate_name)
            
            cv_data.append({
                'text': text,
                'metadata': {
                    'candidate_name': candidate_name,
                    'source': os.path.basename(file_path)
                }
            })
        
        st.session_state.candidate_names = candidate_names
        
        # Step 3: Chunking (Hassan's module)
        status_text.text("✂️ Chunking text (Hassan's module)...")
        progress_bar.progress(40)
        
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_multiple_texts(cv_data)
        st.session_state.total_chunks = len(chunks)
        
        # Step 4: Create vector database (Mariam & Radwa's module)
        status_text.text("🗄️ Creating vector database (Mariam & Radwa's module)...")
        progress_bar.progress(60)
        
        vdb = VectorDatabase(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY)
        vdb.create_vector_store(chunks)
        
        status_text.text("🔍 Creating retriever...")
        progress_bar.progress(75)
        
        vdb.create_retriever(k=retriever_k)
        
        # Step 5: Initialize RAG system (Mohamed & Emad's module)
        status_text.text("🤖 Initializing RAG system (Mohamed & Emad's module)...")
        progress_bar.progress(90)
        
        rag = RAGSystem(
            retriever=vdb.get_retriever(),
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_api_key=AZURE_OPENAI_API_KEY,
            temperature=0.2
        )
        
        # Update the prompt with the one from config
        rag.update_prompt(HR_ANALYSIS_PROMPT)
        
        st.session_state.rag_system = rag
        st.session_state.system_ready = True
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ System ready!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.sidebar.success(f"✅ Successfully processed {len(file_paths)} CVs!")
        st.rerun()
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.sidebar.error(f"❌ Error: {str(e)}")
        st.session_state.system_ready = False


def clear_system():
    """Clear the system and uploaded files."""
    st.session_state.rag_system = None
    st.session_state.cv_files = []
    st.session_state.candidate_names = []
    st.session_state.total_chunks = 0
    st.session_state.system_ready = False
    
    # Clean uploads directory
    deleted = clean_uploads_directory()
    
    st.sidebar.success(f"✅ System cleared! ({deleted} files deleted)")
    st.rerun()


def display_welcome_screen():
    """Display welcome screen when no CVs are uploaded."""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 👋 Welcome to CV Matcher!")
        st.markdown("""
        This intelligent system helps you find the best candidates for your job requirements.
        
        **How it works:**
        1. 📤 Upload 2-5 CV files (PDF format)
        2. ⚙️ Configure chunking parameters (optional)
        3. 🚀 Click "Process CVs" to analyze
        4. 💬 Ask questions about candidates
        
        **Example queries:**
        - "Find candidates with Python and machine learning experience"
        - "Who has cloud deployment experience?"
        - "Rank candidates for a senior developer position"
        - "Which candidate has the most experience with React?"
        
        **Powered by:**
        - 🔪 Hassan's Chunking Module
        - 🗄️ Mariam & Radwa's Vector Database Module
        - 🤖 Mohamed & Emad's RAG Module
        """)
        
        st.info("👈 Start by uploading CVs in the sidebar")


def display_query_interface():
    """Display the query interface when system is ready."""
    
    st.markdown("### 💬 Ask about Candidates")
    st.markdown("Ask questions about the candidates or describe your job requirements")
    
    # Query input
    query = st.text_area(
        "Your Question",
        height=100,
        placeholder="Example: Find candidates with Python and cloud experience...",
        help="Ask about candidate skills, experience, or job requirements"
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    with col2:
        show_evidence = st.checkbox("Show Evidence", value=True)
    
    # Process query
    if search_button and query.strip():
        with st.spinner("🤖 Analyzing candidates..."):
            try:
                if show_evidence:
                    results = st.session_state.rag_system.query_with_evidence(query)
                    display_results_with_evidence(query, results)
                else:
                    answer = st.session_state.rag_system.query(query)
                    display_results(query, answer)
                    
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
    
    elif search_button and not query.strip():
        st.warning("⚠️ Please enter a question")
    
    # Display candidate ranking section
    st.divider()
    display_ranking_interface()


def display_results(query, answer):
    """Display query results without evidence."""
    
    st.markdown("### 📊 Results")
    st.markdown(f"**Query:** {query}")
    st.divider()
    
    st.markdown(answer)


def display_results_with_evidence(query, results):
    """Display query results with evidence."""
    
    st.markdown("### 📊 Results")
    st.markdown(f"**Query:** {query}")
    st.divider()
    
    # Display answer
    st.markdown("#### 🎯 Analysis")
    st.markdown(results['answer'])
    
    # Display evidence
    if results['evidence']:
        st.divider()
        st.markdown("#### 📚 Evidence from CVs")
        
        # Show top chunks grouped by candidate
        candidate_evidence = results['candidate_evidence']
        
        for candidate, chunks in candidate_evidence.items():
            with st.expander(f"📄 {candidate} ({len(chunks)} relevant chunks)"):
                for i, chunk in enumerate(chunks[:3], 1):  # Show top 3 chunks
                    st.markdown(f"**Chunk {i}:**")
                    st.info(chunk)


def display_ranking_interface():
    """Display candidate ranking interface."""
    
    st.markdown("### 🏆 Candidate Ranking")
    
    ranking_query = st.text_input(
        "Job Requirements for Ranking",
        placeholder="Example: Senior Python developer with 5+ years experience...",
        help="Describe the job requirements to rank candidates"
    )
    
    if st.button("📊 Rank Candidates", use_container_width=False):
        if ranking_query.strip():
            with st.spinner("📊 Ranking candidates..."):
                try:
                    rankings = st.session_state.rag_system.rank_candidates(ranking_query)
                    display_rankings(ranking_query, rankings)
                except Exception as e:
                    st.error(f"❌ Error ranking candidates: {str(e)}")
        else:
            st.warning("⚠️ Please enter job requirements")


def display_rankings(query, rankings):
    """Display candidate rankings."""
    
    st.markdown(f"**Requirements:** {query}")
    st.divider()
    
    if not rankings:
        st.info("No candidates found matching the requirements")
        return
    
    for i, rank in enumerate(rankings, 1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {i}. {rank['candidate']}")
            
            with col2:
                st.metric("Relevance Score", rank['relevance_score'])
            
            with st.expander("View Evidence"):
                for j, chunk in enumerate(rank['evidence_chunks'], 1):
                    st.markdown(f"**Evidence {j}:**")
                    st.info(chunk)
            
            st.divider()


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
