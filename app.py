import streamlit as st

from cv_pipeline import build_vectorstore
from retrieval import build_retriever, run_guard, run_rag



st.set_page_config(page_title="📄 HR CV System")
st.title("Chat with CVs")


# Sidebar — CV Upload
with st.sidebar:
    st.header("📂 CV Upload Panel")

    uploaded_files = st.file_uploader(
        "Upload exactly 5 CV PDFs",
        type="pdf",
        accept_multiple_files=True,
    )
    if uploaded_files:
        if len(uploaded_files) != 5:
            st.error("❌ You must upload exactly 5 CVs.")
            st.stop()

        st.success("✅ 5 CVs uploaded successfully.")



# Session State Init
if "cv_loaded" not in st.session_state:
    st.session_state.cv_loaded = False

if not st.session_state.cv_loaded:
    st.session_state.rebuild_collection = True

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# Build / Rebuild Vectorstore
if uploaded_files and len(uploaded_files) == 5:

    if st.session_state.get("rebuild_collection", True):

        vectorstore, total_chunks = build_vectorstore(uploaded_files)

        st.session_state.total_chunks = total_chunks
        st.session_state.vectorstore = vectorstore
        st.session_state.cv_loaded = True
        st.session_state.rebuild_collection = False

    else:
        vectorstore = st.session_state.vectorstore



# Sidebar — Metrics
with st.sidebar:
    st.markdown("### 📊 System Stats")

    total_cvs = len(uploaded_files) if uploaded_files else 0
    st.metric("Total CVs", total_cvs)
    st.metric("Total Chunks", st.session_state.get("total_chunks", 0))


# Build Retriever (if vectorstore ready)
if "vectorstore" in st.session_state:
    vectorstore = st.session_state.vectorstore
    multi_query_retriever = build_retriever(vectorstore)



# Render Chat History
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):

        if chat["role"] == "user":
            st.markdown(chat["message"])

        elif chat["role"] == "assistant":

            is_valid = chat.get("is_valid", False)
            retrieved_docs = chat.get("retrieved_docs", [])
            query_list = chat.get("multi_queries", [])

            # Metadata FIRST (only if valid and results exist)
            if is_valid and len(retrieved_docs) > 0:
                st.caption("🧠 CVs Analyzed")
                st.caption(f"🔎 Retrieved {len(retrieved_docs)} relevant chunks")

                if query_list:
                    with st.expander("🧠 Generated Multi-Queries"):
                        for q in query_list:
                            st.write(q)

            # Answer SECOND
            st.markdown(chat["message"])

            # Chunks LAST
            if is_valid and len(retrieved_docs) > 0:
                with st.expander("📚 Retrieved Chunks Used"):
                    for i, doc in enumerate(retrieved_docs):
                        st.write(f"Chunk {i+1} | File: {doc.metadata.get('file_name')}")
                        st.write(doc.page_content)
                        st.write("------")



# Chat Input & Response Generation
query = st.chat_input("Ask a question about candidates...")

if query and "vectorstore" in st.session_state:

    # Save and render user message
    st.session_state.chat_history.append({"role": "user", "message": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Analyzing CVs..."):

            guard_check = run_guard(query)

            if guard_check == "INVALID_SCOPE":
                response = "This system is designed exclusively for analyzing the uploaded CVs."
                retrieved_docs = []
                query_list = []
                is_valid = False

            elif guard_check == "INVALID_JOB_TITLE":
                response = (
                    "The specified job title does not appear to be a clearly defined and "
                    "recognized professional role. Please clarify or provide a standard industry job title."
                )
                retrieved_docs = []
                query_list = []
                is_valid = False

            else:
                response, retrieved_docs, query_list = run_rag(query, multi_query_retriever)
                is_valid = True

        # Render metadata FIRST
        if is_valid and len(retrieved_docs) > 0:
            st.caption("🧠 CVs Analyzed")
            st.caption(f"🔎 Retrieved {len(retrieved_docs)} relevant chunks")

            if query_list:
                with st.expander("🧠 Generated Multi-Queries"):
                    for q in query_list:
                        st.write(q)

        # Answer AFTER metadata
        st.markdown(response)

        # Chunks LAST
        if is_valid and len(retrieved_docs) > 0:
            with st.expander("📚 Retrieved Chunks Used"):
                for i, doc in enumerate(retrieved_docs):
                    st.write(f"Chunk {i+1} | File: {doc.metadata.get('file_name')}")
                    st.write(doc.page_content)
                    st.write("------")

    # Save assistant message to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "message": response,
        "retrieved_docs": retrieved_docs,
        "retrieval_count": len(retrieved_docs),
        "multi_queries": query_list,
        "is_valid": is_valid
    })