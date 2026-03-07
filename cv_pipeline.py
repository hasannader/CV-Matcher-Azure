import uuid # for generating unique CV IDs
import fitz
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance

from config import (
    qdrant_client,
    embeddings,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    QDRANT_URL,
    QDRANT_API_KEY,
)


def chunking_CVs(uploaded_files):
    chunks = []

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    for file in uploaded_files:
        file_id = str(uuid.uuid4())

        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        md_text = pymupdf4llm.to_markdown(doc)

        # Extract candidate name (first markdown line)
        lines = [l.strip() for l in md_text.split("\n") if l.strip()]
        first_line = lines[0].replace("#", "").strip()

        md_docs = splitter.split_text(md_text)

        for d in md_docs:

            enriched_content = f"""
Candidate: {first_line}

{d.page_content}
"""

            chunks.append(
                Document(
                    page_content=enriched_content,
                    metadata={
                        "cv_id": file_id,
                        "candidate_name": first_line,
                        "file_name": file.name,
                        **d.metadata
                    }
                )
            )

    return chunks


def build_vectorstore(uploaded_files):

    # Delete old collection
    if qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.delete_collection(COLLECTION_NAME)

    # Create new collection
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )

    # Chunk and upload
    chunks = chunking_CVs(uploaded_files)

    vectorstore = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        batch_size=16
    )

    return vectorstore, len(chunks)
