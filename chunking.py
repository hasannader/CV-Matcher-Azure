"""
Text Chunking Module for CV Matcher RAG System
Developer: Hassan

This module is responsible for splitting text into manageable chunks
for embedding and vector storage.
"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TextChunker:
    """
    Handles text chunking with configurable parameters.
    """
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_single_text(self, text: str, metadata: Dict = None) -> List[Document]:
        """
        Split a single text into chunks with metadata.
        
        Args:
            text: The raw text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of Document objects with chunks
        """
        if not text or not text.strip():
            return []
        
        # Create documents from text
        docs = self.text_splitter.create_documents([text])
        
        # Add metadata to each chunk if provided
        if metadata:
            for doc in docs:
                doc.metadata = metadata.copy()
        
        return docs
    
    def chunk_multiple_texts(self, texts: List[Dict]) -> List[Document]:
        """
        Split multiple texts into chunks, each with its own metadata.
        
        Args:
            texts: List of dictionaries with 'text' and 'metadata' keys
                   Example: [
                       {'text': 'CV content...', 'metadata': {'candidate': 'John', 'source': 'cv1.pdf'}},
                       {'text': 'CV content...', 'metadata': {'candidate': 'Jane', 'source': 'cv2.pdf'}}
                   ]
        
        Returns:
            List of all Document chunks from all texts
        """
        all_chunks = []
        
        for item in texts:
            text = item.get('text', '')
            metadata = item.get('metadata', {})
            
            chunks = self.chunk_single_text(text, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_chunk_count(self, docs: List[Document]) -> int:
        """
        Get the total number of chunks.
        
        Args:
            docs: List of Document chunks
            
        Returns:
            Number of chunks
        """
        return len(docs)
    
    def get_chunks_by_metadata(self, docs: List[Document], metadata_key: str, metadata_value: str) -> List[Document]:
        """
        Filter chunks by metadata value.
        
        Args:
            docs: List of Document chunks
            metadata_key: Metadata key to filter by
            metadata_value: Metadata value to match
            
        Returns:
            Filtered list of chunks
        """
        return [doc for doc in docs if doc.metadata.get(metadata_key) == metadata_value]


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Chunk a single text
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    
    sample_text = """
    John Smith
    Senior Python Developer
    
    Professional Summary:
    Experienced software developer with 8 years in Python development.
    Specialized in building scalable web applications and REST APIs.
    
    Skills:
    - Python, Django, Flask
    - PostgreSQL, MongoDB
    - Docker, Kubernetes
    - AWS, CI/CD
    
    Work Experience:
    Senior Developer at Tech Corp (2020-Present)
    - Led team of 5 developers
    - Built microservices architecture
    - Improved system performance by 40%
    """
    
    metadata = {'candidate_name': 'John Smith', 'source': 'john_cv.pdf'}
    chunks = chunker.chunk_single_text(sample_text, metadata)
    
    print(f"✅ Created {len(chunks)} chunks from single text")
    print(f"\nFirst chunk preview:")
    print(chunks[0].page_content[:100] + "...")
    print(f"Metadata: {chunks[0].metadata}")
    
    # Example 2: Chunk multiple texts
    texts = [
        {
            'text': sample_text,
            'metadata': {'candidate_name': 'John Smith', 'source': 'john_cv.pdf'}
        },
        {
            'text': "Jane Doe\nData Scientist\n5 years experience in ML and AI...",
            'metadata': {'candidate_name': 'Jane Doe', 'source': 'jane_cv.pdf'}
        }
    ]
    
    all_chunks = chunker.chunk_multiple_texts(texts)
    print(f"\n✅ Created {len(all_chunks)} chunks from {len(texts)} CVs")
