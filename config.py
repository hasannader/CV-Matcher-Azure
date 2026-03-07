import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient

# ENV
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "hr_cv_collection"
EMBEDDING_DIM = 1536  # text-embedding-3-small

# Initialize Models
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

llm = ChatOpenAI(
    # model="gpt-4.1",
    model="gpt-4.1-mini",
    # model="gpt-4o-mini",
    temperature=0
)

# Qdrant Cloud Setup
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120
)
