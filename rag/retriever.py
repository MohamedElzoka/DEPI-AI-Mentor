"""
RAG (Retrieval-Augmented Generation) pipeline.

Responsibilities:
- Load documents from the knowledge base directory
- Split documents into chunks
- Generate embeddings and store in ChromaDB
- Retrieve relevant chunks given a query
"""

import os
from typing import List

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import settings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "depi_knowledge_base"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ---------------------------------------------------------------------------
# Singleton embedding model
# ---------------------------------------------------------------------------

def get_embeddings() -> OpenAIEmbeddings:
    """Return a shared OpenAI embeddings instance."""
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_text_documents(data_dir: str = DATA_DIR) -> List[Document]:
    """Load all .txt files from the data directory."""
    documents = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            # Tag each doc with its source filename
            for doc in docs:
                doc.metadata["source"] = filename
            documents.extend(docs)
    return documents


def load_pdf_documents(data_dir: str = DATA_DIR) -> List[Document]:
    """Load all .pdf files from the data directory."""
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            documents.extend(docs)
    return documents


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        separators=["\n---\n", "\n\n", "\n", " "],
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# Vector store operations
# ---------------------------------------------------------------------------

def build_vector_store(persist_dir: str = settings.CHROMA_PERSIST_DIR) -> Chroma:
    """Load documents, split, embed, and persist to ChromaDB. Returns the store."""
    os.makedirs(persist_dir, exist_ok=True)

    print("[RAG] Loading documents...")
    text_docs = load_text_documents()
    pdf_docs = load_pdf_documents()
    all_docs = text_docs + pdf_docs
    print(f"[RAG] Loaded {len(all_docs)} source documents.")

    print("[RAG] Splitting documents into chunks...")
    chunks = split_documents(all_docs)
    print(f"[RAG] Created {len(chunks)} chunks.")

    print("[RAG] Embedding and storing in ChromaDB...")
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    print(f"[RAG] Vector store built and persisted to '{persist_dir}'.")
    return vector_store


def load_vector_store(persist_dir: str = settings.CHROMA_PERSIST_DIR) -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def get_or_build_vector_store(persist_dir: str = settings.CHROMA_PERSIST_DIR) -> Chroma:
    """Load the store if it exists; otherwise build it from scratch."""
    index_file = os.path.join(persist_dir, "chroma.sqlite3")
    if os.path.exists(index_file):
        print("[RAG] Loading existing vector store...")
        return load_vector_store(persist_dir)
    return build_vector_store(persist_dir)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RAGRetriever:
    """Retrieves relevant document chunks for a given query."""

    def __init__(self, persist_dir: str = settings.CHROMA_PERSIST_DIR):
        self.vector_store = get_or_build_vector_store(persist_dir)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.RAG_TOP_K},
        )

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """Return top-k relevant documents for the query."""
        if k:
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k},
            )
            return retriever.invoke(query)
        return self.retriever.invoke(query)

    def retrieve_as_text(self, query: str, k: int = None) -> str:
        """Return retrieved chunks concatenated as a single context string."""
        docs = self.retrieve(query, k)
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
        return "\n\n".join(parts)

    def retrieve_as_list(self, query: str, k: int = None) -> List[dict]:
        """Return retrieved chunks as a list of dicts with source and text keys."""
        docs = self.retrieve(query, k)
        return [
            {"source": doc.metadata.get("source", "Unknown"), "text": doc.page_content}
            for doc in docs
        ]
