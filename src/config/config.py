# src/config/config.py
"""
Enhanced Configuration module for Advanced Agentic RAG system
Optimized for biotech and microbe-based therapeutics research
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()

class Config:
    """Configuration for Advanced RAG system"""

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Embedding model - using best performing model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    
    # Chat LLM model - using GPT-4 for best reasoning
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3")) #0.1
    
    # Document processing - optimized for scientific papers
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Vector DB storage
    VECTOR_PERSIST_DIR = os.getenv("VECTOR_PERSIST_DIR", "vector_db")
    
    # Asset extraction directories
    ASSETS_DIR = os.getenv("ASSETS_DIR", "extracted_assets")
    
    # Retrieval settings
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))
    
    # Memory settings
    MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "8"))

    @classmethod
    def get_llm(cls):
        """Return a LangChain Chat model instance"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")
        return ChatOpenAI(
            model_name=cls.LLM_MODEL_NAME, 
            temperature=cls.LLM_TEMPERATURE,
            openai_api_key=cls.OPENAI_API_KEY,
            max_tokens=4000
        )

    @classmethod
    def get_embedding(cls):
        """Return OpenAI Embeddings instance"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")
        return OpenAIEmbeddings(
            model=cls.EMBEDDING_MODEL,
            openai_api_key=cls.OPENAI_API_KEY
        )