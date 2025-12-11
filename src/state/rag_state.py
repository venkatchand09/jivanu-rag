# src/state/rag_state.py
"""
Enhanced RAG State with comprehensive fields for biotech research
"""

from typing import List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class RAGState(BaseModel):
    """State container for RAG workflow"""
    
    # Input
    question: str = Field(description="User's research question")
    
    # Retrieved context
    retrieved_docs: List[Document] = Field(default_factory=list, description="Retrieved document chunks")
    
    # Generated outputs
    answer: str = Field(default="", description="Main answer from researcher perspective")
    reasoning: str = Field(default="", description="Scientific reasoning and context")
    hypothesis: str = Field(default="", description="Testable research hypothesis")
    
    # Interactive elements
    suggestions: List[str] = Field(default_factory=list, description="Follow-up questions/actions")
    sources: List[dict] = Field(default_factory=list, description="Source citations with metadata")
    
    # Conversation memory
    chat_history: List[Tuple[str, str]] = Field(
        default_factory=list, 
        description="Conversation history as (role, message) tuples"
    )
    
    # Additional metadata
    confidence_score: float = Field(default=0.0, description="Answer confidence (0-1)")
    relevant_images: List[str] = Field(default_factory=list, description="Paths to relevant extracted images")
    relevant_tables: List[str] = Field(default_factory=list, description="Paths to relevant extracted tables")
    
    class Config:
        arbitrary_types_allowed = True