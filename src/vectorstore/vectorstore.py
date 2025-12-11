# src/vectorstore/vectorstore.py
"""
Enhanced Vector store with incremental indexing support
Allows adding new documents without reindexing existing ones
"""

from typing import List, Dict, Optional, Set
from pathlib import Path
import json
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config.config import Config

class VectorStore:
    """Enhanced vector store with incremental indexing capabilities"""
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist the vector database
        """
        self.persist_directory = persist_directory or Config.VECTOR_PERSIST_DIR
        self.embedding = Config.get_embedding()
        self.vectorstore: Optional[Chroma] = None
        self.retriever = None
        
        # Track indexed files
        self.indexed_files_path = Path(self.persist_directory) / "indexed_files.json"
        self.indexed_files: Set[str] = self._load_indexed_files()
    
    def _load_indexed_files(self) -> Set[str]:
        """Load the list of already indexed files"""
        if self.indexed_files_path.exists():
            try:
                with open(self.indexed_files_path, 'r') as f:
                    data = json.load(f)
                    return set(data.get('indexed_files', []))
            except Exception as e:
                print(f"Warning: Could not load indexed files list: {e}")
        return set()
    
    def _save_indexed_files(self):
        """Save the list of indexed files"""
        try:
            self.indexed_files_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.indexed_files_path, 'w') as f:
                json.dump({
                    'indexed_files': list(self.indexed_files)
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save indexed files list: {e}")
    
    def get_indexed_files(self) -> List[str]:
        """Get list of already indexed files"""
        return sorted(list(self.indexed_files))
    
    def is_file_indexed(self, filename: str) -> bool:
        """Check if a file has already been indexed"""
        return filename in self.indexed_files
    
    def create_vectorstore(self, documents: List[Document], persist: bool = True):
        """
        Create vector store from documents (initial creation)
        
        Args:
            documents: List of Document objects to index
            persist: Whether to persist to disk
        """
        print("Creating new vector store...")
        
        if not documents:
            print("Warning: No documents provided")
            # Create empty vectorstore
            self.vectorstore = Chroma(
                embedding_function=self.embedding,
                persist_directory=self.persist_directory
            )
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": Config.RETRIEVAL_K}
            )
            return
        
        # Prepare texts and metadata
        texts = []
        metadatas = []
        new_indexed_files = set()
        
        for doc in documents:
            if doc.page_content and len(doc.page_content.strip()) > 20:
                texts.append(doc.page_content)
                
                # Ensure metadata is complete
                meta = dict(doc.metadata or {})
                meta.setdefault("source", "unknown")
                meta.setdefault("page", 0)
                meta.setdefault("type", "page_text")
                meta.setdefault("pdf_name", "unknown")
                
                metadatas.append(meta)
                
                # Track the file
                if meta.get("pdf_name") != "unknown":
                    new_indexed_files.add(meta["pdf_name"])
        
        print(f"Indexing {len(texts)} document chunks from {len(new_indexed_files)} files...")
        
        # Create Chroma vector store
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embedding,
            metadatas=metadatas,
            persist_directory=self.persist_directory
        )
        
        # Update indexed files list
        self.indexed_files.update(new_indexed_files)
        self._save_indexed_files()
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.RETRIEVAL_K}
        )
        
        if persist:
            try:
                self.vectorstore.persist()
                print(f"Vector store persisted to {self.persist_directory}")
                print(f"Indexed files: {len(self.indexed_files)}")
            except Exception as e:
                print(f"Warning: Could not persist vector store: {e}")
    
    def add_documents(self, documents: List[Document], persist: bool = True) -> int:
        """
        Add new documents to existing vector store (incremental indexing)
        
        Args:
            documents: List of new Document objects to add
            persist: Whether to persist to disk
            
        Returns:
            Number of new chunks added
        """
        if not documents:
            print("No documents to add")
            return 0
        
        # Filter out already indexed files
        new_documents = []
        new_files = set()
        skipped_files = set()
        
        for doc in documents:
            pdf_name = doc.metadata.get("pdf_name", "unknown")
            
            if pdf_name != "unknown" and self.is_file_indexed(pdf_name):
                skipped_files.add(pdf_name)
                continue
            
            if doc.page_content and len(doc.page_content.strip()) > 20:
                new_documents.append(doc)
                if pdf_name != "unknown":
                    new_files.add(pdf_name)
        
        if skipped_files:
            print(f"Skipped {len(skipped_files)} already indexed files: {', '.join(list(skipped_files)[:5])}")
        
        if not new_documents:
            print("All files already indexed. No new documents to add.")
            return 0
        
        print(f"Adding {len(new_documents)} new chunks from {len(new_files)} files...")
        
        # If vectorstore doesn't exist, create it
        if self.vectorstore is None:
            self.create_vectorstore(new_documents, persist=persist)
            return len(new_documents)
        
        # Add to existing vectorstore
        texts = [doc.page_content for doc in new_documents]
        metadatas = []
        
        for doc in new_documents:
            meta = dict(doc.metadata or {})
            meta.setdefault("source", "unknown")
            meta.setdefault("page", 0)
            meta.setdefault("type", "page_text")
            meta.setdefault("pdf_name", "unknown")
            metadatas.append(meta)
        
        # Add texts to existing vectorstore
        self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas
        )
        
        # Update indexed files list
        self.indexed_files.update(new_files)
        self._save_indexed_files()
        
        if persist:
            try:
                self.vectorstore.persist()
                print(f"Added {len(new_documents)} new chunks")
                print(f"Total indexed files: {len(self.indexed_files)}")
            except Exception as e:
                print(f"Warning: Could not persist vector store: {e}")
        
        return len(new_documents)
    
    def load_vectorstore(self):
        """Load existing vector store from disk"""
        print(f"Loading vector store from {self.persist_directory}...")
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.RETRIEVAL_K}
        )
        
        print(f"Vector store loaded successfully")
        print(f"Previously indexed files: {len(self.indexed_files)}")
        return self.vectorstore
    
    def get_retriever(self):
        """Get the retriever instance"""
        if self.retriever is None:
            raise RuntimeError("Vectorstore not initialized. Call create_vectorstore or load_vectorstore first.")
        return self.retriever
    
    def retrieve_with_metadata(self, query: str, k: int = None) -> List[Dict]:
        """
        Retrieve documents with full metadata
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of dicts with content and metadata
        """
        k = k or Config.RETRIEVAL_K
        retriever = self.get_retriever()
        
        docs = retriever.get_relevant_documents(query)
        
        results = []
        for doc in docs[:k]:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return results
    
    def search_by_type(self, query: str, doc_type: str, k: int = 5) -> List[Document]:
        """
        Search for documents of a specific type (page_text, image_ocr, table)
        
        Args:
            query: Search query
            doc_type: Type of document to filter by
            k: Number of results
            
        Returns:
            Filtered documents
        """
        if not self.vectorstore:
            raise RuntimeError("Vectorstore not initialized")
        
        docs = self.vectorstore.similarity_search(
            query,
            k=k * 3,  # Get more to filter
            filter={"type": doc_type}
        )
        
        return docs[:k]
    
    def delete_collection(self):
        """Delete the vector store collection and reset indexed files"""
        if self.vectorstore:
            try:
                self.vectorstore.delete_collection()
                print("Vector store collection deleted")
            except Exception as e:
                print(f"Error deleting collection: {e}")
            
            self.vectorstore = None
            self.retriever = None
        
        # Clear indexed files
        self.indexed_files.clear()
        if self.indexed_files_path.exists():
            self.indexed_files_path.unlink()
        
        print("Indexed files list cleared")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store"""
        if not self.vectorstore:
            return {
                "status": "not_initialized",
                "indexed_files_count": len(self.indexed_files),
                "indexed_files": self.get_indexed_files()
            }
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "status": "ready",
                "document_count": count,
                "persist_directory": self.persist_directory,
                "indexed_files_count": len(self.indexed_files),
                "indexed_files": self.get_indexed_files()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "indexed_files_count": len(self.indexed_files)
            }