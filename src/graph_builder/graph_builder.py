# src/graph_builder/graph_builder.py
"""
Enhanced Graph builder for the RAG workflow with conversation memory
"""

from typing import List, Tuple, Dict, Any
from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.node.rag_nodes import RAGNodes

class GraphBuilder:
    """Builds and manages the RAG workflow graph"""
    
    def __init__(self, retriever, llm):
        """
        Initialize graph builder
        
        Args:
            retriever: Document retriever
            llm: Language model
        """
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None
    
    def build(self):
        """Build the LangGraph workflow"""
        print("Building RAG graph...")
        
        # Create state graph
        builder = StateGraph(RAGState)
        
        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)
        
        # Define workflow
        builder.set_entry_point("retriever")
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)
        
        # Compile graph
        self.graph = builder.compile()
        
        print("Graph built successfully")
        return self.graph
    
    def run(
        self, 
        question: str, 
        chat_history: List[Tuple[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run the RAG workflow
        
        Args:
            question: User's research question
            chat_history: Previous conversation turns as (role, message) tuples
            
        Returns:
            Dictionary with answer, reasoning, hypothesis, suggestions, and sources
        """
        if self.graph is None:
            self.build()
        
        # Create initial state
        initial_state = RAGState(
            question=question,
            chat_history=chat_history or []
        )
        
        # Run graph
        print(f"\n{'='*80}")
        print(f"Processing question: {question}")
        print(f"{'='*80}\n")
        
        final_state = self.graph.invoke(initial_state)
        
        # Handle different return types from langgraph
        # Sometimes it returns a dict, sometimes a list of states
        if isinstance(final_state, dict):
            state_data = final_state
        elif isinstance(final_state, list):
            state_data = final_state[-1] if final_state else {}
        else:
            # It's a RAGState object
            state_data = {
                "answer": getattr(final_state, "answer", ""),
                "reasoning": getattr(final_state, "reasoning", ""),
                "hypothesis": getattr(final_state, "hypothesis", ""),
                "suggestions": getattr(final_state, "suggestions", []),
                "sources": getattr(final_state, "sources", []),
                "retrieved_docs": getattr(final_state, "retrieved_docs", []),
                "confidence_score": getattr(final_state, "confidence_score", 0.0),
            }
        
        # Return structured result
        result = {
            "answer": state_data.get("answer", ""),
            "reasoning": state_data.get("reasoning", ""),
            "hypothesis": state_data.get("hypothesis", ""),
            "suggestions": state_data.get("suggestions", []),
            "sources": state_data.get("sources", []),
            "retrieved_docs": state_data.get("retrieved_docs", []),
            "confidence_score": state_data.get("confidence_score", 0.0),
            "question": question
        }
        
        print(f"\n{'='*80}")
        print("Answer generated successfully")
        print(f"{'='*80}\n")
        
        return result
    
    def run_with_memory(
        self,
        question: str,
        chat_history: List[Tuple[str, str]] = None
    ) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
        """
        Run workflow and return updated chat history
        
        Args:
            question: User's question
            chat_history: Existing chat history
            
        Returns:
            Tuple of (result_dict, updated_chat_history)
        """
        result = self.run(question, chat_history)
        
        # Update chat history
        updated_history = list(chat_history or [])
        updated_history.append(("user", question))
        updated_history.append(("assistant", result["answer"]))
        
        return result, updated_history