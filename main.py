# main.py
"""
Command-line interface for Advanced RAG system
Use this for testing or batch processing
"""

import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

def build_system(data_dir: str = None, reindex: bool = False):
    """
    Build the complete RAG system
    
    Args:
        data_dir: Directory containing PDFs (if using path-based loading)
        reindex: Whether to force reindexing
        
    Returns:
        Dictionary with system components
    """
    print("=" * 80)
    print("🧬 Initializing Jivanu Advanced RAG System")
    print("=" * 80)
    print()
    
    # Initialize components
    print("📦 Loading configuration...")
    llm = Config.get_llm()
    
    print("📄 Initializing document processor...")
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    print("🗄️ Setting up vector store...")
    vector_store = VectorStore(persist_directory=Config.VECTOR_PERSIST_DIR)
    
    # Load or create vectorstore
    if not reindex:
        try:
            print(f"🔍 Attempting to load existing vector store from {Config.VECTOR_PERSIST_DIR}...")
            vector_store.load_vectorstore()
            print("✅ Loaded existing vector store")
            print(f"📚 Indexed files: {len(vector_store.indexed_files)}")
        except Exception as e:
            print(f"⚠️ Could not load existing store: {e}")
            reindex = True
    
    if reindex and data_dir:
        print(f"\n📚 Processing documents from: {data_dir}")
        print("This will extract text, images, and tables from all PDFs...")
        
        documents = doc_processor.process_sources(
            sources=[data_dir],
            include_images=True,
            include_tables=True
        )
        
        print(f"\n✅ Processed {len(documents)} document chunks")
        
        print("\n🔨 Creating vector store...")
        vector_store.create_vectorstore(documents)
        print("✅ Vector store created and persisted")
    
    # Build graph
    print("\n🔗 Building LangGraph workflow...")
    graph_builder = GraphBuilder(
        retriever=vector_store.get_retriever(),
        llm=llm
    )
    graph_builder.build()
    print("✅ Graph built successfully")
    
    print("\n" + "=" * 80)
    print("🎉 System ready for research queries!")
    print("=" * 80)
    print()
    
    return {
        "llm": llm,
        "processor": doc_processor,
        "vector_store": vector_store,
        "graph_builder": graph_builder
    }

def interactive_mode(system):
    """Run interactive question-answer loop"""
    print("\n🤖 Interactive Research Assistant Mode")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'save' to save the last response")
    print("Type 'stats' to see system statistics")
    print("-" * 80)
    print()
    
    chat_history = []
    last_response = None
    
    while True:
        try:
            question = input("🧬 Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == 'stats':
                stats = system["vector_store"].get_collection_stats()
                print("\n📊 System Statistics:")
                print(f"  Status: {stats.get('status')}")
                print(f"  Indexed files: {stats.get('indexed_files_count', 0)}")
                print(f"  Document chunks: {stats.get('document_count', 0)}")
                print(f"  Conversation turns: {len(chat_history)}")
                print()
                continue
            
            if question.lower() == 'save' and last_response:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"response_{timestamp}.json"
                
                with open(filename, 'w') as f:
                    json.dump(last_response, f, indent=2, default=str)
                
                print(f"💾 Response saved to {filename}")
                continue
            
            # Run query
            print("\n🔍 Researching...\n")
            
            response, chat_history = system["graph_builder"].run_with_memory(
                question=question,
                chat_history=chat_history
            )
            
            last_response = response
            
            # Display results
            print("=" * 80)
            print("💡 ANSWER:")
            print("-" * 80)
            print(response["answer"])
            print()
            
            if response.get("reasoning"):
                print("🧠 REASONING:")
                print("-" * 80)
                print(response["reasoning"])
                print()
            
            if response.get("hypothesis"):
                print("🧪 HYPOTHESIS:")
                print("-" * 80)
                print(response["hypothesis"])
                print()
            
            if response.get("suggestions"):
                print("📋 FOLLOW-UP SUGGESTIONS:")
                print("-" * 80)
                for i, sugg in enumerate(response["suggestions"], 1):
                    print(f"{i}. {sugg}")
                print()
            
            if response.get("confidence_score"):
                confidence_pct = int(response["confidence_score"] * 100)
                print(f"📊 Confidence: {confidence_pct}%")
                print()
            
            if response.get("sources"):
                print(f"📚 SOURCES ({len(response['sources'])} documents):")
                print("-" * 80)
                for src in response["sources"][:3]:
                    print(f"  [{src.get('index')}] {src.get('pdf_name', 'Unknown')} (Page {src.get('page', 'N/A')})")
                print()
            
            print("=" * 80)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print()

def run_single_query(system, question: str, save: bool = False):
    """Run a single query and optionally save the result"""
    print(f"\n🔍 Processing query: {question}\n")
    
    response = system["graph_builder"].run(question, chat_history=[])
    
    # Display results
    print("=" * 80)
    print("💡 ANSWER:")
    print("-" * 80)
    print(response["answer"])
    print()
    
    if response.get("reasoning"):
        print("🧠 REASONING:")
        print("-" * 80)
        print(response["reasoning"])
        print()
    
    if response.get("hypothesis"):
        print("🧪 HYPOTHESIS:")
        print("-" * 80)
        print(response["hypothesis"])
        print()
    
    if response.get("suggestions"):
        print("📋 SUGGESTIONS:")
        print("-" * 80)
        for i, sugg in enumerate(response["suggestions"], 1):
            print(f"{i}. {sugg}")
        print()
    
    print("=" * 80)
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"response_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(response, f, indent=2, default=str)
        
        print(f"\n💾 Response saved to {filename}")
    
    return response

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Jivanu Advanced RAG System")
    parser.add_argument("--data-dir", type=str, help="Directory containing PDFs")
    parser.add_argument("--reindex", action="store_true", help="Force reindex all documents")
    parser.add_argument("--query", type=str, help="Run a single query")
    parser.add_argument("--save", action="store_true", help="Save response to JSON file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Build system
    system = build_system(data_dir=args.data_dir, reindex=args.reindex)
    
    # Run mode
    if args.query:
        # Single query mode
        run_single_query(system, args.query, save=args.save)
    elif args.interactive:
        # Interactive mode
        interactive_mode(system)
    else:
        # Default: show help and stats
        print("\n💡 Usage Options:")
        print("  --interactive     : Start interactive Q&A mode")
        print("  --query 'text'    : Ask a single question")
        print("  --save            : Save response to JSON")
        print("  --data-dir path   : Specify PDF directory")
        print("  --reindex         : Force reindex all documents")
        print("\n📊 Current System Status:")
        stats = system["vector_store"].get_collection_stats()
        print(f"  Indexed files: {stats.get('indexed_files_count', 0)}")
        print(f"  Document chunks: {stats.get('document_count', 0)}")
        print("\n💡 TIP: Use 'streamlit run streamlit_app.py' for web interface")