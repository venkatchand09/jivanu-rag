# streamlit_app.py
"""
Beautiful Streamlit interface with PDF upload, incremental indexing, and conversation history
"""

import streamlit as st
from pathlib import Path
import sys
import base64
import pandas as pd
from datetime import datetime
import json
import tempfile
import shutil

sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
from src.conversation.conversation_manager import ConversationManager

st.set_page_config(
    page_title="Jivanu Research Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful interface
st.markdown("""
<style>
    :root {
        --primary-color: #1e88e5;
        --secondary-color: #43a047;
        --background-color: #0e1117;
        --card-background: #262730;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
    }
    
    .answer-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
        max-height: none;
    }
    
    .hypothesis-card {
        background: linear-gradient(135deg, #f093fb15 0%, #f5576c15 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(240, 147, 251, 0.3);
    }
    
    .source-card {
        background-color: rgba(67, 160, 71, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid var(--secondary-color);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .user-message {
        background: linear-gradient(135deg, #1e88e520 0%, #1e88e510 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid var(--primary-color);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #43a04720 0%, #43a04710 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid var(--secondary-color);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed rgba(102, 126, 234, 0.5);
        margin: 1rem 0;
    }
    
    .file-list-item {
        background: linear-gradient(135deg, #43a04715 0%, #43a04710 100%);
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        border-left: 3px solid var(--secondary-color);
    }
    
    .conversation-item {
        background-color: rgba(67, 160, 71, 0.1);
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .conversation-item:hover {
        background-color: rgba(67, 160, 71, 0.2);
    }
    
    .conversation-item.active {
        background-color: rgba(102, 126, 234, 0.2);
        border-left: 4px solid var(--primary-color);
    }
    
    .message-timestamp {
        color: #888;
        font-size: 0.8em;
        margin-left: 0.5rem;
    }
    
    .suggestion-box {
        background: linear-gradient(135deg, #43a04720 0%, #43a04710 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #43a047;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if "system" not in st.session_state:
        st.session_state.system = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "uploaded_files_count" not in st.session_state:
        st.session_state.uploaded_files_count = 0
    
    # NEW: Conversation management
    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    if "show_conversation_list" not in st.session_state:
        st.session_state.show_conversation_list = True
    if "edit_title" not in st.session_state:
        st.session_state.edit_title = False

def initialize_system():
    """Initialize RAG system (without documents initially)"""
    if st.session_state.system is not None:
        return st.session_state.system
    
    with st.spinner("🔬 Initializing RAG System..."):
        llm = Config.get_llm()
        processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore(persist_directory=Config.VECTOR_PERSIST_DIR)
        
        # Try to load existing vectorstore
        try:
            vector_store.load_vectorstore()
            st.success(f"✅ Loaded existing index with {len(vector_store.indexed_files)} files")
        except Exception as e:
            st.info("📚 No existing index found. Upload PDFs to get started!")
            # Create empty vectorstore to initialize
            vector_store.create_vectorstore([])
        
        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        system = {
            "llm": llm,
            "processor": processor,
            "vector_store": vector_store,
            "graph_builder": graph_builder
        }
        
        st.session_state.system = system
        return system

def process_uploaded_files(uploaded_files, system):
    """Process uploaded PDF files and add to vector store"""
    if not uploaded_files:
        return 0, []
    
    processor = system["processor"]
    vector_store = system["vector_store"]
    
    # Create temporary directory for uploaded files
    temp_dir = Path(tempfile.mkdtemp())
    processed_files = []
    skipped_files = []
    
    try:
        # Save uploaded files temporarily
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            
            # Check if already indexed
            if vector_store.is_file_indexed(uploaded_file.name):
                skipped_files.append(uploaded_file.name)
                continue
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            processed_files.append(uploaded_file.name)
        
        if not processed_files:
            return 0, skipped_files
        
        # Process all new PDFs
        with st.spinner(f"📄 Processing {len(processed_files)} new PDFs..."):
            all_docs = processor.process_pdf_dir(
                temp_dir,
                include_images=True,
                include_tables=True
            )
        
        if not all_docs:
            st.warning("No documents extracted from uploaded files")
            return 0, skipped_files
        
        # Add to vector store (incremental)
        with st.spinner("🔍 Adding to vector database..."):
            num_added = vector_store.add_documents(all_docs, persist=True)
        
        return num_added, skipped_files
        
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🧬 Jivanu Research Assistant</h1>
        <p>Advanced RAG System for Microbe-Based Therapeutics Research</p>
    </div>
    """, unsafe_allow_html=True)

def render_conversation_sidebar():
    """Render conversation history in sidebar"""
    conv_manager = st.session_state.conversation_manager
    
    st.sidebar.markdown("## 💬 Conversations")
    
    # Stats
    stats = conv_manager.get_stats()
    st.sidebar.metric("Total Conversations", stats["total_conversations"])
    
    # New conversation button
    if st.sidebar.button("➕ New Conversation", use_container_width=True):
        # Create new conversation
        conv_id = conv_manager.create_conversation()
        st.session_state.current_conversation_id = conv_id
        st.session_state.chat_history = []
        st.session_state.query_count = 0
        st.success("✅ New conversation started!")
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Search conversations
    search_query = st.sidebar.text_input("🔍 Search conversations", "")
    
    # List conversations
    if search_query:
        conversations = conv_manager.search_conversations(search_query)
    else:
        conversations = conv_manager.list_conversations(limit=50)
    
    if not conversations:
        st.sidebar.info("No conversations yet. Start by asking a question!")
        return
    
    st.sidebar.markdown("### Recent Conversations")
    
    for conv in conversations:
        # Create a container for each conversation
        col1, col2 = st.sidebar.columns([4, 1])
        
        with col1:
            # Conversation button
            is_current = conv["id"] == st.session_state.current_conversation_id
            button_label = f"{'▶️ ' if is_current else ''}{conv['title'][:30]}"
            
            if st.button(
                button_label,
                key=f"conv_{conv['id']}",
                use_container_width=True,
                type="primary" if is_current else "secondary"
            ):
                # Load this conversation
                st.session_state.current_conversation_id = conv["id"]
                st.session_state.chat_history = conv_manager.get_conversation_history(conv["id"])
                st.rerun()
        
        with col2:
            # Delete button
            if st.button("🗑️", key=f"del_{conv['id']}", help="Delete conversation"):
                conv_manager.delete_conversation(conv["id"])
                if conv["id"] == st.session_state.current_conversation_id:
                    st.session_state.current_conversation_id = None
                    st.session_state.chat_history = []
                st.success("Deleted!")
                st.rerun()
        
        # Show metadata
        st.sidebar.caption(f"📊 {conv['query_count']} queries • {conv['updated_at'][:10]}")
        st.sidebar.markdown("---")

def render_answer_comprehensive(response):
    """Render comprehensive answer with all components"""
    
    # Main answer - ENHANCED DISPLAY
    st.markdown("### 💡 Comprehensive Answer")
    st.markdown(f"""
    <div class="answer-card">
        {response.get('answer', 'No answer available')}
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence score
    confidence = response.get('confidence_score', 0)
    if confidence > 0:
        confidence_pct = int(confidence * 100)
        st.progress(confidence, text=f"Confidence: {confidence_pct}%")
    
    # Reasoning - ALWAYS EXPANDED FOR COMPREHENSIVE VIEW
    if response.get('reasoning'):
        with st.expander("🧠 Scientific Reasoning & Deep Analysis", expanded=True):
            st.markdown(response['reasoning'])
    
    # Hypothesis
    if response.get('hypothesis'):
        st.markdown("### 🧪 Research Hypothesis")
        st.markdown(f"""
        <div class="hypothesis-card">
            <strong>Testable Hypothesis:</strong><br>
            {response['hypothesis']}
        </div>
        """, unsafe_allow_html=True)
    
    # Suggestions - SHOW ALL WITH BETTER FORMATTING
    if response.get('suggestions'):
        st.markdown("### 📋 Detailed Follow-up Suggestions")
        for idx, suggestion in enumerate(response['suggestions'], 1):
            st.markdown(f"""
            <div class="suggestion-box">
                <strong>{idx}.</strong> {suggestion}
            </div>
            """, unsafe_allow_html=True)
    
    # Sources
    if response.get('sources'):
        with st.expander("📚 Source Citations & References", expanded=False):
            for source in response['sources']:
                excerpt = source.get('excerpt', '')[:500]
                st.markdown(f"""
                <div class="source-card">
                    <strong>[{source.get('index', 'N/A')}] {source.get('pdf_name', source.get('source', 'Unknown'))}</strong><br>
                    Page: {source.get('page', 'N/A')} | Type: {source.get('type', 'page_text')}<br>
                    <em>Excerpt:</em> {excerpt}...
                </div>
                """, unsafe_allow_html=True)

def export_conversation(conversation_id: str) -> str:
    """Export conversation as markdown"""
    conv_manager = st.session_state.conversation_manager
    conversation = conv_manager.load_conversation(conversation_id)
    
    if not conversation:
        return ""
    
    markdown = f"# {conversation['title']}\n\n"
    markdown += f"**Created:** {conversation['created_at'][:16]}\n"
    markdown += f"**Updated:** {conversation['updated_at'][:16]}\n"
    markdown += f"**Queries:** {conversation['metadata']['query_count']}\n\n"
    markdown += "---\n\n"
    
    for msg in conversation['messages']:
        role_emoji = "👤" if msg['role'] == 'user' else "🤖"
        markdown += f"## {role_emoji} {msg['role'].title()}\n\n"
        markdown += f"*{msg['timestamp'][:16]}*\n\n"
        markdown += f"{msg['content']}\n\n"
        
        if msg.get('metadata') and msg['role'] == 'assistant':
            if msg['metadata'].get('hypothesis'):
                markdown += f"**Hypothesis:** {msg['metadata']['hypothesis']}\n\n"
            if msg['metadata'].get('confidence_score'):
                markdown += f"**Confidence:** {int(msg['metadata']['confidence_score']*100)}%\n\n"
        
        markdown += "---\n\n"
    
    return markdown

def render_file_upload_section(system):
    """Render the file upload section"""
    st.markdown("### 📤 Upload Research Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files. Already indexed files will be automatically skipped."
    )
    
    if uploaded_files:
        st.markdown(f"**Selected: {len(uploaded_files)} files**")
        
        # Show which files are new vs already indexed
        vector_store = system["vector_store"]
        new_files = []
        existing_files = []
        
        for file in uploaded_files:
            if vector_store.is_file_indexed(file.name):
                existing_files.append(file.name)
            else:
                new_files.append(file.name)
        
        if new_files:
            st.success(f"✅ {len(new_files)} new file(s) to be indexed")
            with st.expander("📋 New files"):
                for fname in new_files:
                    st.markdown(f"- {fname}")
        
        if existing_files:
            st.info(f"ℹ️ {len(existing_files)} file(s) already indexed (will be skipped)")
            with st.expander("📋 Already indexed"):
                for fname in existing_files:
                    st.markdown(f"- {fname}")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 Process Files", type="primary", disabled=len(new_files) == 0):
                num_added, skipped = process_uploaded_files(uploaded_files, system)
                
                if num_added > 0:
                    st.success(f"✅ Successfully added {num_added} document chunks!")
                    st.session_state.uploaded_files_count += len(new_files)
                    st.balloons()
                    st.rerun()
                elif skipped:
                    st.info(f"All {len(skipped)} files were already indexed")
                else:
                    st.warning("No documents were processed")

def render_indexed_files(system):
    """Show currently indexed files"""
    stats = system["vector_store"].get_collection_stats()
    indexed_files = stats.get('indexed_files', [])
    
    if not indexed_files:
        st.info("No files indexed yet. Upload PDFs to get started!")
        return
    
    st.markdown("### 📚 Indexed Documents")
    
    st.markdown(f"""
    <div class="info-card">
        <strong>{len(indexed_files)} files currently indexed</strong>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📋 View all indexed files"):
        for idx, fname in enumerate(indexed_files, 1):
            st.markdown(f"""
            <div class="file-list-item">
                <span style="margin-right: 1rem;">📄 {idx}.</span>
                <span>{fname}</span>
            </div>
            """, unsafe_allow_html=True)

def render_assets_viewer(system):
    """Render extracted assets (images and tables) viewer"""
    st.markdown("### 🖼️ Extracted Assets Explorer")
    
    assets_dir = Path("extracted_assets")
    if not assets_dir.exists():
        st.info("No extracted assets found. Upload and process PDFs to extract images and tables.")
        return
    
    pdf_folders = [d for d in assets_dir.iterdir() if d.is_dir()]
    
    if not pdf_folders:
        st.info("No PDF assets extracted yet.")
        return
    
    selected_pdf = st.selectbox(
        "Select PDF to view assets:",
        options=[d.name for d in pdf_folders]
    )
    
    if selected_pdf:
        pdf_path = assets_dir / selected_pdf
        
        # Display images
        img_dir = pdf_path / "images"
        if img_dir.exists():
            images = list(img_dir.glob("*.png"))
            if images:
                st.markdown("#### 📸 Extracted Images")
                cols = st.columns(3)
                for idx, img in enumerate(images):
                    with cols[idx % 3]:
                        st.image(str(img), caption=img.name, use_container_width=True)
        
        # Display tables
        tbl_dir = pdf_path / "tables"
        if tbl_dir.exists():
            tables = list(tbl_dir.glob("*.csv"))
            if tables:
                st.markdown("#### 📊 Extracted Tables")
                for tbl in tables:
                    with st.expander(f"Table: {tbl.name}"):
                        try:
                            df = pd.read_csv(tbl)
                            st.dataframe(df, use_container_width=True)
                            
                            st.download_button(
                                label="📥 Download CSV",
                                data=df.to_csv(index=False),
                                file_name=tbl.name,
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Error loading table: {e}")

def main():
    """Main application"""
    init_session_state()
    render_header()
    
    # Initialize system
    system = initialize_system()
    
    # Render conversation sidebar FIRST
    render_conversation_sidebar()
    
    # Rest of sidebar
    with st.sidebar:
        st.markdown("## ⚙️ System Control")
        
        # System stats
        stats = system["vector_store"].get_collection_stats()
        st.metric("System Status", stats.get("status", "Unknown"))
        st.metric("Indexed Files", stats.get("indexed_files_count", 0))
        st.metric("Total Chunks", stats.get("document_count", 0))
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Current Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.query_count = 0
            st.success("Chat cleared")
            st.rerun()
        
        if st.button("⚠️ Reset All Data", use_container_width=True):
            if st.checkbox("Confirm reset (will delete all indexed documents)"):
                system["vector_store"].delete_collection()
                st.session_state.chat_history = []
                st.session_state.query_count = 0
                st.success("All data reset")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "What microbes are used for therapeutic peptide delivery?",
            "Explain engineered probiotics in drug delivery",
            "Challenges in scaling microbial therapeutics?",
            "Bacterial vs viral vectors comparison",
            "Regulatory pathways for live biotherapeutics"
        ]
        
        for q in example_questions:
            if st.button(f"📌 {q[:40]}...", key=f"example_{q[:20]}", use_container_width=True):
                st.session_state.pending_question = q
    
    # Main tabs - REMOVED ASSETS TAB
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "💬 Research Query", "📜 Conversation History"])
    
    with tab1:
        render_file_upload_section(system)
        st.markdown("---")
        render_indexed_files(system)
    
    with tab2:
        st.markdown("### Ask Your Research Question")
        
        # Check if documents are indexed
        stats = system["vector_store"].get_collection_stats()
        if stats.get('indexed_files_count', 0) == 0:
            st.warning("⚠️ No documents indexed yet! Please upload PDFs in the 'Upload Documents' tab first.")
        else:
            # Handle pending question from sidebar
            default_question = ""
            if hasattr(st.session_state, 'pending_question'):
                default_question = st.session_state.pending_question
                delattr(st.session_state, 'pending_question')
            
            question = st.text_area(
                "Enter your biotech/microbiology research question:",
                value=default_question,
                height=100,
                placeholder="e.g., What are the latest advances in using engineered bacteria for cancer therapy?"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
            
            if search_clicked and question:
                # Create new conversation if needed
                if st.session_state.current_conversation_id is None:
                    conv_manager = st.session_state.conversation_manager
                    # Use first question as title
                    title = question[:50] + "..." if len(question) > 50 else question
                    conv_id = conv_manager.create_conversation(title=title)
                    st.session_state.current_conversation_id = conv_id
                
                with st.spinner("🧬 Researching your question..."):
                    graph_builder = system["graph_builder"]
                    conv_manager = st.session_state.conversation_manager
                    
                    response, updated_history = graph_builder.run_with_memory(
                        question=question,
                        chat_history=st.session_state.chat_history
                    )
                    
                    st.session_state.chat_history = updated_history
                    st.session_state.query_count += 1
                    st.session_state.last_response = response
                    
                    # Save to conversation history
                    conv_manager.add_message(
                        st.session_state.current_conversation_id,
                        "user",
                        question
                    )
                    conv_manager.add_message(
                        st.session_state.current_conversation_id,
                        "assistant",
                        response["answer"],
                        metadata={
                            "reasoning": response.get("reasoning", ""),
                            "hypothesis": response.get("hypothesis", ""),
                            "suggestions": response.get("suggestions", []),
                            "sources": response.get("sources", []),
                            "confidence_score": response.get("confidence_score", 0.0)
                        }
                    )
                
                render_answer_comprehensive(response)
                
                if st.button("💾 Save Response as JSON"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"research_response_{timestamp}.json"
                    
                    st.download_button(
                        label="📥 Download JSON",
                        data=json.dumps(response, indent=2, default=str),
                        file_name=filename,
                        mime="application/json"
                    )

    with tab3:
        st.markdown("### 📜 Conversation History")

        if st.session_state.current_conversation_id:
            conv_manager = st.session_state.conversation_manager
            conversation = conv_manager.load_conversation(st.session_state.current_conversation_id)

            if conversation:
                # Show conversation metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Title", conversation["title"][:30])
                with col2:
                    st.metric("Queries", conversation["metadata"]["query_count"])
                with col3:
                    created_date = conversation["created_at"][:10]
                    st.metric("Created", created_date)

                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✏️ Edit Title"):
                        st.session_state.edit_title = True

                with col2:
                    if st.button("📥 Export Markdown"):
                        markdown_content = export_conversation(st.session_state.current_conversation_id)
                        st.download_button(
                            label="💾 Download",
                            data=markdown_content,
                            file_name=f"conversation_{st.session_state.current_conversation_id}.md",
                            mime="text/markdown"
                        )

                with col3:
                    if st.button("🗑️ Delete Conversation"):
                        if st.checkbox("Confirm deletion"):
                            conv_manager.delete_conversation(st.session_state.current_conversation_id)
                            st.session_state.current_conversation_id = None
                            st.session_state.chat_history = []
                            st.success("Conversation deleted!")
                            st.rerun()

                # Edit title interface
                if st.session_state.get("edit_title", False):
                    new_title = st.text_input("New title:", value=conversation["title"])
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Save Title"):
                            conv_manager.update_conversation_title(
                                st.session_state.current_conversation_id,
                                new_title
                            )
                            st.session_state.edit_title = False
                            st.success("Title updated!")
                            st.rerun()
                    with col2:
                        if st.button("❌ Cancel"):
                            st.session_state.edit_title = False
                            st.rerun()

                st.markdown("---")

                # Display messages with full metadata
                messages = conv_manager.get_conversation_messages(st.session_state.current_conversation_id)

                if not messages:
                    st.info("No messages in this conversation yet.")
                else:
                    for idx, msg in enumerate(messages):
                        if msg["role"] == "user":
                            st.markdown(f"""
                            <div class="user-message">
                                <strong>👤 You</strong> <span class="message-timestamp">{msg['timestamp'][11:16]}</span><br>
                                {msg['content']}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="assistant-message">
                                <strong>🤖 Assistant</strong> <span class="message-timestamp">{msg['timestamp'][11:16]}</span><br>
                                {msg['content'][:1000]}{'...' if len(msg['content']) > 1000 else ''}
                            </div>
                            """, unsafe_allow_html=True)

                            # Show metadata if available
                            if msg.get("metadata"):
                                with st.expander(f"📊 View Full Response Details #{(idx//2) + 1}"):
                                    meta = msg["metadata"]

                                    # Full answer
                                    st.markdown("**💡 Complete Answer:**")
                                    st.write(msg['content'])

                                    # Reasoning
                                    if meta.get("reasoning"):
                                        st.markdown("**🧠 Reasoning:**")
                                        st.write(meta["reasoning"])

                                    # Hypothesis
                                    if meta.get("hypothesis"):
                                        st.markdown("**🧪 Hypothesis:**")
                                        st.info(meta["hypothesis"])

                                    # Suggestions
                                    if meta.get("suggestions"):
                                        st.markdown("**📋 Suggestions:**")
                                        for i, sugg in enumerate(meta["suggestions"], 1):
                                            st.write(f"{i}. {sugg}")

                                    # Confidence
                                    if meta.get("confidence_score"):
                                        confidence_pct = int(meta["confidence_score"] * 100)
                                        st.metric("Confidence", f"{confidence_pct}%")

                                    # Sources
                                    if meta.get("sources"):
                                        st.markdown("**📚 Sources:**")
                                        for src in meta["sources"][:5]:
                                            st.caption(f"[{src.get('index')}] {src.get('pdf_name', 'Unknown')} (Page {src.get('page', 'N/A')})")
        else:
            st.info("No active conversation. Start by asking a question in the Research Query tab!")

            # Show all conversations list
            conv_manager = st.session_state.conversation_manager
            all_conversations = conv_manager.list_conversations(limit=20)

            if all_conversations:
                st.markdown("### 📚 All Conversations")
                for conv in all_conversations:
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"**{conv['title']}**")
                            st.caption(f"📊 {conv['query_count']} queries • {conv['updated_at'][:16]}")
                        with col2:
                            if st.button("📖 Open", key=f"open_{conv['id']}"):
                                st.session_state.current_conversation_id = conv["id"]
                                st.session_state.chat_history = conv_manager.get_conversation_history(conv["id"])
                                st.rerun()
                        with col3:
                            if st.button("🗑️", key=f"delete_{conv['id']}"):
                                conv_manager.delete_conversation(conv["id"])
                                st.rerun()
                        st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p>🧬 Jivanu Research Assistant | Powered by Advanced RAG & GPT-4</p>
        <p style="font-size: 0.8rem;">Specialized for microbe-based therapeutics research</p>
        <p style="font-size: 0.7rem; margin-top: 0.5rem;">💡 Tip: All conversations are automatically saved!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()