# src/conversation/conversation_manager.py
"""
Conversation history management with persistent storage
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib

class ConversationManager:
    """Manages conversation history with persistent storage"""
    
    def __init__(self, storage_dir: str = "conversations"):
        """
        Initialize conversation manager
        
        Args:
            storage_dir: Directory to store conversation files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.index_file = self.storage_dir / "index.json"
        self.conversations = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load conversation index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading conversation index: {e}")
        return {}
    
    def _save_index(self):
        """Save conversation index to disk"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversations, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversation index: {e}")
    
    def _generate_conversation_id(self, title: str) -> str:
        """Generate unique conversation ID"""
        timestamp = datetime.now().isoformat()
        unique_string = f"{title}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def _get_conversation_file(self, conversation_id: str) -> Path:
        """Get file path for a conversation"""
        return self.storage_dir / f"{conversation_id}.json"
    
    def create_conversation(self, title: str = None) -> str:
        """
        Create a new conversation
        
        Args:
            title: Optional title for the conversation
            
        Returns:
            conversation_id
        """
        if not title:
            title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        conversation_id = self._generate_conversation_id(title)
        
        conversation_data = {
            "id": conversation_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
            "metadata": {
                "query_count": 0,
                "total_tokens": 0
            }
        }
        
        # Save to index
        self.conversations[conversation_id] = {
            "title": title,
            "created_at": conversation_data["created_at"],
            "updated_at": conversation_data["updated_at"],
            "query_count": 0
        }
        self._save_index()
        
        # Save conversation file
        self._save_conversation(conversation_id, conversation_data)
        
        return conversation_id
    
    def _save_conversation(self, conversation_id: str, data: Dict):
        """Save conversation data to file"""
        file_path = self._get_conversation_file(conversation_id)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversation {conversation_id}: {e}")
    
    def load_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Load conversation data from file"""
        file_path = self._get_conversation_file(conversation_id)
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading conversation {conversation_id}: {e}")
            return None
    
    def add_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str,
        metadata: Dict = None
    ):
        """
        Add a message to conversation
        
        Args:
            conversation_id: ID of the conversation
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (sources, confidence, etc.)
        """
        conversation = self.load_conversation(conversation_id)
        if not conversation:
            return
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        conversation["messages"].append(message)
        conversation["updated_at"] = datetime.now().isoformat()
        
        if role == "user":
            conversation["metadata"]["query_count"] += 1
        
        # Update index
        self.conversations[conversation_id]["updated_at"] = conversation["updated_at"]
        self.conversations[conversation_id]["query_count"] = conversation["metadata"]["query_count"]
        self._save_index()
        
        # Save conversation
        self._save_conversation(conversation_id, conversation)
    
    def get_conversation_history(self, conversation_id: str) -> List[Tuple[str, str]]:
        """
        Get conversation history as list of (role, content) tuples
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List of (role, content) tuples
        """
        conversation = self.load_conversation(conversation_id)
        if not conversation:
            return []
        
        return [(msg["role"], msg["content"]) for msg in conversation["messages"]]
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict]:
        """Get full conversation messages with metadata"""
        conversation = self.load_conversation(conversation_id)
        if not conversation:
            return []
        return conversation["messages"]
    
    def list_conversations(self, limit: int = 100) -> List[Dict]:
        """
        List all conversations sorted by updated time
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation summaries
        """
        conversations_list = []
        
        for conv_id, conv_data in self.conversations.items():
            conversations_list.append({
                "id": conv_id,
                "title": conv_data["title"],
                "created_at": conv_data["created_at"],
                "updated_at": conv_data["updated_at"],
                "query_count": conv_data.get("query_count", 0)
            })
        
        # Sort by updated_at (most recent first)
        conversations_list.sort(key=lambda x: x["updated_at"], reverse=True)
        
        return conversations_list[:limit]
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation
        
        Args:
            conversation_id: ID of conversation to delete
            
        Returns:
            True if deleted, False if not found
        """
        if conversation_id not in self.conversations:
            return False
        
        # Delete file
        file_path = self._get_conversation_file(conversation_id)
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Error deleting conversation file: {e}")
        
        # Remove from index
        del self.conversations[conversation_id]
        self._save_index()
        
        return True
    
    def update_conversation_title(self, conversation_id: str, new_title: str):
        """Update conversation title"""
        if conversation_id not in self.conversations:
            return
        
        conversation = self.load_conversation(conversation_id)
        if conversation:
            conversation["title"] = new_title
            self.conversations[conversation_id]["title"] = new_title
            self._save_conversation(conversation_id, conversation)
            self._save_index()
    
    def search_conversations(self, query: str) -> List[Dict]:
        """Search conversations by title or content"""
        query_lower = query.lower()
        results = []
        
        for conv_id, conv_data in self.conversations.items():
            # Search in title
            if query_lower in conv_data["title"].lower():
                results.append({
                    "id": conv_id,
                    "title": conv_data["title"],
                    "updated_at": conv_data["updated_at"],
                    "match_type": "title"
                })
                continue
            
            # Search in messages
            conversation = self.load_conversation(conv_id)
            if conversation:
                for msg in conversation["messages"]:
                    if query_lower in msg["content"].lower():
                        results.append({
                            "id": conv_id,
                            "title": conv_data["title"],
                            "updated_at": conv_data["updated_at"],
                            "match_type": "content"
                        })
                        break
        
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about conversations"""
        total_conversations = len(self.conversations)
        total_queries = sum(conv.get("query_count", 0) for conv in self.conversations.values())
        
        return {
            "total_conversations": total_conversations,
            "total_queries": total_queries
        }