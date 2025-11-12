"""
Session memory and context management for the agent.
Maintains conversation history and context across interactions.
"""

from typing import List, Dict, Optional
from datetime import datetime
import json


class SessionMemory:
    """
    Manages session memory and context for the agent.
    Stores conversation history, tool calls, and analysis results.
    """
    
    def __init__(self, max_history: int = 50):
        """
        Initialize session memory.
        
        Args:
            max_history: Maximum number of conversation turns to keep
        """
        self.max_history = max_history
        self.conversation_history: List[Dict] = []
        self.tool_calls: List[Dict] = []
        self.analysis_cache: Dict[str, Dict] = {}
        self.context: Dict = {}
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add a message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.conversation_history.append(message)
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def add_tool_call(self, tool_name: str, parameters: Dict, result: Dict):
        """
        Record a tool call.
        
        Args:
            tool_name: Name of the tool called
            parameters: Parameters passed to the tool
            result: Result returned by the tool
        """
        tool_call = {
            'tool_name': tool_name,
            'parameters': parameters,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        self.tool_calls.append(tool_call)
    
    def cache_analysis(self, key: str, analysis: Dict):
        """
        Cache analysis results for reuse.
        
        Args:
            key: Cache key
            analysis: Analysis results
        """
        self.analysis_cache[key] = {
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_cached_analysis(self, key: str) -> Optional[Dict]:
        """
        Retrieve cached analysis.
        
        Args:
            key: Cache key
        
        Returns:
            Cached analysis or None
        """
        return self.analysis_cache.get(key)
    
    def update_context(self, key: str, value: any):
        """
        Update context information.
        
        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value
    
    def get_context(self, key: str, default: any = None) -> any:
        """
        Get context information.
        
        Args:
            key: Context key
            default: Default value if key not found
        
        Returns:
            Context value or default
        """
        return self.context.get(key, default)
    
    def get_recent_history(self, n: int = 5) -> List[Dict]:
        """
        Get recent conversation history.
        
        Args:
            n: Number of recent messages to return
        
        Returns:
            List of recent messages
        """
        return self.conversation_history[-n:]
    
    def get_conversation_summary(self) -> str:
        """
        Generate a summary of the conversation.
        
        Returns:
            Conversation summary
        """
        if not self.conversation_history:
            return "No conversation history."
        
        summary_parts = []
        summary_parts.append(f"Conversation started: {self.conversation_history[0]['timestamp']}")
        summary_parts.append(f"Total messages: {len(self.conversation_history)}")
        summary_parts.append(f"Tool calls made: {len(self.tool_calls)}")
        
        if self.tool_calls:
            tool_names = [tc['tool_name'] for tc in self.tool_calls]
            summary_parts.append(f"Tools used: {', '.join(set(tool_names))}")
        
        return "\n".join(summary_parts)
    
    def clear(self):
        """Clear all memory."""
        self.conversation_history = []
        self.tool_calls = []
        self.analysis_cache = {}
        self.context = {}
    
    def to_dict(self) -> Dict:
        """Convert memory to dictionary for serialization."""
        return {
            'conversation_history': self.conversation_history,
            'tool_calls': self.tool_calls,
            'analysis_cache': self.analysis_cache,
            'context': self.context
        }
    
    def from_dict(self, data: Dict):
        """Load memory from dictionary."""
        self.conversation_history = data.get('conversation_history', [])
        self.tool_calls = data.get('tool_calls', [])
        self.analysis_cache = data.get('analysis_cache', {})
        self.context = data.get('context', {})
