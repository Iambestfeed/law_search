from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseRetrievalEngine(ABC):
    """Abstract base class for retrieval engines.
    
    This class defines the interface that all retrieval engine implementations must follow.
    It provides methods for indexing documents, searching, and managing the index.
    """
    
    @abstractmethod
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Index a list of documents.
        
        Args:
            documents: List of document dictionaries, each containing 'content' and 'metadata'.
            
        Returns:
            bool: True if indexing was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Search the indexed documents.
        
        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance score threshold.
            
        Returns:
            List of document dictionaries with relevance scores.
        """
        pass
    
    @abstractmethod
    def save(self, directory: str) -> bool:
        """Save the index to disk.
        
        Args:
            directory: Directory path to save the index.
            
        Returns:
            bool: True if save was successful, False otherwise.
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, directory: str) -> 'BaseRetrievalEngine':
        """Load an index from disk.
        
        Args:
            directory: Directory path containing the saved index.
            
        Returns:
            BaseRetrievalEngine: Loaded retrieval engine instance.
        """
        pass