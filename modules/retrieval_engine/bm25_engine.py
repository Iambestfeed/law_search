import os
import json
from typing import List, Dict, Any, Optional
import bm25s
import Stemmer
from .base_engine import BaseRetrievalEngine

class BM25RetrievalEngine(BaseRetrievalEngine):
    """BM25 implementation of the retrieval engine using bm25s library.
    
    This implementation provides document indexing and search functionality using
    the BM25 ranking algorithm. It supports multiple languages and includes features
    like document metadata management and index persistence.
    """
    
    def __init__(self, 
                        k1: float = 1.5, 
                        b: float = 0.75):
        """Initialize the BM25 retrieval engine.
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5).
            b: Length normalization parameter (default: 0.75).
        """
        self.k1 = k1
        self.b = b
        self.retriever = bm25s.BM25(k1=k1, b=b)
        self.document_store = []
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Index a list of documents.
        
        Args:
            documents: List of document dictionaries, each containing 'content' and 'metadata'.
            
        Returns:
            bool: True if indexing was successful.
        """
        try:          
            self.document_store = documents
            # Tokenize corpus
            corpus_tokens = bm25s.tokenize(documents)
            
            # Index the corpus
            self.retriever.index(corpus_tokens)
            
            return True
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")
            return False
    
    def search(self, query: str, 
                    top_k: int = 5, 
                    min_score: float = 0.0) -> tuple:
        """Search the indexed documents.
        
        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance score threshold.
            
        Returns:
            tuple: (doc_ids, scores) as numpy arrays after filtering by min_score.
        """
        if not query.strip():
            return [], []
        
        try:                
            # Tokenize query
            query_tokens = bm25s.tokenize(query)
            
            # Retrieve results
            doc_ids, scores = self.retriever.retrieve(query_tokens, k=top_k)
            
            # Filter by min_score using boolean indexing
            mask = scores >= min_score
            return doc_ids[mask], scores[mask]
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return [], []
    
    def save(self, directory: str) -> bool:
        """Save the index to disk.
        
        Args:
            directory: Directory path to save the index.
            
        Returns:
            bool: True if save was successful.
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save BM25 index
            self.retriever.save(directory)
            
            # Save additional engine state
            state = {
                'k1': self.k1,
                'b': self.b,
                'corpus' : self.document_store
            }
            
            with open(os.path.join(directory, 'engine_state.json'), 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            return False
    
    @classmethod
    def load(cls, directory: str) -> 'BM25RetrievalEngine':
        """Load an index from disk.
        
        Args:
            directory: Directory path containing the saved index.
            
        Returns:
            BM25RetrievalEngine: Loaded retrieval engine instance.
        """
        try:
            # Load engine state
            with open(os.path.join(directory, 'engine_state.json'), 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Create new engine instance with saved parameters
            engine = cls(
                k1=state['k1'],
                b=state['b'],
            )
            
            # Load BM25 index
            engine.retriever = bm25s.BM25.load(directory, load_corpus = True)
            
            # Restore document store and count
            engine.document_store = state['corpus']
            

            
            return engine
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return None
