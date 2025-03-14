import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
from .base_engine import BaseRetrievalEngine

class DenseRetrievalEngine(BaseRetrievalEngine):
    """Dense retrieval engine implementation using sentence transformers and FAISS.
    
    This implementation provides document indexing and search functionality using
    dense embeddings generated by sentence transformers and FAISS for efficient
    similarity search.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 dimension: int = 384,
                 metric: str = "cosine"):
        """Initialize the dense retrieval engine.
        
        Args:
            model_name: Name of the sentence transformer model (default: all-MiniLM-L6-v2).
            dimension: Dimension of embeddings (default: 384).
            metric: Distance metric for FAISS index (default: cosine).
        """
        self.model_name = model_name
        self.dimension = dimension
        self.metric = metric
        
        # Initialize sentence transformer model
        self.model = SentenceTransformer(model_name, token=False)
        
        # Initialize FAISS index
        if metric == "cosine":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
        else:
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance
            
        self.document_store = []
        self.doc_count = 0
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Index a list of documents.
        
        Args:
            documents: List of document dictionaries, each containing 'content' and 'metadata'.
            
        Returns:
            bool: True if indexing was successful.
        """
        try:
            # Reset document store and index
            self.document_store = documents
            self.index = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" else faiss.IndexFlatL2(self.dimension)
            corpus = documents
            
            # Generate embeddings
            embeddings = self.model.encode(corpus, convert_to_numpy=True)
            
            # Normalize embeddings if using cosine similarity
            if self.metric == "cosine":
                faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Update document count
            self.doc_count = len(self.document_store)
            
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
            min_score: Minimum similarity score threshold.
            
        Returns:
            tuple: (doc_ids, scores) as numpy arrays after filtering by min_score.
        """
        if not query.strip():
            return [], []
        
        try:
            # Adjust top_k if it's larger than corpus size
            adjusted_top_k = min(top_k, self.doc_count)
            if adjusted_top_k == 0:
                return [], []
            
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Normalize query embedding if using cosine similarity
            if self.metric == "cosine":
                faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            scores, doc_ids = self.index.search(query_embedding, adjusted_top_k)
            
            # Convert to 1D arrays
            scores = scores[0]
            doc_ids = doc_ids[0]
            
            # Filter by min_score
            if self.metric == "cosine":
                mask = scores >= min_score
            else:
                # Convert L2 distance to similarity score (1 / (1 + distance))
                scores = 1 / (1 + scores)
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
            
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(directory, 'faiss.index'))
            
            # Save additional engine state
            state = {
                'model_name': self.model_name,
                'dimension': self.dimension,
                'metric': self.metric,
                'doc_count': self.doc_count,
                'document_store': self.document_store
            }
            
            with open(os.path.join(directory, 'engine_state.json'), 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            return False
    
    @classmethod
    def load(cls, directory: str) -> 'DenseRetrievalEngine':
        """Load an index from disk.
        
        Args:
            directory: Directory path containing the saved index.
            
        Returns:
            DenseRetrievalEngine: Loaded retrieval engine instance.
        """
        try:
            # Load engine state
            with open(os.path.join(directory, 'engine_state.json'), 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Create new engine instance with saved parameters
            engine = cls(
                model_name=state['model_name'],
                dimension=state['dimension'],
                metric=state['metric']
            )
            
            # Restore document store and count
            engine.document_store = state['document_store']
            engine.doc_count = state['doc_count']
            
            # Load FAISS index
            engine.index = faiss.read_index(os.path.join(directory, 'faiss.index'))
            
            return engine
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return None