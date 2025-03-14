import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from .base_engine import BaseRetrievalEngine
from .bm25_engine import BM25RetrievalEngine
from .dense_engine import DenseRetrievalEngine

class HybridRetrievalEngine(BaseRetrievalEngine):
    """Hybrid retrieval engine combining BM25 and dense retrieval using RRF.
    
    This implementation combines results from BM25 and dense retrievers using
    reciprocal rank fusion (RRF) to leverage both sparse and dense representations.
    """
    
    def __init__(self, 
                 bm25_params: Dict[str, Any] = None,
                 dense_params: Dict[str, Any] = None,
                 bm25_weight: float = 0.5,
                 dense_weight: float = 0.5,
                 rrf_k: float = 60.0):
        """Initialize the hybrid retrieval engine.
        
        Args:
            bm25_params: Parameters for BM25 engine initialization (default: None).
            dense_params: Parameters for dense engine initialization (default: None).
            bm25_weight: Weight for BM25 scores in fusion (default: 0.5).
            dense_weight: Weight for dense scores in fusion (default: 0.5).
            rrf_k: Constant for RRF score computation (default: 60.0).
        """
        # Initialize retrieval engines
        self.bm25_engine = BM25RetrievalEngine(**(bm25_params or {}))
        self.dense_engine = DenseRetrievalEngine(**(dense_params or {}))
        
        # Set fusion parameters
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        
        self.document_store = []
        self.doc_count = 0
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Index documents using both BM25 and dense engines.
        
        Args:
            documents: List of document dictionaries, each containing 'content' and 'metadata'.
            
        Returns:
            bool: True if indexing was successful.
        """
        try:
            # Reset document store
            self.document_store = []
            
            # Store documents
            self.document_store.extend(documents)
            self.doc_count = len(self.document_store)
            
            # Index documents in parallel
            with ThreadPoolExecutor() as executor:
                bm25_future = executor.submit(self.bm25_engine.index_documents, documents)
                dense_future = executor.submit(self.dense_engine.index_documents, documents)
                
                # Wait for both indexing operations to complete
                bm25_success = bm25_future.result()
                dense_success = dense_future.result()
            
            return bm25_success and dense_success
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")
            return False
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0,1] range.
        
        Args:
            scores: Raw scores array.
            
        Returns:
            numpy.ndarray: Normalized scores.
        """
        if len(scores) == 0:
            return scores
        score_min = np.min(scores)
        score_max = np.max(scores)
        if score_max == score_min:
            return np.ones_like(scores)
        return (scores - score_min) / (score_max - score_min)
    
    def _apply_bm25_strategy(self, scores: np.ndarray, strategy: str = 'bm25') -> np.ndarray:
        """Apply different scoring strategies to BM25 scores.
        
        Args:
            scores: Raw BM25 scores.
            strategy: Scoring strategy ('bm25', 'log', or 'sqrt').
            
        Returns:
            numpy.ndarray: Transformed scores.
        """
        if strategy == 'log':
            return np.log1p(scores)
        elif strategy == 'sqrt':
            return np.sqrt(scores)
        return scores  # raw bm25
    
    def _compute_rrf_scores(self, doc_ids_scores: List[tuple], bm25_strategy: str = 'bm25') -> tuple:
        """Compute RRF scores for a list of document IDs and scores.
        
        Args:
            doc_ids_scores: List of (doc_ids, scores) tuples from retrievers.
            bm25_strategy: Strategy for BM25 score transformation.
            
        Returns:
            tuple: (unique_doc_ids, final_scores) as numpy arrays.
        """
        if not doc_ids_scores:
            return np.array([]), np.array([])
            
        # Process BM25 and dense scores separately
        bm25_doc_ids, bm25_scores = doc_ids_scores[0][1]
        dense_doc_ids, dense_scores = doc_ids_scores[1][1]
        
        # Apply BM25 strategy and normalize scores
        bm25_scores = self._normalize_scores(self._apply_bm25_strategy(bm25_scores, bm25_strategy))
        dense_scores = self._normalize_scores(dense_scores)
        
        # Get ranks (1-based)
        bm25_ranks = np.argsort(-bm25_scores).argsort() + 1
        dense_ranks = np.argsort(-dense_scores).argsort() + 1
        
        # Compute RRF scores
        rrf_scores = {}
        for i, doc_id in enumerate(bm25_doc_ids):
            rrf_scores[doc_id] = self.bm25_weight / (self.rrf_k + bm25_ranks[i])
            
        for i, doc_id in enumerate(dense_doc_ids):
            dense_score = self.dense_weight / (self.rrf_k + dense_ranks[i])
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += dense_score
            else:
                rrf_scores[doc_id] = dense_score
        
        # Convert to arrays
        unique_doc_ids = np.array(list(rrf_scores.keys()))
        final_scores = np.array(list(rrf_scores.values()))
        
        return unique_doc_ids, final_scores
    
    def search(self, query: str, 
               top_k: int = 5, 
               min_score: float = 0.0,
               bm25_strategy: str = 'bm25') -> tuple:
        """Search using both engines and combine results using RRF.
        
        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance score threshold.
            bm25_strategy: Strategy for BM25 score transformation ('bm25', 'log', or 'sqrt').
            
        Returns:
            tuple: (doc_ids, scores) as numpy arrays.
        """
        if not query.strip():
            return [], []
        
        try:
            # Adjust top_k for individual retrievers
            adjusted_top_k = min(top_k * 2, self.doc_count)  # Get more results for better fusion
            if adjusted_top_k == 0:
                return [], []
            
            # Search with both engines sequentially
            bm25_results = self.bm25_engine.search(query, adjusted_top_k)
            dense_results = self.dense_engine.search(query, adjusted_top_k)
            
            # Combine results using RRF
            doc_ids_scores = [
                (self.bm25_weight, bm25_results),
                (self.dense_weight, dense_results)
            ]
            
            unique_doc_ids, final_scores = self._compute_rrf_scores(doc_ids_scores, bm25_strategy)
            
            # Sort by score
            rank_order = np.argsort(-final_scores)  # Descending order
            
            # Apply top_k and min_score filters
            mask = final_scores[rank_order] >= min_score
            filtered_indices = rank_order[mask][:top_k]
            
            result_doc_ids = unique_doc_ids[filtered_indices]
            result_scores = final_scores[filtered_indices]
            
            return result_doc_ids, result_scores
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return [], []
    
    def save(self, directory: str) -> bool:
        """Save both engines and hybrid state to disk.
        
        Args:
            directory: Directory path to save the engines.
            
        Returns:
            bool: True if save was successful.
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save individual engines
            bm25_dir = os.path.join(directory, 'bm25')
            dense_dir = os.path.join(directory, 'dense')
            os.makedirs(bm25_dir, exist_ok=True)
            os.makedirs(dense_dir, exist_ok=True)
            
            bm25_success = self.bm25_engine.save(bm25_dir)
            dense_success = self.dense_engine.save(dense_dir)
            
            # Save hybrid engine state
            state = {
                'bm25_weight': self.bm25_weight,
                'dense_weight': self.dense_weight,
                'rrf_k': self.rrf_k,
                'doc_count': self.doc_count,
                'document_store': self.document_store
            }
            
            with open(os.path.join(directory, 'engine_state.json'), 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            return bm25_success and dense_success
        except Exception as e:
            print(f"Error saving engines: {str(e)}")
            return False
    
    @classmethod
    def load(cls, directory: str) -> 'HybridRetrievalEngine':
        """Load both engines and hybrid state from disk.
        
        Args:
            directory: Directory path containing the saved engines.
            
        Returns:
            HybridRetrievalEngine: Loaded hybrid engine instance.
        """
        try:
            # Load engine state
            with open(os.path.join(directory, 'engine_state.json'), 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Create new engine instance
            engine = cls(
                bm25_weight=state['bm25_weight'],
                dense_weight=state['dense_weight'],
                rrf_k=state['rrf_k']
            )
            
            # Restore document store and count
            engine.document_store = state['document_store']
            engine.doc_count = state['doc_count']
            
            # Load individual engines
            bm25_dir = os.path.join(directory, 'bm25')
            dense_dir = os.path.join(directory, 'dense')
            
            engine.bm25_engine = BM25RetrievalEngine.load(bm25_dir)
            engine.dense_engine = DenseRetrievalEngine.load(dense_dir)
            
            if not engine.bm25_engine or not engine.dense_engine:
                return None
            
            return engine
        except Exception as e:
            print(f"Error loading engines: {str(e)}")
            return None