from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Optional, Set

class BaseEvaluationEngine(ABC):
    """Base class for evaluation engines.
    
    This class defines the core evaluation metrics and interfaces for evaluating
    retrieval systems. It supports standard metrics like NDCG, MRR, and Recall.
    """
    
    @staticmethod
    def dcg_at_k(relevance: List[float], k: int) -> float:
        """Calculate Discounted Cumulative Gain at k.
        
        Args:
            relevance: List of relevance scores.
            k: Position to calculate DCG at.
            
        Returns:
            float: DCG value at position k.
        """
        relevance = np.asarray(relevance)[:k]
        if relevance.size:
            return float(np.sum(relevance / np.log2(np.arange(2, relevance.size + 2))))
        return 0.0
    
    @staticmethod
    def ndcg_at_k(relevance: List[float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k.
        
        Args:
            relevance: List of relevance scores.
            k: Position to calculate NDCG at.
            
        Returns:
            float: NDCG value at position k.
        """
        dcg_max = BaseEvaluationEngine.dcg_at_k(sorted(relevance, reverse=True), k)
        if not dcg_max:
            return 0.0
        return BaseEvaluationEngine.dcg_at_k(relevance, k) / dcg_max
    
    @staticmethod
    def mrr_at_k(relevance: List[float], k: int) -> float:
        """Calculate Mean Reciprocal Rank at k.
        
        Args:
            relevance: List of relevance scores.
            k: Position to calculate MRR at.
            
        Returns:
            float: MRR value at position k.
        """
        relevance = np.asarray(relevance)[:k]
        if relevance.size and np.any(relevance > 0):
            return 1.0 / (np.argmax(relevance > 0) + 1)
        return 0.0
    
    @staticmethod
    def recall_at_k(retrieved_relevant: Set[str], total_relevant: Set[str], k: int) -> float:
        """Calculate Recall at k.
        
        Args:
            retrieved_relevant: Set of relevant documents retrieved.
            total_relevant: Set of all relevant documents.
            k: Position to calculate Recall at.
            
        Returns:
            float: Recall value at position k.
        """
        if not total_relevant:
            return 0.0
        return len(retrieved_relevant) / len(total_relevant)
    
    @abstractmethod
    def evaluate(self,
                 queries: Dict[str, str],
                 corpus: Dict[str, str],
                 relevant_docs: Dict[str, Set[str]],
                 top_k_values: List[int] = [3, 5, 10],
                 **kwargs) -> Dict[str, float]:
        """Evaluate retrieval performance.
        
        Args:
            queries: Dictionary mapping query IDs to query texts.
            corpus: Dictionary mapping document IDs to document texts.
            relevant_docs: Dictionary mapping query IDs to sets of relevant document IDs.
            top_k_values: List of k values for evaluation metrics.
            **kwargs: Additional parameters for specific implementations.
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics and their values.
        """
        pass