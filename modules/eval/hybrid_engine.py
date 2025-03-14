from typing import List, Dict, Any, Optional, Set
import numpy as np
from tqdm import tqdm
from .base_engine import BaseEvaluationEngine
from modules.retrieval_engine.hybrid_engine import HybridRetrievalEngine

class HybridEvaluationEngine(BaseEvaluationEngine):
    """Hybrid evaluation engine combining BM25 and dense retrieval using RRF.
    
    This implementation evaluates hybrid retrieval performance by combining
    BM25 and dense retrieval results using reciprocal rank fusion (RRF).
    """
    
    def __init__(self,
                 model_name: str,
                 bm25_weight: float = 0.5,
                 dense_weight: float = 0.5,
                 rrf_k: float = 60.0,
                 dimension: int = 768,
                 trust_remote_code: bool = True):
        """Initialize the hybrid evaluation engine.
        
        Args:
            model_name: Name/path of the sentence transformer model.
            bm25_weight: Weight for BM25 scores in fusion (default: 0.5).
            dense_weight: Weight for dense scores in fusion (default: 0.5).
            rrf_k: Constant for RRF score computation (default: 60.0).
            batch_size: Batch size for encoding (default: 32).
            trust_remote_code: Whether to trust remote code (default: True).
        """
        # Initialize the HybridRetrievalEngine with dense parameters
        dense_params = {
            'model_name': model_name,
            'dimension' : dimension
        }
        
        self.retrieval_engine = HybridRetrievalEngine(
            dense_params=dense_params,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            rrf_k=rrf_k
        )
        self.batch_size = 32
    
    # Using HybridRetrievalEngine's methods instead of reimplementing them
    
    def evaluate(self,
                queries: Dict[str, str],
                corpus: Dict[str, str],
                relevant_docs: Dict[str, Set[str]],
                top_k_values: List[int] = [3, 5, 10],
                bm25_strategy: str = 'bm25',
                show_progress: bool = True) -> Dict[str, float]:
        """Evaluate hybrid retrieval performance.
        
        Args:
            queries: Dictionary mapping query IDs to query texts.
            corpus: Dictionary mapping document IDs to document texts.
            relevant_docs: Dictionary mapping query IDs to sets of relevant document IDs.
            top_k_values: List of k values for evaluation metrics.
            bm25_strategy: Strategy for BM25 score transformation.
            show_progress: Whether to show progress bar.
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics and their values.
        """
        # Prepare documents for indexing
        documents = corpus['text']
        
        # Index documents using the retrieval engine
        print("Indexing documents...")
        self.retrieval_engine.index_documents(documents)
        
        # Initialize results dictionary
        metrics = ['ndcg', 'mrr', 'recall', 'precision', 'f1']
        results = {f"{m}@{k}": [] for m in metrics for k in top_k_values}
        
        # Evaluate each query
        query_iter = tqdm(range(len(queries))) if show_progress else range(len(queries))
        print("Evaluating queries...")
        for qid in query_iter:
            query = queries[qid]['question']
            query_id = queries[qid]['query_id']
            # Evaluate at different k values
            for k in top_k_values:
                # Use the retrieval engine to get top results
                doc_ids, scores = self.retrieval_engine.search(
                    query=query,
                    top_k=k,
                    bm25_strategy=bm25_strategy
                )
                
                # Convert to list if it's a numpy array
                if isinstance(doc_ids, np.ndarray):
                    doc_ids = doc_ids.tolist()
                
                # Calculate relevance scores
                # Convert numeric indices to actual document IDs from corpus
                relevance = [1.0 if corpus[int(doc_id)]['id'] in relevant_docs[query_id] else 0.0
                            for doc_id in doc_ids]
                
                # Calculate metrics
                results[f'ndcg@{k}'].append(self.ndcg_at_k(relevance, k))
                results[f'mrr@{k}'].append(self.mrr_at_k(relevance, k))
                
                # Convert numeric indices to actual document IDs for set operations
                retrieved_doc_ids = [corpus[int(doc_id)]['id'] for doc_id in doc_ids]
                retrieved_relevant = set(retrieved_doc_ids) & relevant_docs[query_id]
                num_relevant = len(relevant_docs[query_id])
                num_retrieved = len(doc_ids)
                num_retrieved_relevant = len(retrieved_relevant)
                
                # Calculate recall
                recall = num_retrieved_relevant / num_relevant if num_relevant > 0 else 0.0
                results[f'recall@{k}'].append(recall)
                
                # Calculate precision
                precision = num_retrieved_relevant / num_retrieved if num_retrieved > 0 else 0.0
                results[f'precision@{k}'].append(precision)
                
                # Calculate F1
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                results[f'f1@{k}'].append(f1)
        
        # Average results
        return {metric: float(np.mean(values))
                for metric, values in results.items()}