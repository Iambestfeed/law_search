from typing import Dict, List, Set, Any, Optional
import numpy as np
import random
from datasets import load_dataset
from tqdm import tqdm
from .hybrid_engine import HybridEvaluationEngine

def evaluate_hybrid_retrieval(
    model_name: str,
    dataset_name: str,
    word_segment: bool = False,
    num_corpus_addition: int = 0,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
    rrf_k: float = 60.0,
    batch_size: int = 32,
    top_k_values: List[int] = [3, 5, 10],
    bm25_strategy: str = 'bm25',
    show_progress: bool = True,
    trust_remote_code: bool = True
) -> Dict[str, float]:
    """
    Evaluate hybrid retrieval performance using the specified model and dataset.
    
    Args:
        model_name: Name/path of the sentence transformer model.
        dataset_name: Name of the dataset to load.
        word_segment: Whether to use word segmentation version of the dataset.
        num_corpus_addition: Number of additional random corpus documents to include.
        bm25_weight: Weight for BM25 scores in fusion.
        dense_weight: Weight for dense scores in fusion.
        rrf_k: Constant for RRF score computation.
        batch_size: Batch size for encoding.
        top_k_values: List of k values for evaluation metrics.
        bm25_strategy: Strategy for BM25 score transformation.
        show_progress: Whether to show progress bar.
        trust_remote_code: Whether to trust remote code.
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics and their values.
    """
    # Apply word segmentation suffix if needed
    if word_segment:
        dataset_name += '-wseg'
    
    # Load datasets
    print(f"Loading dataset: {dataset_name}")
    corpus = load_dataset(dataset_name, "corpus", split='train')
    queries = load_dataset(dataset_name, "queries", split='train')
    relevant_docs_data = load_dataset(dataset_name, 'data_ir', split="test")
    
    # Filter corpus to include only relevant documents and optional additions
    required_corpus_ids = list(map(str, relevant_docs_data["corpus_id"]))
    if num_corpus_addition:
        required_corpus_ids += random.sample(corpus["id"], k=num_corpus_addition)
    print(f"Total corpus documents: {len(required_corpus_ids)}")
    corpus = corpus.filter(lambda x: x["id"] in required_corpus_ids)
    
    # Filter queries to include only those in the test set
    required_queries_ids = list(map(str, relevant_docs_data["query_id"]))
    queries = queries.filter(lambda x: x["query_id"] in required_queries_ids)
    
    # Create dictionaries for corpus and queries
    corpus_dict = dict(zip(corpus["id"], corpus["text"]))
    queries_dict = dict(zip(queries["query_id"], queries["question"]))
    
    # Build relevance mapping
    relevant_docs = {}
    for qid, corpus_ids in zip(relevant_docs_data["query_id"], relevant_docs_data["corpus_id"]):
        qid = str(qid)
        corpus_ids = str(corpus_ids)
        if qid not in relevant_docs:
            relevant_docs[qid] = set()
        relevant_docs[qid].add(corpus_ids)
    
    # Initialize evaluation engine
    print(f"Initializing HybridEvaluationEngine with model: {model_name}")
    eval_engine = HybridEvaluationEngine(
        model_name=model_name,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        rrf_k=rrf_k,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code
    )
    
    # Format corpus for evaluation
    formatted_corpus = {}
    for doc_id, text in corpus_dict.items():
        formatted_corpus[doc_id] = {
            'id': doc_id,
            'content': text
        }
    
    # Run evaluation
    print("Starting evaluation...")
    results = eval_engine.evaluate(
        queries=queries_dict,
        corpus=formatted_corpus,
        relevant_docs=relevant_docs,
        top_k_values=top_k_values,
        bm25_strategy=bm25_strategy,
        show_progress=show_progress
    )
    
    return results