import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from modules.eval.hybrid_engine import HybridEvaluationEngine

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Hybrid Retrieval Evaluation")
parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
args = parser.parse_args()

# Load configuration from JSON file
with open(args.config, "r") as f:
    config = json.load(f)

bm25_weight = config.get("bm25_weight", 0.5)
dense_weight = config.get("dense_weight", 0.5)
rrf_k = config.get("rrf_k", 60.0)
batch_size = config.get("batch_size", 32)
dimension = config.get("dimension", 768)
models = config.get("models", {})
dataset_name_base = config.get("dataset_name_base", "another-symato/VMTEB-Zalo-legel-retrieval")
num_corpus_addition = config.get("num_corpus_addition", 10000)
output_dir = config.get("output_dir", "output/scores.json")

eval_results = {}

for model_key, model_config in models.items():
    dataset_name = dataset_name_base + "-wseg" if model_config["word_segment"] else dataset_name_base
    
    # Load datasets
    corpus = load_dataset(dataset_name, "corpus", split='train')
    queries = load_dataset(dataset_name, "queries", split='train')
    relevant_docs_data = load_dataset(dataset_name, "data_ir", split="test")
    
    # Filter corpus and queries based on relevant docs
    required_corpus_ids = list(map(str, relevant_docs_data["corpus_id"]))
    if num_corpus_addition:
        required_corpus_ids += random.sample(corpus["id"], k=num_corpus_addition)
    corpus = corpus.filter(lambda x: x["id"] in required_corpus_ids)
    
    required_queries_ids = list(map(str, relevant_docs_data["query_id"]))
    queries = queries.filter(lambda x: x["query_id"] in required_queries_ids)
    
    # Convert to dictionary format
    corpus_dict = dict(zip(corpus["id"], corpus["text"]))
    queries_dict = dict(zip(queries["query_id"], queries["question"]))
    relevant_docs = {}
    for qid, corpus_id in zip(relevant_docs_data["query_id"], relevant_docs_data["corpus_id"]):
        qid = str(qid)
        corpus_id = str(corpus_id)
        if qid not in relevant_docs:
            relevant_docs[qid] = set()
        relevant_docs[qid].add(corpus_id)
    
    # Initialize evaluation engine
    engine = HybridEvaluationEngine(
        model_name=model_config["model_name"],
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        rrf_k=rrf_k,
        dimension=dimension, 
        batch_size = batch_size
    )
    
    # Run evaluation
    eval_results[model_key] = engine.evaluate(queries, corpus, relevant_docs)

# Create score board
df_score_board = pd.DataFrame(eval_results)
df_score_board.to_json(output_dir)