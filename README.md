# Legal Document Retrieval System

A Python-based system for efficient legal document processing, retrieval, and knowledge graph construction. The system implements multiple retrieval strategies, including BM25, dense retrieval, and hybrid retrieval for accurate document search. It also maintains a knowledge graph for semantic relationships.

## Features

- BM25, dense, and hybrid retrieval engines
- Document processing with stemming and stopword removal (under development)
- Knowledge graph construction from legal documents (under development)
- Configurable retrieval models

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd law_graph_agent
```

2. Create and activate the Conda environment:
```bash
conda env create -f environment.yml  # Using Conda
conda activate law_graph_agent
```

## Usage

### Retrieval Engine

#### BM25 Retrieval Engine

```python
from modules.retrieval_engine.bm25_engine import BM25RetrievalEngine

# Initialize the engine
engine = BM25RetrievalEngine(k1=1.5, b=0.75)

# Index documents
documents = ['Xin chào Việt Nam', 'Xin chào']
engine.index_documents(documents)

# Search documents
results = engine.search('Việt Nam', top_k=2)
```

## Configuration Files

The retrieval models can be configured using JSON files in the `config/` directory. These files specify parameters for different retrieval strategies.

## Model Scores

The performance scores for different retrieval models are stored in the `output/` directory. These scores provide insights into model effectiveness across various retrieval strategies. Example files:

```
output/
├── bm25_scores.json
├── simcse_scores.json
├── bge_m3_scores.json
├── bkai_scores.json
├── halong_scores.json
├── vietnam_embedding_scores.json
└── hybrid retrieval scores...
```

### Scores

| model                                        | type   |   ndcg@3 |   ndcg@5 |   ndcg@10 |    mrr@3 |    mrr@5 |   mrr@10 |
|:---------------------------------------------|:-------|---------:|---------:|----------:|---------:|---------:|---------:|
| AITeamVN/Vietnamese_Embedding                | dense  | 0.842687 | 0.854993 |  0.865006 | 0.822135 | 0.82901  | 0.833389 |
| bkai-foundation-models/vietnamese-bi-encoder | hybrid | 0.827247 | 0.844781 |  0.846937 | 0.799219 | 0.809505 | 0.806771 |
| bkai-foundation-models/vietnamese-bi-encoder | dense  | 0.814116 | 0.82965  |  0.839567 | 0.796615 | 0.805286 | 0.809572 |
| AITeamVN/Vietnamese_Embedding                | hybrid | 0.788724 | 0.810062 |  0.820797 | 0.758333 | 0.77224  | 0.776461 |
| BAAI/bge-m3                                  | dense  | 0.784056 | 0.80665  |  0.817016 | 0.763281 | 0.775859 | 0.780293 |
| BAAI/bge-m3                                  | hybrid | 0.775239 | 0.797382 |  0.811962 | 0.747656 | 0.763333 | 0.77128  |
| hiieu/halong_embedding                       | hybrid | 0.73627  | 0.757183 |  0.779169 | 0.710417 | 0.721901 | 0.731976 |
| bm25                                         | bm25   | 0.728122 | 0.74974  |  0.761612 | 0.699479 | 0.711198 | 0.715738 |
| dangvantuan/vietnamese-embedding             | dense  | 0.718971 | 0.746521 |  0.763416 | 0.696354 | 0.711953 | 0.718854 |
| dangvantuan/vietnamese-embedding             | hybrid | 0.71711  | 0.743537 |  0.758315 | 0.690104 | 0.704792 | 0.712261 |
| VoVanPhuc/sup-SimCSE-VietNamese-phobert-base | hybrid | 0.688483 | 0.713829 |  0.733894 | 0.660156 | 0.671198 | 0.676961 |
| hiieu/halong_embedding                       | dense  | 0.656377 | 0.675881 |  0.701368 | 0.630469 | 0.641406 | 0.652057 |
| VoVanPhuc/sup-SimCSE-VietNamese-phobert-base | dense  | 0.558852 | 0.584799 |  0.611329 | 0.536979 | 0.55112  | 0.562218 |

## Project Structure

```
law_graph_agent/
├── config/
│   ├── aiteamvn_vn_embedding_hybrid.json
│   ├── aiteamvn_vn_embedding.json
│   ├── bm25.json
│   ├── simcse.json
│   └── ...
├── environment.yml
├── main.py
├── modules/
│   ├── document_processor/  # Under development
│   ├── eval/
│   ├── knowledge_graph/  # Under development
│   └── retrieval_engine/
├── notebooks/
├── output/
├── scripts/
├── tests/
└── requirements.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

