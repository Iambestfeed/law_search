# Legal Document Retrieval System

A Python-based system for efficient legal document processing, retrieval, and knowledge graph construction. The system implements BM25 ranking algorithm for accurate document search and maintains a knowledge graph for semantic relationships.

## Features

- BM25-based document retrieval engine
- Document processing with stemming and stopword removal
- Knowledge graph construction from legal documents
- Comprehensive test suite

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd law_graph_agent
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Unix/MacOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from modules.retrieval_engine.bm25_engine import BM25RetrievalEngine

# Initialize the engine
engine = BM25RetrievalEngine()

# Index documents
documents = [
    {
        'content': 'Legal document content...',
        'metadata': {'case_id': '001', 'date': '2023-01-01'}
    }
]
engine.index_documents(documents)

# Search documents
results = engine.search('search query', top_k=5)
```

### Configuration

The BM25 retrieval engine can be configured with the following parameters:

- `k1` (default: 1.5): Term frequency saturation parameter
- `b` (default: 0.75): Length normalization parameter
- `use_stemming` (default: False): Enable word stemming
- `use_stopwords` (default: True): Enable stopword removal
- `language` (default: "english"): Language for stemming

## Development

### Project Structure

```
law_graph_agent/
├── modules/
│   ├── document_processor/
│   ├── knowledge_graph/
│   └── retrieval_engine/
├── tests/
│   ├── test_document_processor.py
│   ├── test_knowledge_graph.py
│   └── test_retrieval_engine.py
└── main.py
```

### Running Tests

```bash
python -m unittest discover tests
```

### Adding New Features

1. Implement new functionality in the appropriate module
2. Add corresponding test cases
3. Update documentation as needed
4. Run the test suite to ensure everything works

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
