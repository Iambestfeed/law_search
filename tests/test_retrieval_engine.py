import unittest
import os
from modules.retrieval_engine.bm25_engine import BM25RetrievalEngine

class TestRetrievalEngine(unittest.TestCase):
    def setUp(self):
        # Initialize the retrieval engine
        self.engine = BM25RetrievalEngine(language="vietnamese")
        
        # Sample documents
        self.documents = [
            {
                'content': 'Điều 1. Vị trí, chức năng của Chính phủ.',
                'metadata': {'article': 1}
            },
            {
                'content': 'Điều 2. Cơ cấu tổ chức và thành viên của Chính phủ.',
                'metadata': {'article': 2}
            }
        ]
        
        # Index the documents
        self.engine.index_documents(self.documents)
    
    def test_search(self):
        """Test exact phrase matching"""
        query = "Vị trí, chức năng của Chính phủ."
        results = self.engine.search(query, top_k=1)
        
        self.assertEqual(results[0][0], 0)  # Should match first document
        
    def test_minimum_score_threshold(self):
        """Test minimum score threshold filtering"""
        query = "Quốc hội"
        results = self.engine.search(query, top_k=2, min_score=0.5)
        
        self.assertTrue(all(score >= 0.5 for score in results[1]))
        
    def test_no_match_search(self):
        """Test search with no matching results"""
        query = "nhiem"
        results = self.engine.search(query, min_score=0.1)
        self.assertEqual(len(results[0]), 0)
        
    def test_save_and_load(self):
        """Test saving and loading the index"""
        save_dir = "test_index"
        
        # Save the index
        success = self.engine.save(save_dir)
        self.assertTrue(success)
        
        # Load the index
        loaded_engine = BM25RetrievalEngine.load(save_dir)
        self.assertIsNotNone(loaded_engine)
        
        # Compare search results
        query = "Chính phủ"
        original_results = self.engine.search(query)
        loaded_results = loaded_engine.search(query)
        
        self.assertEqual(len(original_results), len(loaded_results))
        
        # Cleanup
        if os.path.exists(save_dir):
            for file in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, file))
            os.rmdir(save_dir)

if __name__ == '__main__':
    unittest.main()