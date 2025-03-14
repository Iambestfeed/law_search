import unittest
from typing import List
from modules.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test case."""
        self.processor = DocumentProcessor()
        self.sample_docs = [
            "Article 1. This is a sample legal document that outlines basic rights.",
            "Section 2.1 The party of the first part hereby agrees to the terms.",
            "Case Law: Smith v. Jones (2023) established precedent for..."
        ]

    def test_process_documents(self):
        """Test basic document processing functionality."""
        result = self.processor.process_documents(self.sample_docs)
        self.assertTrue(result)

    def test_empty_document(self):
        """Test processing of empty document."""
        result = self.processor.process_documents([""])
        self.assertTrue(result)

    def test_invalid_input(self):
        """Test processing with invalid input."""
        with self.assertRaises(ValueError):
            self.processor.process_documents(None)

    def test_multiple_documents(self):
        """Test processing of multiple documents at once."""
        result = self.processor.process_documents(self.sample_docs * 2)
        self.assertTrue(result)

    def test_special_characters(self):
        """Test processing of documents with special characters."""
        special_doc = ["§1.2.3 Legal clause with special characters: ©®™"]
        result = self.processor.process_documents(special_doc)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()