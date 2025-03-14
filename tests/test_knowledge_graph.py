import unittest
from typing import List, Dict, Any
from modules.knowledge_graph import KnowledgeGraph

class TestKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test case."""
        self.graph = KnowledgeGraph()
        self.sample_nodes = [
            {
                'id': 'case001',
                'type': 'case',
                'properties': {
                    'title': 'Smith v. Jones',
                    'year': 2023
                }
            },
            {
                'id': 'statute001',
                'type': 'statute',
                'properties': {
                    'name': 'Criminal Code',
                    'section': '123'
                }
            }
        ]
        self.sample_relations = [
            {
                'source': 'case001',
                'target': 'statute001',
                'type': 'cites'
            }
        ]

    def test_add_nodes(self):
        """Test adding nodes to the graph."""
        result = self.graph.add_nodes(self.sample_nodes)
        self.assertTrue(result)
        self.assertEqual(self.graph.node_count(), 2)

    def test_add_relations(self):
        """Test adding relations between nodes."""
        self.graph.add_nodes(self.sample_nodes)
        result = self.graph.add_relations(self.sample_relations)
        self.assertTrue(result)
        self.assertEqual(self.graph.relation_count(), 1)

    def test_get_node(self):
        """Test retrieving a specific node."""
        self.graph.add_nodes(self.sample_nodes)
        node = self.graph.get_node('case001')
        self.assertIsNotNone(node)
        self.assertEqual(node['properties']['title'], 'Smith v. Jones')

    def test_get_relations(self):
        """Test retrieving relations for a node."""
        self.graph.add_nodes(self.sample_nodes)
        self.graph.add_relations(self.sample_relations)
        relations = self.graph.get_relations('case001')
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]['type'], 'cites')

    def test_remove_node(self):
        """Test removing a node and its relations."""
        self.graph.add_nodes(self.sample_nodes)
        self.graph.add_relations(self.sample_relations)
        result = self.graph.remove_node('case001')
        self.assertTrue(result)
        self.assertEqual(self.graph.node_count(), 1)
        self.assertEqual(self.graph.relation_count(), 0)

    def test_invalid_relation(self):
        """Test adding relation with non-existent nodes."""
        self.graph.add_nodes([self.sample_nodes[0]])
        invalid_relation = {
            'source': 'case001',
            'target': 'nonexistent',
            'type': 'cites'
        }
        with self.assertRaises(ValueError):
            self.graph.add_relations([invalid_relation])

if __name__ == '__main__':
    unittest.main()