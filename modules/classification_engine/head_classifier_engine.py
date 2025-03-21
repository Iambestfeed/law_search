import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from .base_engine import BaseClassificationEngine

class HeadClassifierEngine(BaseClassificationEngine):
    """Classification engine with a scikit-learn model on top of frozen embeddings.
    
    This implementation uses a pre-trained embedding model to generate text embeddings,
    which are then used as input features for a scikit-learn classifier. The embedding
    model is kept frozen (not fine-tuned) during training.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 classifier_type: str = "logistic_regression",
                 batch_size: int = 32,
                 random_state: int = 42,
                 **kwargs):
        """Initialize the head classifier engine.
        
        Args:
            model_name: Name of the sentence transformer model for embeddings.
            classifier_type: Type of classifier to use ('logistic_regression', 'svm', 'random_forest').
            batch_size: Batch size for embedding generation.
            random_state: Random seed for reproducibility.
            **kwargs: Additional parameters for the classifier.
        """
        self.model_name = model_name
        self.classifier_type = classifier_type
        self.batch_size = batch_size
        self.random_state = random_state
        self.classifier_params = kwargs
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name, token=False)
        
        # Initialize classifier
        self.classifier = self._create_classifier()
        
        # Cache for embeddings
        self.embedding_cache = {}
        
    def _create_classifier(self):
        """Create a classifier based on the specified type.
        
        Returns:
            A scikit-learn classifier instance.
        """
        if self.classifier_type == "logistic_regression":
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                **{k: v for k, v in self.classifier_params.items() 
                   if k in LogisticRegression().get_params()}
            )
        elif self.classifier_type == "svm":
            return SVC(
                random_state=self.random_state,
                probability=True,
                **{k: v for k, v in self.classifier_params.items() 
                   if k in SVC().get_params()}
            )
        elif self.classifier_type == "random_forest":
            return RandomForestClassifier(
                random_state=self.random_state,
                **{k: v for k, v in self.classifier_params.items() 
                   if k in RandomForestClassifier().get_params()}
            )
        else:
            # Default to logistic regression
            return LogisticRegression(random_state=self.random_state, max_iter=1000)
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for the input texts.
        
        Args:
            texts: List of text samples.
            
        Returns:
            np.ndarray: Array of embeddings.
        """
        # Generate embeddings
        return self.embedding_model.encode(
            texts, 
            convert_to_numpy=True, 
            batch_size=self.batch_size, 
            show_progress_bar=True
        )
    
    def train(self, texts: List[str], labels: List[Any], **kwargs) -> bool:
        """Train the classification model.
        
        Args:
            texts: List of text samples for training.
            labels: List of labels corresponding to the text samples.
            **kwargs: Additional parameters for training.
            
        Returns:
            bool: True if training was successful.
        """
        try:
            # Generate embeddings for training texts
            print("Generating embeddings for training texts...")
            embeddings = self._get_embeddings(texts)
            
            # Train classifier on embeddings
            print(f"Training {self.classifier_type} classifier...")
            self.classifier.fit(embeddings, labels)
            
            return True
        except Exception as e:
            print(f"Error training classifier: {str(e)}")
            return False
    
    def predict(self, texts: List[str], **kwargs) -> List[Any]:
        """Predict labels for the given texts.
        
        Args:
            texts: List of text samples to classify.
            **kwargs: Additional parameters for prediction.
            
        Returns:
            List[Any]: Predicted labels for the input texts.
        """
        try:
            # Generate embeddings for input texts
            embeddings = self._get_embeddings(texts)
            
            # Predict using classifier
            return self.classifier.predict(embeddings).tolist()
        except Exception as e:
            print(f"Error predicting labels: {str(e)}")
            return []
    
    def predict_proba(self, texts: List[str], **kwargs) -> np.ndarray:
        """Predict class probabilities for the given texts.
        
        Args:
            texts: List of text samples to classify.
            **kwargs: Additional parameters for prediction.
            
        Returns:
            np.ndarray: Predicted class probabilities for the input texts.
        """
        try:
            # Generate embeddings for input texts
            embeddings = self._get_embeddings(texts)
            
            # Predict probabilities using classifier
            return self.classifier.predict_proba(embeddings)
        except Exception as e:
            print(f"Error predicting probabilities: {str(e)}")
            return np.array([])
    
    def evaluate(self, texts: List[str], true_labels: List[Any], average: str = 'weighted', **kwargs) -> Dict[str, float]:
        """Evaluate classification performance.
        
        Args:
            texts: List of text samples to classify.
            true_labels: List of true labels for the text samples.
            average: Averaging strategy for multi-class metrics (default: 'weighted').
            **kwargs: Additional parameters for evaluation.
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics and their values.
        """
        try:
            # Predict labels
            pred_labels = self.predict(texts)
            
            # Calculate metrics
            results = {
                'accuracy': float(accuracy_score(true_labels, pred_labels)),
                'precision': float(precision_score(true_labels, pred_labels, average=average, zero_division=0)),
                'recall': float(recall_score(true_labels, pred_labels, average=average, zero_division=0)),
                'f1': float(f1_score(true_labels, pred_labels, average=average, zero_division=0))
            }
            
            return results
        except Exception as e:
            print(f"Error evaluating classifier: {str(e)}")
            return {}
    
    def save(self, directory: str) -> bool:
        """Save the model to disk.
        
        Args:
            directory: Directory path to save the model.
            
        Returns:
            bool: True if save was successful.
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save classifier using joblib
            from joblib import dump
            dump(self.classifier, os.path.join(directory, 'classifier.joblib'))
            
            # Save model configuration
            config = {
                'model_name': self.model_name,
                'classifier_type': self.classifier_type,
                'batch_size': self.batch_size,
                'random_state': self.random_state,
                'classifier_params': self.classifier_params
            }
            
            with open(os.path.join(directory, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    @classmethod
    def load(cls, directory: str) -> 'HeadClassifierEngine':
        """Load a model from disk.
        
        Args:
            directory: Directory path containing the saved model.
            
        Returns:
            HeadClassifierEngine: Loaded classification engine instance.
        """
        try:
            # Load configuration
            with open(os.path.join(directory, 'config.json'), 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Create engine instance
            engine = cls(
                model_name=config['model_name'],
                classifier_type=config['classifier_type'],
                batch_size=config['batch_size'],
                random_state=config['random_state'],
                **config['classifier_params']
            )
            
            # Load classifier
            from joblib import load
            engine.classifier = load(os.path.join(directory, 'classifier.joblib'))
            
            return engine
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None