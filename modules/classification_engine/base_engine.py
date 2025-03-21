from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

class BaseClassificationEngine(ABC):
    """Abstract base class for classification engines.
    
    This class defines the interface that all classification engine implementations must follow.
    It provides methods for training models, predicting labels, and evaluating performance.
    """
    
    @abstractmethod
    def train(self, texts: List[str], labels: List[Any], **kwargs) -> bool:
        """Train the classification model.
        
        Args:
            texts: List of text samples for training.
            labels: List of labels corresponding to the text samples.
            **kwargs: Additional parameters for specific implementations.
            
        Returns:
            bool: True if training was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def predict(self, texts: List[str], **kwargs) -> List[Any]:
        """Predict labels for the given texts.
        
        Args:
            texts: List of text samples to classify.
            **kwargs: Additional parameters for specific implementations.
            
        Returns:
            List[Any]: Predicted labels for the input texts.
        """
        pass
    
    @abstractmethod
    def evaluate(self, texts: List[str], true_labels: List[Any], **kwargs) -> Dict[str, float]:
        """Evaluate classification performance.
        
        Args:
            texts: List of text samples to classify.
            true_labels: List of true labels for the text samples.
            **kwargs: Additional parameters for specific implementations.
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics and their values.
        """
        pass
    
    @abstractmethod
    def save(self, directory: str) -> bool:
        """Save the model to disk.
        
        Args:
            directory: Directory path to save the model.
            
        Returns:
            bool: True if save was successful, False otherwise.
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, directory: str) -> 'BaseClassificationEngine':
        """Load a model from disk.
        
        Args:
            directory: Directory path containing the saved model.
            
        Returns:
            BaseClassificationEngine: Loaded classification engine instance.
        """
        pass