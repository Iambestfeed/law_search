from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Optional, Set
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class BaseClassificationEvaluationEngine(ABC):
    """Base class for classification evaluation engines.
    
    This class defines the core evaluation metrics and interfaces for evaluating
    classification systems. It supports standard metrics like accuracy, precision,
    recall, and F1-score.
    """
    
    @staticmethod
    def accuracy(y_true: List[Any], y_pred: List[Any]) -> float:
        """Calculate accuracy score.
        
        Args:
            y_true: List of true labels.
            y_pred: List of predicted labels.
            
        Returns:
            float: Accuracy score.
        """
        return float(accuracy_score(y_true, y_pred))
    
    @staticmethod
    def precision(y_true: List[Any], y_pred: List[Any], average: str = 'weighted') -> float:
        """Calculate precision score.
        
        Args:
            y_true: List of true labels.
            y_pred: List of predicted labels.
            average: Averaging strategy for multi-class classification.
            
        Returns:
            float: Precision score.
        """
        return float(precision_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def recall(y_true: List[Any], y_pred: List[Any], average: str = 'weighted') -> float:
        """Calculate recall score.
        
        Args:
            y_true: List of true labels.
            y_pred: List of predicted labels.
            average: Averaging strategy for multi-class classification.
            
        Returns:
            float: Recall score.
        """
        return float(recall_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def f1(y_true: List[Any], y_pred: List[Any], average: str = 'weighted') -> float:
        """Calculate F1 score.
        
        Args:
            y_true: List of true labels.
            y_pred: List of predicted labels.
            average: Averaging strategy for multi-class classification.
            
        Returns:
            float: F1 score.
        """
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def confusion_matrix(y_true: List[Any], y_pred: List[Any]) -> np.ndarray:
        """Calculate confusion matrix.
        
        Args:
            y_true: List of true labels.
            y_pred: List of predicted labels.
            
        Returns:
            np.ndarray: Confusion matrix.
        """
        return confusion_matrix(y_true, y_pred)
    
    @abstractmethod
    def evaluate(self,
                 texts: List[str],
                 true_labels: List[Any],
                 **kwargs) -> Dict[str, float]:
        """Evaluate classification performance.
        
        Args:
            texts: List of text samples to classify.
            true_labels: List of true labels for the text samples.
            **kwargs: Additional parameters for specific implementations.
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics and their values.
        """
        pass