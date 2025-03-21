import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .base_engine import BaseClassificationEngine

class DenseClassifierEngine(BaseClassificationEngine):
    """Classification engine using a fine-tuned transformer model.
    
    This implementation uses a pre-trained transformer model with a classification head
    that can be fine-tuned on classification data. The entire model (transformer + classification head)
    is fine-tuned during training.
    """
    
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 num_labels: int = 2,
                 batch_size: int = 16,
                 learning_rate: float = 5e-5,
                 epochs: int = 3,
                 max_length: int = 512,
                 device: str = None,
                 **kwargs):
        """Initialize the dense classifier engine.
        
        Args:
            model_name: Name of the pre-trained transformer model.
            num_labels: Number of classification labels.
            batch_size: Batch size for training and inference.
            learning_rate: Learning rate for fine-tuning.
            epochs: Number of training epochs.
            max_length: Maximum sequence length for tokenization.
            device: Device to use for training ('cpu', 'cuda', or None for auto-detection).
            **kwargs: Additional parameters for training.
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_length = max_length
        self.training_args = kwargs
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.label_map = None
        
        # Load tokenizer
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load the tokenizer for the specified model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            print(f"Error loading tokenizer: {str(e)}")
    
    def _load_model(self):
        """Load the model for the specified architecture."""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=self.num_labels
            )
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def _tokenize_texts(self, texts: List[str]):
        """Tokenize a list of texts.
        
        Args:
            texts: List of text samples.
            
        Returns:
            Dict: Tokenized texts with attention masks.
        """
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def _prepare_dataset(self, texts: List[str], labels: Optional[List[Any]] = None):
        """Prepare a dataset for training or inference.
        
        Args:
            texts: List of text samples.
            labels: Optional list of labels.
            
        Returns:
            Dataset: HuggingFace dataset.
        """
        # Create label mapping if not exists and labels are provided
        if self.label_map is None and labels is not None:
            unique_labels = sorted(set(labels))
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
            self.num_labels = len(unique_labels)
            
            # Reload model with correct number of labels
            self._load_model()
        
        # Tokenize texts
        tokenized = self._tokenize_texts(texts)
        
        # Create dataset
        dataset_dict = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        
        # Add labels if provided
        if labels is not None:
            # Convert labels to integers using label map
            dataset_dict["labels"] = torch.tensor([self.label_map[label] for label in labels])
        
        return Dataset.from_dict(dataset_dict)
    
    def train(self, texts: List[str], labels: List[Any], validation_split: float = 0.1, **kwargs) -> bool:
        """Train the classification model.
        
        Args:
            texts: List of text samples for training.
            labels: List of labels corresponding to the text samples.
            validation_split: Proportion of data to use for validation.
            **kwargs: Additional parameters for training.
            
        Returns:
            bool: True if training was successful.
        """
        try:
            # Load tokenizer and model if not already loaded
            if self.tokenizer is None:
                self._load_tokenizer()
                
            # Prepare dataset
            dataset = self._prepare_dataset(texts, labels)
            
            # Split dataset
            if validation_split > 0:
                train_size = int((1 - validation_split) * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            else:
                train_dataset = dataset
                val_dataset = None
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
                evaluation_strategy="epoch" if val_dataset else "no",
                save_strategy="epoch",
                load_best_model_at_end=True if val_dataset else False,
                **{k: v for k, v in self.training_args.items() if k in TrainingArguments.__init__.__code__.co_varnames}
            )
            
            # Set up trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset
            )
            
            # Train model
            print("Training dense classifier model...")
            trainer.train()
            
            # Save best model
            self.model = trainer.model
            
            return True
        except Exception as e:
            print(f"Error training model: {str(e)}")
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
            # Check if model is loaded
            if self.model is None:
                raise ValueError("Model not trained or loaded. Please train or load a model first.")
            
            # Prepare dataset without labels
            dataset = self._prepare_dataset(texts)
            
            # Set up trainer for prediction
            trainer = Trainer(model=self.model)
            
            # Get predictions
            predictions = trainer.predict(dataset).predictions
            predicted_class_ids = np.argmax(predictions, axis=1)
            
            # Convert class IDs back to original labels
            reverse_label_map = {v: k for k, v in self.label_map.items()}
            predicted_labels = [reverse_label_map[class_id] for class_id in predicted_class_ids]
            
            return predicted_labels
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
            # Check if model is loaded
            if self.model is None:
                raise ValueError("Model not trained or loaded. Please train or load a model first.")
            
            # Prepare dataset without labels
            dataset = self._prepare_dataset(texts)
            
            # Set up trainer for prediction
            trainer = Trainer(model=self.model)
            
            # Get predictions
            predictions = trainer.predict(dataset).predictions
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(torch.tensor(predictions), dim=1).numpy()
            
            return probabilities
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
            print(f"Error evaluating model: {str(e)}")
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
            
            # Save model and tokenizer
            if self.model is not None:
                self.model.save_pretrained(os.path.join(directory, 'model'))
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(os.path.join(directory, 'tokenizer'))
            
            # Save configuration
            config = {
                'model_name': self.model_name,
                'num_labels': self.num_labels,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs,
                'max_length': self.max_length,
                'device': self.device,
                'label_map': self.label_map,
                'training_args': self.training_args
            }
            
            with open(os.path.join(directory, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    @classmethod
    def load(cls, directory: str) -> 'DenseClassifierEngine':
        """Load a model from disk.
        
        Args:
            directory: Directory path containing the saved model.
            
        Returns:
            DenseClassifierEngine: Loaded classification engine instance.
        """
        try:
            # Load configuration
            with open(os.path.join(directory, 'config.json'), 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Create engine instance
            engine = cls(
                model_name=config['model_name'],
                num_labels=config['num_labels'],
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate'],
                epochs=config['epochs'],
                max_length=config['max_length'],
                device=config['device'],
                **config.get('training_args', {})
            )
            
            # Set label map
            engine.label_map = config['label_map']
            
            # Load tokenizer and model
            engine.tokenizer = AutoTokenizer.from_pretrained(os.path.join(directory, 'tokenizer'))
            engine.model = AutoModelForSequenceClassification.from_pretrained(os.path.join(directory, 'model'))
            engine.model.to(engine.device)
            
            return engine
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None