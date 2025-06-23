# model_predictor.py - Core prediction logic
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from typing import Dict, List, Tuple, Optional

class TableClassifier:
    def __init__(self, model_path: str, class_indices_path: str):
        """
        Initialize the classifier with trained model and class indices
        
        Args:
            model_path: Path to the saved model (.h5 file)
            class_indices_path: Path to class_indices.json file
        """
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class indices
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
        
        # Create reverse mapping (index -> class name)
        self.index_to_class = {v: k for k, v in self.class_indices.items()}
        
        # Model input shape
        self.input_shape = (224, 224, 3)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load and resize image
            image = Image.open(image_path)
            image = image.convert('RGB')  # Ensure RGB format
            image = image.resize((224, 224))
            
            # Convert to array and normalize
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            
            return image_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")
    
    def preprocess_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image from bytes (for API use)
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Preprocessed image array
        """
        try:
            from io import BytesIO
            
            # Load image from bytes
            image = Image.open(BytesIO(image_bytes))
            image = image.convert('RGB')
            image = image.resize((224, 224))
            
            # Convert to array and normalize
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image from bytes: {str(e)}")
    
    def predict(self, image_path: str, top_k: int = 3) -> Dict:
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        try:
            # Preprocess image
            image_array = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(image_array, verbose=0)
            
            # Get top k predictions
            top_indices = np.argsort(predictions[0])[::-1][:top_k]
            
            results = {
                'predictions': [],
                'top_prediction': {
                    'class': self.index_to_class[top_indices[0]],
                    'confidence': float(predictions[0][top_indices[0]])
                }
            }
            
            # Add all top k predictions
            for idx in top_indices:
                results['predictions'].append({
                    'class': self.index_to_class[idx],
                    'confidence': float(predictions[0][idx])
                })
            
            return results
            
        except Exception as e:
            raise ValueError(f"Error making prediction: {str(e)}")
    
    def predict_from_bytes(self, image_bytes: bytes, top_k: int = 3) -> Dict:
        """
        Make prediction from image bytes (for API use)
        
        Args:
            image_bytes: Image data as bytes
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        try:
            # Preprocess image
            image_array = self.preprocess_image_from_bytes(image_bytes)
            
            # Make prediction
            predictions = self.model.predict(image_array, verbose=0)
            
            # Get top k predictions
            top_indices = np.argsort(predictions[0])[::-1][:top_k]
            
            results = {
                'predictions': [],
                'top_prediction': {
                    'class': self.index_to_class[top_indices[0]],
                    'confidence': float(predictions[0][top_indices[0]])
                }
            }
            
            # Add all top k predictions
            for idx in top_indices:
                results['predictions'].append({
                    'class': self.index_to_class[idx],
                    'confidence': float(predictions[0][idx])
                })
            
            return results
            
        except Exception as e:
            raise ValueError(f"Error making prediction: {str(e)}")
        

