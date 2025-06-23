# requirements.txt
"""
tensorflow==2.15.0
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pillow==10.1.0
numpy==1.24.3
"""

# docker/Dockerfile
"""
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# client_example.py - Example client code
import requests
import json
from pathlib import Path

class TableClassifierClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self):
        """Check if API is running"""
        try:
            response = requests.get(f"{self.base_url}/")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_classes(self):
        """Get available classes"""
        try:
            response = requests.get(f"{self.base_url}/classes")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_image(self, image_path: str, top_k: int = 3):
        """
        Predict table class for single image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
        
        Returns:
            Prediction results
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                params = {'top_k': top_k}
                response = requests.post(
                    f"{self.base_url}/predict",
                    files=files,
                    params=params
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def predict_batch(self, image_paths: list):
        """
        Predict table classes for multiple images
        
        Args:
            image_paths: List of paths to image files
        
        Returns:
            Batch prediction results
        """
        try:
            files = []
            for path in image_paths:
                with open(path, 'rb') as f:
                    files.append(
                        ('files', (Path(path).name, f.read(), 'image/jpeg'))
                    )
            
            response = requests.post(
                f"{self.base_url}/predict_batch",
                files=files
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"error": str(e)}

# Usage examples
def example_usage():
    """Example usage of the client"""
    client = TableClassifierClient()
    
    print("=== Table Classifier Client Example ===")
    
    # Health check
    print("1. Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    print()
    
    # Get classes
    print("2. Available Classes:")
    classes = client.get_classes()
    print(json.dumps(classes, indent=2))
    print()
    
    # Single prediction (replace with actual image path)
    image_path = "/Users/daniilkaimekin/Projects/nurik_forensics/dl/data/adilbayev/photo_5384098555213509461_y.jpg"  # Replace with your test image
    if Path(image_path).exists():
        print(f"3. Single Prediction for {image_path}:")
        result = client.predict_image(image_path)
        print(json.dumps(result, indent=2))
        print()
    
    # Batch prediction (replace with actual image paths)
    batch_images = ["image1.jpg", "image2.jpg"]  # Replace with your images
    existing_images = [img for img in batch_images if Path(img).exists()]
    
    if existing_images:
        print("4. Batch Prediction:")
        batch_result = client.predict_batch(existing_images)
        print(json.dumps(batch_result, indent=2))