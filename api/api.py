# api.py - Standalone FastAPI server
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional, List
import logging
from model_predictor import TableClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Table Classifier API",
    description="API for classifying table images",
    version="1.0.0"
)

# Global classifier instance
classifier = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global classifier
    try:
        classifier = TableClassifier(
            model_path="table_classifier.h5",
            class_indices_path="class_indices.json"
        )
        logger.info("Model loaded successfully")
        logger.info(f"Available classes: {list(classifier.class_indices.keys())}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "message": "Table Classifier API is running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if classifier is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": list(classifier.class_indices.keys()),
        "num_classes": len(classifier.class_indices)
    }

@app.get("/classes")
async def get_classes():
    """Get available classes"""
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "classes": list(classifier.class_indices.keys()),
        "num_classes": len(classifier.class_indices),
        "class_mapping": classifier.class_indices
    }

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    top_k: Optional[int] = 3
):
    """
    Predict table class from uploaded image
    
    Args:
        file: Image file to classify
        top_k: Number of top predictions to return (default: 3)
    
    Returns:
        JSON response with predictions
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"File must be an image. Received: {file.content_type}"
        )
    
    # Validate top_k parameter
    if top_k < 1 or top_k > len(classifier.class_indices):
        top_k = min(3, len(classifier.class_indices))
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Make prediction
        result = classifier.predict_from_bytes(image_bytes, top_k=top_k)
        
        # Add metadata
        result.update({
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(image_bytes),
            "status": "success"
        })
        
        logger.info(f"Prediction made for {file.filename}: {result['top_prediction']['class']} ({result['top_prediction']['confidence']:.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict table classes for multiple images
    
    Args:
        files: List of image files to classify
    
    Returns:
        JSON response with predictions for each image
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400, 
            detail="Maximum 10 files allowed per batch"
        )
    
    results = []
    
    for file in files:
        if not file.content_type or not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": f"File must be an image. Received: {file.content_type}"
            })
            continue
        
        try:
            image_bytes = await file.read()
            result = classifier.predict_from_bytes(image_bytes)
            result.update({
                "filename": file.filename,
                "status": "success"
            })
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )