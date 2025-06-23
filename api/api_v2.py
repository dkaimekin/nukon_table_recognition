# api.py - Standalone FastAPI server
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from typing import Optional, List
import logging
import pandas as pd
import os
from pathlib import Path
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
TABLES_FOLDER = "tables"  # Folder containing CSV files

def load_csv_table(class_name: str, return_format: str = "json"):
    """
    Load CSV table based on predicted class
    
    Args:
        class_name: Name of the predicted class
        return_format: Format to return data in ("json", "dict", "records")
    
    Returns:
        Table data in specified format
    """
    try:
        csv_path = Path(TABLES_FOLDER) / f"{class_name}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        if return_format == "json":
            return df.to_json(orient="records", indent=2)
        elif return_format == "dict":
            return df.to_dict(orient="records")
        elif return_format == "records":
            return df.to_dict(orient="records")
        else:
            return df.to_dict(orient="records")
            
    except Exception as e:
        raise ValueError(f"Error loading CSV for {class_name}: {str(e)}")

def get_available_tables():
    """Get list of available CSV tables"""
    tables_path = Path(TABLES_FOLDER)
    if not tables_path.exists():
        return []
    
    csv_files = list(tables_path.glob("*.csv"))
    return [f.stem for f in csv_files]  # Return filename without extension

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
        
        # Check tables folder
        tables_path = Path(TABLES_FOLDER)
        if tables_path.exists():
            available_tables = get_available_tables()
            logger.info(f"Available CSV tables: {available_tables}")
        else:
            logger.warning(f"Tables folder '{TABLES_FOLDER}' not found. Creating it...")
            tables_path.mkdir(exist_ok=True)
            
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

@app.get("/tables")
async def get_tables():
    """Get list of available CSV tables"""
    try:
        available_tables = get_available_tables()
        return {
            "available_tables": available_tables,
            "tables_folder": TABLES_FOLDER,
            "count": len(available_tables)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing tables: {str(e)}")

@app.get("/tables/{table_name}")
async def get_table_data(table_name: str):
    """
    Get CSV table data by name
    
    Args:
        table_name: Name of the table (without .csv extension)
    
    Returns:
        CSV data as JSON
    """
    try:
        table_data = load_csv_table(table_name, return_format="dict")
        return {
            "table_name": table_name,
            "data": table_data,
            "row_count": len(table_data)
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/tables/{table_name}/download")
async def download_table(table_name: str):
    """
    Download CSV table as file
    
    Args:
        table_name: Name of the table (without .csv extension)
    
    Returns:
        CSV file for download
    """
    try:
        csv_path = Path(TABLES_FOLDER) / f"{table_name}.csv"
        
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
        
        return FileResponse(
            path=str(csv_path),
            media_type='text/csv',
            filename=f"{table_name}.csv"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading table: {str(e)}")
    
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
    top_k: Optional[int] = 3,
    include_table: Optional[bool] = True,
    table_format: Optional[str] = "dict"
):
    """
    Predict table class from uploaded image and optionally return corresponding CSV data
    
    Args:
        file: Image file to classify
        top_k: Number of top predictions to return (default: 3)
        include_table: Whether to include CSV table data for the top prediction (default: True)
        table_format: Format for table data - "dict", "json", or "records" (default: "dict")
    
    Returns:
        JSON response with predictions and optionally table data
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
        
        # Include CSV table data if requested
        if include_table:
            top_class = result['top_prediction']['class']
            try:
                table_data = load_csv_table(top_class, return_format=table_format)
                result['table_data'] = {
                    "table_name": top_class,
                    "data": table_data,
                    "format": table_format
                }
                if isinstance(table_data, list):
                    result['table_data']['row_count'] = len(table_data)
                
                logger.info(f"Included table data for class: {top_class}")
                
            except Exception as e:
                logger.warning(f"Could not load table for {top_class}: {e}")
                result['table_data'] = {
                    "table_name": top_class,
                    "error": f"Table not available: {str(e)}",
                    "available_tables": get_available_tables()
                }
        
        logger.info(f"Prediction made for {file.filename}: {result['top_prediction']['class']} ({result['top_prediction']['confidence']:.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    include_tables: Optional[bool] = True
):
    """
    Predict table classes for multiple images and optionally return corresponding CSV data
    
    Args:
        files: List of image files to classify
        include_tables: Whether to include CSV table data for predictions (default: True)
    
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
            
            # Include CSV table data if requested
            if include_tables:
                top_class = result['top_prediction']['class']
                try:
                    table_data = load_csv_table(top_class, return_format="dict")
                    result['table_data'] = {
                        "table_name": top_class,
                        "data": table_data,
                        "row_count": len(table_data)
                    }
                except Exception as e:
                    result['table_data'] = {
                        "table_name": top_class,
                        "error": f"Table not available: {str(e)}"
                    }
            
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