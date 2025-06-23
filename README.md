# Table Classifier API

A FastAPI-based service for classifying table images using a deep learning model.

## Project Structure

```
.
├── api/
│   ├── api.py                # Main FastAPI server
│   ├── model_predictor.py    # Model loading and prediction logic
│   ├── ProductionConfig.py   # Production configuration (paths, limits, etc.)
│   ├── miscellaneous.py      # Client class and utilities
│   ├── client_example.py     # Example Python client
│   ├── test_client.py        # API test script
│   ├── test_model.py         # Model test script
│   ├── class_indices.json    # Class label mapping
│   ├── table_classifier.h5   # Trained model file
│   ├── test_images/          # Test images for API testing
│   └── __pycache__/          # Python cache files
├── requirements.txt          # Python dependencies
├── .gitignore
└── ...
```

## Setup

1. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

2. **Model files**

   - Place your trained model (`table_classifier.h5`) and class mapping (`class_indices.json`) in the `api/` directory.

3. **Configuration**

   - Adjust settings in [`api/ProductionConfig.py`](api/ProductionConfig.py) if needed (model paths, allowed extensions, etc).

## Running the API

From the root directory, start the API server:

```sh
cd api
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## API Endpoints

- `GET /` — Health check
- `GET /health` — Detailed health check (model status, classes)
- `GET /classes` — List available classes
- `POST /predict` — Predict class for a single image (`multipart/form-data`, field: `file`)
- `POST /predict_batch` — Predict classes for multiple images

## Example Usage

### Python Client

See [`api/client_example.py`](api/client_example.py) for a usage example:

```python
from miscellaneous import TableClassifierClient

client = TableClassifierClient()
print(client.health_check())
print(client.get_classes())
result = client.predict_image("path/to/image.jpg")
print(result)
```

### Testing

- Use [`api/test_client.py`](api/test_client.py) to test the API endpoints.
- Place test images in [`api/test_images/`](api/test_images/).

## Notes

- Only the `api/` folder is used for production deployment.
- Model and class mapping files must be present in the `api/` directory.
- See [`requirements.txt`](requirements.txt) for required Python packages.

---

**License:** MIT (add your license here)