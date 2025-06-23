from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import hashlib
import json
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO

app = FastAPI()

# --- Model Setup ---
model = tf.keras.models.load_model("path_to_your_model.h5")  # Replace with your model
class_indices = {"table_1": 0, "table_2": 1}  # Replace with your class mapping
class_to_csv = {0: "table_1.csv", 1: "table_2.csv"}  # Map class IDs to CSV files

# --- Cache Setup ---
CACHE_FILE = "cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

cache = load_cache()

# --- Image Preprocessing ---
def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# --- API Endpoint ---
@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        # Read image and compute MD5
        image_data = await file.read()
        md5_hash = hashlib.md5(image_data).hexdigest()

        # Check cache
        if md5_hash in cache:
            return JSONResponse({"csv": cache[md5_hash], "cached": True})

        # Preprocess and classify
        img_array = preprocess_image(image_data)
        pred = model.predict(img_array)
        class_id = np.argmax(pred[0])
        csv_file = class_to_csv[class_id]

        # Update cache
        cache[md5_hash] = csv_file
        save_cache(cache)

        return JSONResponse({"csv": csv_file, "cached": False})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))