from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

# --- Configuration ---
MODEL_PATH = 'pneumonia_model.h5'
IMAGE_SIZE = (224, 224)

# --- App Initialization ---
app = FastAPI(
    title="Pneumonia Detection API",
    description="An API to predict pneumonia from chest X-ray images using a ResNet-50 model.",
    version="1.0.0"
)

# --- Model Loading ---
model = None

@app.on_event("startup")
def load_pneumonia_model():
    """Load the model at startup."""
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)

# --- Image Preprocessing ---
def preprocess_image(image_bytes):
    """Preprocess the image for the model."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

# --- API Endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file and returns a prediction for pneumonia.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail="Unsupported file type. Please upload a JPEG or PNG image.")

    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    prediction = model.predict(processed_image)
    confidence = float(prediction[0][0])

    if confidence > 0.5:
        label = "Pneumonia"
        score = confidence
    else:
        label = "Normal"
        score = 1 - confidence

    return JSONResponse(content={
        "prediction": label,
        "confidence": score
    })

# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
    """Health check endpoint to verify service is running."""
    return {"status": "ok"}

# To run this API:
# uvicorn api:app --host 0.0.0.0 --port 8000
