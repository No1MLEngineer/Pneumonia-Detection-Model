import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = 'pneumonia_model.h5'
IMAGE_SIZE = (224, 224)

# --- Model Loading ---
@st.cache(allow_output_mutation=True)
def load_pneumonia_model():
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        return None
    model = load_model(MODEL_PATH)
    return model

model = load_pneumonia_model()

# --- Image Preprocessing ---
def preprocess_image(image):
    """Preprocess the uploaded image for the model."""
    image = image.resize(IMAGE_SIZE)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize to [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# --- UI ---
st.title("Pneumonia Detection from Chest X-Rays")

# Sidebar for instructions and information
with st.sidebar:
    st.header("About")
    st.write("""
        This application uses a deep learning model (ResNet-50) to predict
        the presence of pneumonia from chest X-ray images.
    """)
    st.header("Instructions")
    st.write("""
        1.  Upload a chest X-ray image (JPEG or PNG).
        2.  The model will predict whether the image indicates 'Normal' or 'Pneumonia'.
        3.  A confidence score for the prediction will also be displayed.
    """)

# Main content area for image upload and prediction
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-Ray', use_column_width=True)

    if model is not None:
        with st.spinner('Analyzing the image...'):
            # Preprocess and predict
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            confidence = prediction[0][0]

            # Display the result
            st.subheader("Prediction")
            if confidence > 0.5:
                st.write(f"**Result:** Pneumonia")
                st.write(f"**Confidence:** {confidence:.2%}")
            else:
                st.write(f"**Result:** Normal")
                st.write(f"**Confidence:** {1 - confidence:.2%}")

            # Show a progress bar for confidence
            st.progress(confidence)

    else:
        st.warning("Model is not loaded. Cannot perform prediction.")

# Error handling for file format
elif uploaded_file is not None:
    st.error("Invalid file format. Please upload a JPG, JPEG, or PNG image.")
