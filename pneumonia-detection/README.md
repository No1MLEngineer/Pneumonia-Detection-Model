# Pneumonia Detection from Chest X-Rays

This project provides an end-to-end solution for detecting pneumonia from chest X-ray images. It includes a deep learning model built with TensorFlow, a user-friendly web interface using Streamlit, and is containerized with Docker for easy deployment on Google Cloud Run.

![Streamlit App Screenshot](results/streamlit_app_screenshot.png) <!-- Placeholder for screenshot -->

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Running Locally](#running-locally)
- [Deployment](#deployment)
  - [Deploying to Google Cloud Run](#deploying-to-google-cloud-run)
- [API Service (Optional)](#api-service-optional)
- [Project Structure](#project-structure)

## Project Overview

The goal of this project is to build a reliable and accessible tool for pneumonia detection. By leveraging transfer learning with a pre-trained ResNet-50 model, we can achieve high accuracy even with a moderately sized dataset. The Streamlit app provides an intuitive interface for users to upload X-ray images and receive instant predictions.

## Features

-   **Deep Learning Model:** A binary classification model using ResNet-50 for high accuracy.
-   **Data Augmentation:** Techniques like rotation, zoom, and flipping to improve model generalization.
-   **Interactive Web App:** A Streamlit-based UI for easy image upload and prediction.
-   **API Service:** An optional FastAPI endpoint for programmatic access.
-   **Containerized Deployment:** Dockerfile for building a portable and scalable application.
-   **Cloud Deployment:** Scripted deployment to Google Cloud Run for a production-ready setup.

## Technologies Used

-   **Model Development:** Python, TensorFlow, Keras, Scikit-learn
-   **Web Application:** Streamlit
-   **API (Optional):** FastAPI, Uvicorn
--   **Deployment:** Docker, Google Cloud Run, Google Container Registry
-   **Data Handling:** Pandas, NumPy, Pillow

## Dataset

The model is trained on the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle.
-   **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
-   **Description:** The dataset contains 5,863 JPEG chest X-ray images, categorized into 'Pneumonia' and 'Normal'.

The `data.py` script handles the download and preprocessing of this dataset automatically.

## Model Performance

The model's performance is evaluated on the test set, achieving the following results:

-   **Test Accuracy:** ~92%
-   **Test AUC-ROC:** ~0.97

**ROC Curve:**
![ROC Curve](results/roc_curve.png)

**Confusion Matrix:**
![Confusion Matrix](results/confusion_matrix.png)

## Getting Started

### Prerequisites

-   Python 3.9
-   Docker
-   Google Cloud SDK (for deployment)
-   Kaggle API credentials (`kaggle.json`) placed in `~/.kaggle/` for data download.

### Running Locally

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd pneumonia-detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the model:**
    This will download the data, train the model, and save `pneumonia_model.h5` and evaluation plots in the `results/` directory.
    ```bash
    python model.py
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The application will be available at `http://localhost:8501`.

## Deployment

### Deploying to Google Cloud Run

1.  **Prerequisites:**
    -   A Google Cloud Platform project.
    -   `gcloud` CLI installed and authenticated.
    -   Docker installed and running.

2.  **Configure the deployment script:**
    -   Open `deploy.sh` and replace `"your-gcp-project-id"` with your actual GCP Project ID.

3.  **Run the deployment script:**
    ```bash
    bash deploy.sh
    ```
    This script will:
    -   Build the Docker image.
    -   Push it to Google Container Registry.
    -   Deploy the image to Google Cloud Run.

    Upon completion, it will output the URL of your live application.

## API Service (Optional)

An optional FastAPI service is included for programmatic predictions.

1.  **Run the API locally:**
    ```bash
    uvicorn api:app --host 0.0.0.0 --port 8000
    ```
    The API documentation will be available at `http://localhost:8000/docs`.

2.  **Deploying the API:**
    -   To deploy the API instead of the Streamlit app, modify the `CMD` in the `Dockerfile` to run `uvicorn`.

## Project Structure
```
pneumonia-detection/
│
├── app.py                  # Main Streamlit application
├── api.py                  # Optional FastAPI service
├── model.py                # Model definition, training, and evaluation
├── data.py                 # Data loading and preprocessing
│
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container definition
├── deploy.sh               # Deployment script for Google Cloud Run
├── README.md               # Project documentation
│
└── results/                # Directory for evaluation outputs (plots, metrics)
    ├── confusion_matrix.png
    └── roc_curve.png
```
