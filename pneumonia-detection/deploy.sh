#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Google Cloud Project ID
PROJECT_ID="your-gcp-project-id"
# Google Cloud Region
REGION="us-central1"
# Name of the service on Cloud Run
SERVICE_NAME="pneumonia-detection-app"
# Name of the Docker image
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# --- Script ---

echo "--- Authenticating with Google Cloud ---"
# Authenticate gcloud, if not already done.
gcloud auth login
gcloud config set project $PROJECT_ID

echo "--- Enabling required Google Cloud services ---"
# Enable APIs for Container Registry and Cloud Run.
gcloud services enable containerregistry.googleapis.com
gcloud services enable run.googleapis.com

echo "--- Building the Docker image ---"
# Build the Docker image locally.
docker build -t $IMAGE_NAME .

echo "--- Pushing the Docker image to Google Container Registry ---"
# Push the image to GCR.
docker push $IMAGE_NAME

echo "--- Deploying the service to Google Cloud Run ---"
# Deploy the container to Cloud Run.
# --platform managed: Use the fully managed version of Cloud Run.
# --allow-unauthenticated: Allow public access to the service.
# --port 8501: Specify the port the container listens on (8501 for Streamlit).
# --memory 2Gi: Allocate sufficient memory for the model.
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8501 \
  --memory 2Gi

echo "--- Deployment complete ---"
# Get the URL of the deployed service.
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')
echo "Service is available at: $SERVICE_URL"
