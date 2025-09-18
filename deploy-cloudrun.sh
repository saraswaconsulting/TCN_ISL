#!/bin/bash
# Quick deploy script for Google Cloud Run
# Run this from your project root directory

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-asia-southeast1}"
SERVICE_NAME="${SERVICE_NAME:-isl-translation-demo}"
REPO_NAME="${REPO_NAME:-isl-demo}"
IMAGE_NAME="${IMAGE_NAME:-isl-demo}"

echo "🚀 Deploying ISL Demo to Google Cloud Run"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo ""

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first."
    exit 1
fi

# Set project and region
echo "📋 Setting up project and region..."
gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

# Create Artifact Registry repository
echo "📦 Creating Artifact Registry repository..."
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="ISL demo images" \
    --async || echo "Repository already exists, continuing..."

# Build and push image
echo "🔨 Building and pushing Docker image..."
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME"
gcloud builds submit --tag $IMAGE_URI

# Check if secret exists, create if not
echo "🔐 Setting up Gemini API key secret..."
if ! gcloud secrets describe GEMINI_API_KEY &> /dev/null; then
    echo "Please enter your Gemini API key:"
    read -s GEMINI_API_KEY_VALUE
    echo -n "$GEMINI_API_KEY_VALUE" | gcloud secrets create GEMINI_API_KEY --data-file=-
else
    echo "Secret GEMINI_API_KEY already exists. Update it manually if needed."
fi

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_URI \
    --region $REGION \
    --allow-unauthenticated \
    --port 7860 \
    --cpu 1 \
    --memory 2Gi \
    --concurrency 10 \
    --set-env-vars MODEL_PATH=checkpoints/best_gru.pt,PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1 \
    --update-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest

# Get service URL
echo "✅ Deployment complete!"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format "value(status.url)")
echo ""
echo "🌐 Your app is available at:"
echo "   Health check: $SERVICE_URL/"
echo "   UI: $SERVICE_URL/app"
echo ""
echo "📊 To view logs:"
echo "   gcloud logs tail --service $SERVICE_NAME --region $REGION"