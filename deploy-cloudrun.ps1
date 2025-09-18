# Quick Deploy to Google Cloud Run
# For Windows PowerShell

# Set your project ID and run this script
$env:PROJECT_ID = "your-project-id-here"
$env:REGION = "asia-southeast1"
$env:SERVICE_NAME = "isl-translation-demo"
$env:REPO_NAME = "isl-demo"
$env:IMAGE_NAME = "isl-demo"

Write-Host "🚀 Deploying ISL Demo to Google Cloud Run" -ForegroundColor Green
Write-Host "Project: $env:PROJECT_ID"
Write-Host "Region: $env:REGION"
Write-Host "Service: $env:SERVICE_NAME"
Write-Host ""

# Set project and region
Write-Host "📋 Setting up project and region..." -ForegroundColor Yellow
gcloud config set project $env:PROJECT_ID
gcloud config set run/region $env:REGION

# Enable required APIs
Write-Host "🔧 Enabling required APIs..." -ForegroundColor Yellow
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

# Create Artifact Registry repository
Write-Host "📦 Creating Artifact Registry repository..." -ForegroundColor Yellow
try {
    gcloud artifacts repositories create $env:REPO_NAME --repository-format=docker --location=$env:REGION --description="ISL demo images" --async
} catch {
    Write-Host "Repository already exists, continuing..." -ForegroundColor Gray
}

# Build and push image
Write-Host "🔨 Building and pushing Docker image..." -ForegroundColor Yellow
$env:IMAGE_URI = "$env:REGION-docker.pkg.dev/$env:PROJECT_ID/$env:REPO_NAME/$env:IMAGE_NAME"
gcloud builds submit --tag $env:IMAGE_URI

# Check if secret exists
Write-Host "🔐 Setting up Gemini API key secret..." -ForegroundColor Yellow
try {
    gcloud secrets describe GEMINI_API_KEY | Out-Null
    Write-Host "Secret GEMINI_API_KEY already exists. Update it manually if needed." -ForegroundColor Gray
} catch {
    $env:GEMINI_API_KEY_VALUE = Read-Host "Please enter your Gemini API key" -AsSecureString
    $env:GEMINI_API_KEY_VALUE = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($env:GEMINI_API_KEY_VALUE))
    $env:TEMP_FILE = "$PWD\temp_key.txt"
    Set-Content -Path $env:TEMP_FILE -Value $env:GEMINI_API_KEY_VALUE -NoNewline
    gcloud secrets create GEMINI_API_KEY --data-file=$env:TEMP_FILE
    Remove-Item $env:TEMP_FILE
}

# Deploy to Cloud Run
Write-Host "🚀 Deploying to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $env:SERVICE_NAME `
    --image $env:IMAGE_URI `
    --region $env:REGION `
    --allow-unauthenticated `
    --port 7860 `
    --cpu 1 `
    --memory 2Gi `
    --concurrency 10 `
    --set-env-vars MODEL_PATH=checkpoints/best_gru.pt,PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1 `
    --update-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest

# Get service URL
Write-Host "✅ Deployment complete!" -ForegroundColor Green
$env:SERVICE_URL = gcloud run services describe $env:SERVICE_NAME --region $env:REGION --format "value(status.url)"
Write-Host ""
Write-Host "🌐 Your app is available at:" -ForegroundColor Cyan
Write-Host "   Health check: $env:SERVICE_URL/"
Write-Host "   UI: $env:SERVICE_URL/app"
Write-Host ""
Write-Host "📊 To view logs:" -ForegroundColor Cyan
Write-Host "   gcloud logs tail --service $env:SERVICE_NAME --region $env:REGION"