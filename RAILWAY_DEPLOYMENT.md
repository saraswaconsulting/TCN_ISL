# 🚀 Railway Deployment Guide for ISL Demo

## 📋 Pre-Deployment Checklist

### ✅ Files Created:
- [x] `gradio_isl_demo.py` - Main Gradio application
- [x] `requirements.txt` - Python dependencies
- [x] `Dockerfile` - Container configuration
- [x] `railway.toml` - Railway configuration
- [x] `.env.example` - Environment variables template
- [x] Updated `.gitignore` - Security for sensitive files

## 🔐 Security Setup

### 1. Environment Variables (CRITICAL!)
In Railway dashboard, set these variables:

```bash
# REQUIRED: Your Gemini API key
GEMINI_API_KEY=your_actual_gemini_api_key_here

# OPTIONAL: Model path (defaults to checkpoints/best_gru.pt)
MODEL_PATH=checkpoints/best_gru.pt

# OPTIONAL: Port (Railway auto-sets this)
PORT=7860
```

### 2. Model File Setup
Ensure your model file exists:
```bash
checkpoints/best_gru.pt
```

## 🚀 Deployment Steps

### Option 1: GitHub + Railway (Recommended)

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Railway deployment files"
   git push origin main
   ```

2. **Deploy on Railway:**
   - Go to [railway.app](https://railway.app)
   - Sign up/login with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your ISL repository
   - Railway will auto-detect the Dockerfile and deploy

3. **Set Environment Variables:**
   - In Railway dashboard, go to your project
   - Click "Variables" tab
   - Add: `GEMINI_API_KEY` = `your_actual_api_key`

4. **Monitor Deployment:**
   - Check "Deployments" tab for build logs
   - Once deployed, get your public URL

### Option 2: Railway CLI

1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and Deploy:**
   ```bash
   railway login
   railway init
   railway up
   ```

3. **Set Environment Variables:**
   ```bash
   railway variables set GEMINI_API_KEY=your_actual_api_key
   ```

## 🔧 Troubleshooting

### Common Issues:

#### 1. **Model File Too Large**
```bash
# If checkpoints/best_gru.pt > 100MB, use Git LFS:
git lfs track "*.pt"
git add .gitattributes
git add checkpoints/best_gru.pt
git commit -m "Add model with Git LFS"
```

#### 2. **Memory Issues**
```bash
# In Railway dashboard, upgrade to Hobby plan ($5/month) for more RAM
```

#### 3. **Gemini API Errors**
```bash
# Verify your API key in Railway variables
# Check API key has proper permissions
```

#### 4. **MediaPipe Issues**
```bash
# The Dockerfile includes all necessary system dependencies
# If issues persist, try opencv-python-headless==4.8.1.78
```

## 📊 Expected Performance

### Railway Specs:
- **Free Tier**: 512MB RAM, 1vCPU
- **Hobby Tier**: 8GB RAM, 8vCPU ($5/month)

### Performance:
- **Model Loading**: 10-30 seconds
- **Video Processing**: 2-5 seconds per video
- **Real-time**: Works well with Hobby tier

## 🌐 Access Your Demo

Once deployed, you'll get a URL like:
```
https://your-app-name.railway.app
```

## 🔄 Updates

To update your deployed app:
```bash
git add .
git commit -m "Update ISL demo"
git push origin main
```

Railway will automatically redeploy!

## 💡 Tips for Success

1. **Start with Hobby Tier**: Free tier might be too limited for the model
2. **Monitor Logs**: Use Railway dashboard to watch deployment logs
3. **Test Locally First**: Run `python gradio_isl_demo.py` locally
4. **Keep API Key Secret**: Never commit actual API keys to GitHub

## 🆘 Support

If deployment fails:
1. Check Railway deployment logs
2. Verify all environment variables are set
3. Ensure model file exists and is accessible
4. Check that requirements.txt includes all dependencies

## 🎉 Success!

Your ISL demo will be live at your Railway URL with:
- ✅ Secure Gemini API integration
- ✅ Real-time video processing
- ✅ Beautiful Gradio interface
- ✅ Automatic scaling
- ✅ HTTPS by default

Enjoy your deployed ISL translator! 🤟