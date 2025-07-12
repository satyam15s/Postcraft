# Alternative Deployment Methods

## 🚨 Azure AI Foundry CLI Not Available

The `az ai-foundry` CLI extension is not yet available. Here are alternative deployment methods:

## 🎯 Option 1: Azure Portal Deployment (Recommended)

### Step 1: Create AI Foundry Resource via Portal
1. Go to [Azure Portal](https://portal.azure.com)
2. Search for "AI Foundry" 
3. Click "Create" → "AI Foundry"
4. Fill in details:
   - **Resource Group**: `video-generation-rg`
   - **Workspace Name**: `video-generation-workspace`
   - **Region**: `East US`
   - **Compute**: `Standard_DS3_v2` (or smaller for MVP)

### Step 2: Deploy Models via Portal
1. Navigate to your AI Foundry workspace
2. Go to "Models" section
3. Click "Deploy Model"
4. Deploy these models:
   - **Stable Video Diffusion**: `stability-ai/stable-video-diffusion-img2vid-xt`
   - **GPT-4 Vision**: `gpt-4-vision-preview`

## 🎯 Option 2: Azure Machine Learning (Alternative)

### Use Azure ML for Model Deployment
```bash
# Install Azure ML CLI
az extension add --name ml

# Create ML workspace
az ml workspace create \
  --name "video-ml-workspace" \
  --resource-group "video-generation-rg" \
  --location "eastus"

# Deploy models using Azure ML
az ml model deploy \
  --name "stable-video-diffusion" \
  --model "stability-ai/stable-video-diffusion-img2vid-xt" \
  --workspace "video-ml-workspace" \
  --resource-group "video-generation-rg"
```

## 🎯 Option 3: Pre-hosted Models (Easiest)

### Use Azure OpenAI + Pre-hosted Models
The app is designed to work with pre-hosted models. Just set these environment variables:

```env
# Azure OpenAI (already working)
AZURE_OPENAI_TEXT_ENDPOINT=https://your-text-endpoint.openai.azure.com/
AZURE_OPENAI_TEXT_KEY=your-text-key
AZURE_OPENAI_IMAGE_ENDPOINT=https://your-image-endpoint.openai.azure.com/
AZURE_OPENAI_IMAGE_KEY=your-image-key

# Pre-hosted model endpoints (pay-per-use)
SVD_ENDPOINT=https://api.aifoundry.azure.com/v1/models/stability-ai/stable-video-diffusion-img2vid-xt
GPT4V_ENDPOINT=https://api.aifoundry.azure.com/v1/models/gpt-4-vision-preview

# Placeholder for AI Foundry (app will use simulated responses)
AZURE_AI_FOUNDRY_ENDPOINT=https://placeholder.aifoundry.azure.com
AZURE_AI_FOUNDRY_API_KEY=placeholder-key
```

## 🎯 Option 4: Simulated Mode (Recommended for MVP)

### No Deployment Required
The app works perfectly with simulated responses. Just run:

```bash
streamlit run postcraft_azure_ready.py
```

The app will:
- ✅ Use DALL-E 3 for image generation (already working)
- ✅ Use GPT-4 for text generation (already working)
- ✅ Simulate video generation responses
- ✅ Provide full user experience

## 🚀 Quick Start - Simulated Mode

### 1. Set Environment Variables
```env
# Your existing Azure OpenAI credentials
AZURE_OPENAI_TEXT_ENDPOINT=https://your-text-endpoint.openai.azure.com/
AZURE_OPENAI_TEXT_KEY=your-text-key
AZURE_OPENAI_IMAGE_ENDPOINT=https://your-image-endpoint.openai.azure.com/
AZURE_OPENAI_IMAGE_KEY=your-image-key

# Placeholder for AI Foundry (simulated mode)
AZURE_AI_FOUNDRY_ENDPOINT=https://placeholder.aifoundry.azure.com
AZURE_AI_FOUNDRY_API_KEY=placeholder-key
```

### 2. Run the App
```bash
streamlit run postcraft_azure_ready.py
```

### 3. Test Video Generation
- Go to "🎬 Video Reel" tab
- Upload an image or use DALL-E 3 generation
- Click "Generate Video Reel"
- See simulated video generation pipeline

## 💰 Cost Analysis

### Simulated Mode (Recommended for MVP)
- **DALL-E 3**: ~$0.04-0.08 per image
- **GPT-4**: ~$0.01-0.02 per caption
- **Video Generation**: Simulated (free)
- **Total per video reel**: ~$0.05-0.10

### Real Deployment (When Available)
- **Stable Video Diffusion**: ~$0.10-0.20 per video
- **GPT-4 Vision**: ~$0.01-0.03 per caption
- **Total per video reel**: ~$0.11-0.23

## 🔧 Testing the Pipeline

The app includes comprehensive simulated responses that demonstrate:

1. **Frame Extraction** → DALL-E 3 variations
2. **Frame Ranking** → Simulated scoring
3. **Caption Generation** → GPT-4 text
4. **Caption Polishing** → Simulated enhancement
5. **Music Selection** → Simulated matching
6. **Frame Styling** → DALL-E 3 with brand colors
7. **Video Transitions** → Simulated video generation
8. **Timeline Assembly** → Complete video metadata

## 📈 Future Deployment

When Azure AI Foundry CLI becomes available:

1. **Install the extension**: `az extension add --name ai-foundry`
2. **Follow the original deployment guide**
3. **Update environment variables** with real endpoints
4. **The app will automatically use real models**

## 🎯 Benefits of Simulated Mode

### ✅ **Immediate Testing**
- No deployment required
- Full pipeline demonstration
- Professional UI/UX

### ✅ **Cost Effective**
- Only pay for DALL-E 3 and GPT-4
- No expensive model deployment
- Perfect for MVP validation

### ✅ **Easy Scaling**
- Add real models when needed
- Gradual deployment approach
- No infrastructure complexity

The simulated mode provides a complete video generation experience while you validate your MVP! 