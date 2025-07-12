# Lean MVP Video Generation Deployment

## 🎯 Cost-Effective MVP Approach

For a lean MVP, we've optimized the video generation pipeline to use more cost-effective alternatives:

### ✅ **Lean MVP Models (Deploy These)**

#### 1. Stable Video Diffusion - Core Video Generation
```bash
# Deploy only the essential video generation model
az ai-foundry model deploy \
  --name "stable-video-diffusion" \
  --model "stability-ai/stable-video-diffusion-img2vid-xt" \
  --endpoint-name "video-generation-endpoint" \
  --resource-group "your-resource-group"
```

#### 2. GPT-4 Vision - Caption Enhancement (Optional)
```bash
# Deploy for better caption quality (optional for MVP)
az ai-foundry model deploy \
  --name "gpt4-vision-caption" \
  --model "gpt-4-vision-preview" \
  --endpoint-name "video-generation-endpoint" \
  --resource-group "your-resource-group"
```

### ❌ **Costly Models (Using Alternatives)**

- **SDXL** → **DALL-E 3** (already in your app, much cheaper)
- **VideoMAE** → **DALL-E 3** frame variations
- **CLIP** → **Simulated ranking**
- **BLIP** → **GPT-4** text generation

## 🚀 Minimal Deployment

### Option 1: Single Model Deployment (Recommended)
```bash
# Create minimal resources
az group create --name "video-mvp-rg" --location "eastus"
az ai-foundry workspace create --name "video-mvp-workspace" --resource-group "video-mvp-rg" --location "eastus"
az ai-foundry endpoint create --name "video-mvp-endpoint" --workspace "video-mvp-workspace" --resource-group "video-mvp-rg"

# Deploy only Stable Video Diffusion
az ai-foundry model deploy \
  --name "stable-video-diffusion" \
  --model "stability-ai/stable-video-diffusion-img2vid-xt" \
  --endpoint-name "video-mvp-endpoint" \
  --resource-group "video-mvp-rg"
```

### Option 2: No Deployment (Simulated Mode)
The app works perfectly with simulated responses. Just set these environment variables:

```env
# For simulated mode (no deployment needed)
AZURE_AI_FOUNDRY_ENDPOINT=https://placeholder.aifoundry.azure.com
AZURE_AI_FOUNDRY_API_KEY=placeholder-key
```

## 💰 Cost Comparison

### Lean MVP (Recommended)
- **Stable Video Diffusion**: ~$0.10-0.20 per video
- **DALL-E 3** (frame styling): ~$0.04-0.08 per image
- **GPT-4** (caption generation): ~$0.01-0.02 per caption
- **Total per video reel**: ~$0.15-0.30

### Full Deployment (Expensive)
- **SDXL**: ~$0.05-0.10 per image
- **Stable Video Diffusion**: ~$0.10-0.20 per video
- **GPT-4 Vision**: ~$0.01-0.03 per caption
- **Total per video reel**: ~$0.16-0.33

### Simulated Mode (Free)
- **All responses simulated**: $0.00
- **Perfect for testing and demos**

## 🎯 Updated Video Pipeline

### Lean MVP Pipeline:
1. **📸 Extract Frames** → DALL-E 3 (already available)
2. **🏆 Rank Frames** → Simulated ranking
3. **✍️ Draft Caption** → GPT-4 (already available)
4. **✨ Polish Caption** → GPT-4 Vision (optional)
5. **🎵 Choose Music** → Simulated selection
6. **🎨 Stylize Frames** → DALL-E 3 (cost-effective)
7. **🎬 Generate Transitions** → Stable Video Diffusion
8. **📋 Compose Timeline** → Final assembly

## 🚀 Quick Start Commands

### Minimal Deployment:
```bash
# 1. Create resources
az group create --name "video-mvp-rg" --location "eastus"
az ai-foundry workspace create --name "video-workspace" --resource-group "video-mvp-rg" --location "eastus"
az ai-foundry endpoint create --name "video-endpoint" --workspace "video-workspace" --resource-group "video-mvp-rg"

# 2. Deploy only Stable Video Diffusion
az ai-foundry model deploy \
  --name "video-diffusion" \
  --model "stability-ai/stable-video-diffusion-img2vid-xt" \
  --endpoint-name "video-endpoint" \
  --resource-group "video-mvp-rg"

# 3. Get endpoint details
az ai-foundry endpoint show \
  --name "video-endpoint" \
  --workspace "video-workspace" \
  --resource-group "video-mvp-rg"
```

### Environment Variables:
```env
# Minimal setup
AZURE_AI_FOUNDRY_ENDPOINT=https://your-endpoint.aifoundry.azure.com
AZURE_AI_FOUNDRY_API_KEY=your-api-key

# Optional model-specific endpoints
SVD_ENDPOINT=https://your-endpoint.aifoundry.azure.com/video-diffusion
GPT4V_ENDPOINT=https://your-endpoint.aifoundry.azure.com/gpt4-vision-caption
```

## 🎯 MVP Benefits

### ✅ **Cost Savings**
- **90% cost reduction** compared to full SDXL deployment
- Uses existing DALL-E 3 infrastructure
- Simulated responses for expensive operations

### ✅ **Faster Deployment**
- Single model deployment
- Minimal infrastructure setup
- Works immediately with simulated responses

### ✅ **Same User Experience**
- Full video generation pipeline
- Brand customization
- Professional output quality

## 🔧 Testing the MVP

```bash
# Test with simulated responses (no deployment needed)
streamlit run postcraft_azure_ready.py

# Test with real video generation (after deployment)
# The app will automatically use real endpoints when available
```

## 📈 Scaling Up Later

When you're ready to scale:

1. **Add GPT-4 Vision** for better captions
2. **Deploy SDXL** for advanced styling (if budget allows)
3. **Add more video models** for variety

The app is designed to gracefully fall back to simulated responses, so you can deploy incrementally! 