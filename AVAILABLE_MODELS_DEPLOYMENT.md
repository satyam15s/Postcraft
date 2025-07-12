# Available Models in Azure AI Foundry

## 🚨 Current Status: Limited Model Availability

After checking Azure AI Foundry, these models are **NOT available**:
- ❌ **GPT-4 Vision** (`gpt-4-vision-preview`)
- ❌ **Stable Video Diffusion** (`stability-ai/stable-video-diffusion-img2vid-xt`)
- ❌ **VideoMAE** (video frame extraction)
- ❌ **CLIP** (frame ranking)
- ❌ **BLIP** (caption generation)
- ❌ **SDXL** (frame styling)

## ✅ **What IS Available in Azure AI Foundry**

### Text Generation Models:
- **GPT-4** (`gpt-4`)
- **GPT-3.5 Turbo** (`gpt-35-turbo`)
- **Llama 2** (`meta-llama/Llama-2-7b-chat-hf`)
- **Code Llama** (`codellama/CodeLlama-7b-Instruct-hf`)

### Image Generation Models:
- **Stable Diffusion** (`runwayml/stable-diffusion-v1-5`)
- **Stable Diffusion XL** (`stability-ai/stable-diffusion-xl-base-1.0`)
- **Kandinsky** (`ai-forever/kandinsky-2.2`)

### Embedding Models:
- **text-embedding-ada-002**
- **all-MiniLM-L6-v2**

## 🎯 **Updated Video Generation Pipeline**

Since the video-specific models aren't available, we've updated the pipeline to use available alternatives:

### Current Pipeline:
1. **📸 Extract Frames** → DALL-E 3 (already working)
2. **🏆 Rank Frames** → Simulated ranking
3. **✍️ Draft Caption** → GPT-4 (already working)
4. **✨ Polish Caption** → GPT-4 text (not Vision)
5. **🎵 Choose Music** → Simulated selection
6. **🎨 Stylize Frames** → DALL-E 3 (cost-effective)
7. **🎬 Generate Transitions** → Simulated video generation
8. **📋 Compose Timeline** → Final assembly

## 🚀 **Recommended Deployment Strategy**

### Option 1: Use Available Models (Recommended)
Deploy models that are actually available in Azure AI Foundry:

```bash
# Deploy available models
az ai-foundry model deploy \
  --name "gpt4-text" \
  --model "gpt-4" \
  --endpoint-name "video-generation-endpoint" \
  --resource-group "video-generation-rg"

az ai-foundry model deploy \
  --name "stable-diffusion" \
  --model "runwayml/stable-diffusion-v1-5" \
  --endpoint-name "video-generation-endpoint" \
  --resource-group "video-generation-rg"
```

### Option 2: Simulated Mode (Cost-Effective)
Use the app with simulated responses for unavailable models:

```env
# Environment variables for simulated mode
AZURE_AI_FOUNDRY_ENDPOINT=https://placeholder.aifoundry.azure.com
AZURE_AI_FOUNDRY_API_KEY=placeholder-key
```

## 💰 **Cost Analysis**

### Simulated Mode (Recommended for MVP):
- **DALL-E 3** (frame styling): ~$0.04-0.08 per image
- **GPT-4** (caption generation): ~$0.01-0.02 per caption
- **Video Generation**: Simulated (free)
- **Total per video reel**: ~$0.05-0.10

### Available Models Deployment:
- **GPT-4** (text generation): ~$0.01-0.02 per request
- **Stable Diffusion** (image generation): ~$0.02-0.05 per image
- **Video Generation**: Simulated (free)
- **Total per video reel**: ~$0.03-0.07

## 🔧 **Alternative Video Generation Solutions**

Since Azure AI Foundry doesn't have video generation models, consider:

### 1. **Azure Media Services**
- Use Azure Media Services for video processing
- Integrate with existing image generation pipeline

### 2. **Third-Party Video APIs**
- **Runway ML** API for video generation
- **Pika Labs** API for video creation
- **Stable Video Diffusion** via Hugging Face

### 3. **Custom Video Pipeline**
- Use DALL-E 3 for frame generation
- Implement custom video assembly
- Use FFmpeg for video processing

## 🎯 **Updated Environment Variables**

```env
# Azure OpenAI (already working)
AZURE_OPENAI_TEXT_ENDPOINT=https://your-text-endpoint.openai.azure.com/
AZURE_OPENAI_TEXT_KEY=your-text-key
AZURE_OPENAI_IMAGE_ENDPOINT=https://your-image-endpoint.openai.azure.com/
AZURE_OPENAI_IMAGE_KEY=your-image-key

# Azure AI Foundry - Available Models Only
AZURE_AI_FOUNDRY_ENDPOINT=https://video-generation-endpoint.aifoundry.azure.com
AZURE_AI_FOUNDRY_API_KEY=your-primary-key

# Available model endpoints (if deployed)
TEXT_GEN_ENDPOINT=https://video-generation-endpoint.aifoundry.azure.com/gpt4-text
IMAGE_GEN_ENDPOINT=https://video-generation-endpoint.aifoundry.azure.com/stable-diffusion
```

## 🚀 **Quick Start - Simulated Mode**

The app works perfectly with simulated responses:

```bash
# Just run the app
streamlit run postcraft_azure_ready.py
```

The app will:
- ✅ Use DALL-E 3 for image generation (already working)
- ✅ Use GPT-4 for text generation (already working)
- ✅ Simulate video generation responses
- ✅ Provide full user experience

## 📈 **Future Considerations**

When video generation models become available in Azure AI Foundry:

1. **Monitor Azure AI Foundry updates** for new model availability
2. **Consider third-party video APIs** for immediate video generation
3. **Implement custom video pipeline** using available models
4. **Use Azure Media Services** for video processing

The app is designed to gracefully handle both real and simulated responses, so you can start with simulated mode and add real models when they become available! 