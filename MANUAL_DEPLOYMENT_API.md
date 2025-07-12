# Manual Deployment & API Configuration Guide

## 🎯 Manual Deployment Setup

Since the Azure AI Foundry CLI isn't available, here's how to deploy models manually and configure the API endpoints.

## 📋 Prerequisites

1. **Azure Subscription** with AI Foundry access
2. **Azure CLI** installed and logged in
3. **Resource Group** created for your models

## 🚀 Step 1: Create Azure AI Foundry Resources

### Via Azure Portal:
1. Go to [Azure Portal](https://portal.azure.com)
2. Search for "AI Foundry"
3. Click "Create" → "AI Foundry"
4. Configure:
   - **Resource Group**: `video-generation-rg`
   - **Workspace Name**: `video-generation-workspace`
   - **Region**: `East US`
   - **Compute**: `Standard_DS3_v2` (or smaller for MVP)

### Via Azure CLI (if available):
```bash
# Create resource group
az group create --name "video-generation-rg" --location "eastus"

# Create AI Foundry workspace (if CLI extension becomes available)
az ai-foundry workspace create \
  --name "video-generation-workspace" \
  --resource-group "video-generation-rg" \
  --location "eastus"
```

## 🎯 Step 2: Deploy Models Manually

### Deploy Stable Video Diffusion:
1. Navigate to your AI Foundry workspace
2. Go to "Models" → "Deploy Model"
3. Select: `stability-ai/stable-video-diffusion-img2vid-xt`
4. Configure:
   - **Model Name**: `stable-video-diffusion`
   - **Endpoint Name**: `video-generation-endpoint`
   - **Compute**: `Standard_DS3_v2`

### Deploy GPT-4 Vision:
1. Go to "Models" → "Deploy Model"
2. Select: `gpt-4-vision-preview`
3. Configure:
   - **Model Name**: `gpt4-vision-caption`
   - **Endpoint Name**: `video-generation-endpoint`
   - **Compute**: `Standard_DS3_v2`

## 🔧 Step 3: Get API Endpoints

### From Azure Portal:
1. Go to your AI Foundry workspace
2. Navigate to "Endpoints"
3. Click on your endpoint
4. Copy the **Scoring URI** and **Primary Key**

### Example Endpoints:
```
# Your endpoints will look like:
SVD_ENDPOINT=https://video-generation-endpoint.aifoundry.azure.com/stable-video-diffusion
GPT4V_ENDPOINT=https://video-generation-endpoint.aifoundry.azure.com/gpt4-vision-caption
AZURE_AI_FOUNDRY_API_KEY=your-primary-key
```

## 📝 Step 4: Configure Environment Variables

Create or update your `.env` file:

```env
# Azure OpenAI (already working)
AZURE_OPENAI_TEXT_ENDPOINT=https://your-text-endpoint.openai.azure.com/
AZURE_OPENAI_TEXT_KEY=your-text-key
AZURE_OPENAI_IMAGE_ENDPOINT=https://your-image-endpoint.openai.azure.com/
AZURE_OPENAI_IMAGE_KEY=your-image-key

# Azure AI Foundry - Manual Deployment
AZURE_AI_FOUNDRY_ENDPOINT=https://video-generation-endpoint.aifoundry.azure.com
AZURE_AI_FOUNDRY_API_KEY=your-primary-key

# Model-specific endpoints
SVD_ENDPOINT=https://video-generation-endpoint.aifoundry.azure.com/stable-video-diffusion
GPT4V_ENDPOINT=https://video-generation-endpoint.aifoundry.azure.com/gpt4-vision-caption
```

## 🧪 Step 5: Test API Endpoints

### Test GPT-4 Vision:
```bash
curl -X POST "https://video-generation-endpoint.aifoundry.azure.com/gpt4-vision-caption/score" \
  -H "Authorization: Bearer your-primary-key" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Polish this caption for a technology brand with professional tone: Amazing tech insights!"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": "https://example.com/image.jpg"
              }
            }
          ]
        }
      ],
      "max_tokens": 200,
      "temperature": 0.7
    }
  }'
```

### Test Stable Video Diffusion:
```bash
curl -X POST "https://video-generation-endpoint.aifoundry.azure.com/stable-video-diffusion/score" \
  -H "Authorization: Bearer your-primary-key" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_url": "https://example.com/image.jpg",
      "num_frames": 16,
      "fps": 8,
      "motion_bucket_id": 127,
      "cond_aug": 0.02,
      "decoding_t": 7,
      "output_format": "mp4"
    }
  }'
```

## 🎯 Step 6: Update App Configuration

The app is already configured to work with manual deployment. Just ensure your environment variables are set correctly.

### API Response Format:
The app handles these response formats:

```json
// GPT-4 Vision Response
{
  "output": {
    "choices": [
      {
        "message": {
          "content": "Polished caption text"
        }
      }
    ]
  }
}

// Stable Video Diffusion Response
{
  "output": {
    "video_url": "https://generated-video-url.mp4"
  }
}
```

## 🔍 Step 7: Troubleshooting

### Common Issues:

1. **Authentication Failed**:
   - Verify API key is correct
   - Check endpoint URL format
   - Ensure key has proper permissions

2. **Model Not Found**:
   - Verify model is deployed
   - Check endpoint URL includes model name
   - Ensure model is in "Succeeded" state

3. **Timeout Errors**:
   - Increase timeout in app (already set to 60-180 seconds)
   - Check model deployment status
   - Verify compute resources

4. **API Format Errors**:
   - Check payload format matches Azure AI Foundry requirements
   - Verify content-type headers
   - Ensure JSON structure is correct

### Debug Commands:
```bash
# Check endpoint status
curl -H "Authorization: Bearer your-key" \
  "https://video-generation-endpoint.aifoundry.azure.com/stable-video-diffusion/score"

# Test with minimal payload
curl -X POST "https://video-generation-endpoint.aifoundry.azure.com/gpt4-vision-caption/score" \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]}}'
```

## 🚀 Step 8: Run the App

```bash
# Start the app with manual deployment
streamlit run postcraft_azure_ready.py
```

The app will:
- ✅ Use real GPT-4 Vision for caption polishing
- ✅ Use real Stable Video Diffusion for video generation
- ✅ Use DALL-E 3 for frame styling (cost-effective)
- ✅ Use GPT-4 for caption generation (already working)
- ✅ Fall back to simulated responses if APIs fail

## 💰 Cost Optimization

### Manual Deployment Costs:
- **Stable Video Diffusion**: ~$0.10-0.20 per video
- **GPT-4 Vision**: ~$0.01-0.03 per caption
- **DALL-E 3** (frame styling): ~$0.04-0.08 per image
- **GPT-4** (caption generation): ~$0.01-0.02 per caption

**Total per video reel**: ~$0.16-0.33

### Simulated Mode (for testing):
- **All responses simulated**: $0.00
- **Perfect for development and demos**

The app gracefully handles both real and simulated responses, so you can test with simulated mode and switch to real APIs when ready! 