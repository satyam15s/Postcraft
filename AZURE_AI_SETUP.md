# 🚀 Azure AI Foundry Setup Guide for PostCraft Pro

## Quick Fix for Bicep Error

The error in line 5 of the Bicep template was due to incorrect resource types. Here's the corrected approach:

## Method 1: Azure Portal (Recommended)

### Step 1: Create AI Foundry Workspace
1. Go to [Azure Portal](https://portal.azure.com)
2. Search for "AI Foundry"
3. Click "Create"
4. Fill in the details:
   - **Workspace name**: `postcraft-workspace`
   - **Resource group**: Create new `postcraft-ai-foundry`
   - **Region**: `East US`
   - **Subscription**: Your subscription

### Step 2: Deploy Models
1. Go to your AI Foundry workspace
2. Click "Models" → "Browse models"
3. Search and deploy these models:

#### Required Models:
```
1. microsoft/videomae-base
2. openai/clip-vit-large-patch14
3. Salesforce/blip2-flan-t5-xxl
4. gpt-4o
5. stabilityai/stable-diffusion-xl-base-1.0
6. stabilityai/stable-video-diffusion-img2vid-xt
```

### Step 3: Get Endpoints
1. Go to "Endpoints" in your workspace
2. Copy the endpoint URLs and keys
3. Update your `.env` file

## Method 2: Azure CLI (Alternative)

### Step 1: Install Prerequisites
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Install AI Foundry extension
az extension add --name ai-foundry
```

### Step 2: Create Resource Group
```bash
az group create --name "postcraft-ai-foundry" --location "eastus"
```

### Step 3: Create Workspace
```bash
az ml workspace create \
  --name "postcraft-workspace" \
  --resource-group "postcraft-ai-foundry" \
  --location "eastus"
```

### Step 4: Deploy Models
```bash
# Deploy each model individually
az ai-foundry model deploy \
  --workspace-name "postcraft-workspace" \
  --resource-group "postcraft-ai-foundry" \
  --name "videomae-base" \
  --model "microsoft/videomae-base" \
  --sku "Standard_NC24rs_v3"

az ai-foundry model deploy \
  --workspace-name "postcraft-workspace" \
  --resource-group "postcraft-ai-foundry" \
  --name "clip-vit-l14" \
  --model "openai/clip-vit-large-patch14" \
  --sku "Standard_NC24rs_v3"

az ai-foundry model deploy \
  --workspace-name "postcraft-workspace" \
  --resource-group "postcraft-ai-foundry" \
  --name "blip-2-flan-t5-xxl" \
  --model "Salesforce/blip2-flan-t5-xxl" \
  --sku "Standard_NC24rs_v3"

az ai-foundry model deploy \
  --workspace-name "postcraft-workspace" \
  --resource-group "postcraft-ai-foundry" \
  --name "gpt-4o-vision" \
  --model "gpt-4o" \
  --sku "Standard_NC24rs_v3"

az ai-foundry model deploy \
  --workspace-name "postcraft-workspace" \
  --resource-group "postcraft-ai-foundry" \
  --name "sdxl-style-transfer-lora" \
  --model "stabilityai/stable-diffusion-xl-base-1.0" \
  --sku "Standard_NC24rs_v3"

az ai-foundry model deploy \
  --workspace-name "postcraft-workspace" \
  --resource-group "postcraft-ai-foundry" \
  --name "stable-video-diffusion" \
  --model "stabilityai/stable-video-diffusion-img2vid-xt" \
  --sku "Standard_NC24rs_v3"
```

## Method 3: PowerShell Script

Use the provided `deploy-models.ps1` script:

```powershell
# Run the PowerShell script
.\deploy-models.ps1 -WorkspaceName "postcraft-workspace" -ResourceGroup "postcraft-ai-foundry"
```

## Environment Configuration

### Update your `.env` file:
```env
# Azure AI Foundry Video Generation
AZURE_AI_FOUNDRY_ENDPOINT=https://your-workspace.eastus.ai-foundry.azure.com/
AZURE_AI_FOUNDRY_API_KEY=your_actual_api_key_here

# Model-specific endpoints (optional)
VIDEOMAE_ENDPOINT=https://videomae-base.eastus.inference.ml.azure.com/
CLIP_ENDPOINT=https://clip-vit-l14.eastus.inference.ml.azure.com/
BLIP_ENDPOINT=https://blip-2-flan-t5-xxl.eastus.inference.ml.azure.com/
GPT4V_ENDPOINT=https://gpt-4o-vision.eastus.inference.ml.azure.com/
SDXL_ENDPOINT=https://sdxl-style-transfer-lora.eastus.inference.ml.azure.com/
SVD_ENDPOINT=https://stable-video-diffusion.eastus.inference.ml.azure.com/
```

## Troubleshooting

### Common Issues:

1. **"Model not available"**
   - Some models may not be available in your region
   - Try different regions: `westus2`, `northeurope`, `southeastasia`

2. **"Quota exceeded"**
   - Check your Azure subscription quotas
   - Request quota increase from Azure support

3. **"Authentication failed"**
   - Verify your Azure credentials
   - Check if you have proper permissions

4. **"Deployment failed"**
   - Ensure you have sufficient compute quota
   - Try with smaller SKU: `Standard_NC6s_v3`

### Fallback Mode:
The PostCraft Pro app will work without Azure AI Foundry using simulated responses. Video generation will be limited but functional.

## Cost Estimation

- **GPU SKUs**: `Standard_NC24rs_v3` (~$3.60/hour)
- **Monthly estimate**: $500-1000 for full pipeline
- **Cost optimization**: Use smaller SKUs for testing

## Next Steps

1. **Deploy the workspace** using any method above
2. **Deploy the models** one by one
3. **Get endpoint URLs and keys**
4. **Update your `.env` file**
5. **Test video generation** in PostCraft Pro
6. **Monitor costs** in Azure Portal

## Support

If you encounter issues:
- Check Azure AI Foundry documentation
- Review Azure quotas and limits
- Contact Azure support for quota increases
- Use the fallback mode for testing

---

**Note**: Azure AI Foundry is currently in preview. Some features may change or have limited availability. 