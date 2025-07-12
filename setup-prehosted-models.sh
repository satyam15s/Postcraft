#!/bin/bash

# Setup script for pre-hosted Azure AI Foundry models
echo "🚀 Setting up pre-hosted Azure AI Foundry models for PostCraft Pro..."

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI not found. Installing..."
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
fi

# Login to Azure
echo "🔐 Logging into Azure..."
az login

# Get workspace details
echo "📋 Getting workspace details..."
WORKSPACE_NAME="satyamss219-7024_ai"
RESOURCE_GROUP="rg-satyamss219-5182_ai"

# Get workspace endpoint
WORKSPACE_ENDPOINT=$(az ml workspace show \
  --name $WORKSPACE_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "properties.discoveryUrl" \
  --output tsv)

echo "✅ Workspace endpoint: $WORKSPACE_ENDPOINT"

# Get API key
echo "🔑 Getting API key..."
API_KEY=$(az ml workspace get-credentials \
  --name $WORKSPACE_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "primaryKey" \
  --output tsv)

echo "✅ API key retrieved"

# Create .env file with pre-hosted model endpoints
echo "📝 Creating .env file with pre-hosted model endpoints..."
cat > .env << EOF
# Azure OpenAI Text Service
AZURE_OPENAI_TEXT_ENDPOINT=https://ai-satyamss2197024ai511680019271.openai.azure.com/
AZURE_OPENAI_TEXT_KEY=3K0lCByFz8rtuLuDRuSK6mAGQk0ySRFRKMxQ1RQjp6t9OrxyB5mkJQQJ99BEACHYHv6XJ3w3AAAAACOGyF0C

# Azure OpenAI Image Service (DALL-E)
AZURE_OPENAI_IMAGE_ENDPOINT=https://satya-mcr9l71d-swedencentral.openai.azure.com/
AZURE_OPENAI_IMAGE_KEY=2weolebzkqDIUSaEX7l5EE32sFlGmaOr0DC4BDzmINRCpaZvTI2hJQQJ99BGACfhMk5XJ3w3AAAAACOGF45A

# Azure AI Foundry Pre-hosted Models (Pay-per-use)
AZURE_AI_FOUNDRY_ENDPOINT=$WORKSPACE_ENDPOINT
AZURE_AI_FOUNDRY_API_KEY=$API_KEY

# Pre-hosted model endpoints (pay-per-use)
VIDEOMAE_ENDPOINT=https://videomae-base.eastus2.inference.ml.azure.com/
CLIP_ENDPOINT=https://clip-vit-l14.eastus2.inference.ml.azure.com/
BLIP_ENDPOINT=https://blip-2-flan-t5-xxl.eastus2.inference.ml.azure.com/
GPT4V_ENDPOINT=https://gpt-4o-vision.eastus2.inference.ml.azure.com/
SDXL_ENDPOINT=https://sdxl-style-transfer-lora.eastus2.inference.ml.azure.com/
SVD_ENDPOINT=https://stable-video-diffusion.eastus2.inference.ml.azure.com/

# Reddit User Agent (optional)
REDDIT_USER_AGENT=PostCraftPro/1.0
EOF

echo "✅ .env file created with pre-hosted model endpoints"
echo ""
echo "🎉 Setup complete! Your PostCraft Pro app is now configured to use:"
echo "   • Pre-hosted VideoMAE model for frame extraction"
echo "   • Pre-hosted CLIP model for frame ranking"
echo "   • Pre-hosted BLIP model for caption generation"
echo "   • Pre-hosted GPT-4 Vision for caption polishing"
echo "   • Pre-hosted SDXL for style transfer"
echo "   • Pre-hosted Stable Video Diffusion for transitions"
echo ""
echo "💰 Pay-per-use pricing:"
echo "   • VideoMAE: ~$0.01 per image"
echo "   • CLIP: ~$0.005 per image"
echo "   • BLIP: ~$0.01 per caption"
echo "   • GPT-4 Vision: ~$0.03 per request"
echo "   • SDXL: ~$0.02 per image"
echo "   • Stable Video Diffusion: ~$0.05 per video"
echo ""
echo "🚀 Run your app with: streamlit run postcraft_azure_ready.py" 