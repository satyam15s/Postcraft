#!/bin/bash

# Azure AI Foundry Setup Script for PostCraft Pro
echo "🚀 Setting up Azure AI Foundry for PostCraft Pro..."

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI not found. Installing..."
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
fi

# Login to Azure
echo "🔐 Logging into Azure..."
az login

# Install AI Foundry extension
echo "📦 Installing Azure AI Foundry extension..."
az extension add --name ai-foundry

# Create resource group
echo "🏗️ Creating resource group..."
az group create --name "postcraft-ai-foundry" --location "eastus"

# Deploy Bicep template
echo "🚀 Deploying AI Foundry resources..."
az deployment group create \
  --resource-group "postcraft-ai-foundry" \
  --template-file "azure-ai-foundry.bicep" \
  --parameters workspaceName="postcraft-workspace"

# Get workspace details
echo "📋 Getting workspace details..."
WORKSPACE_NAME=$(az deployment group show \
  --resource-group "postcraft-ai-foundry" \
  --name "azure-ai-foundry" \
  --query "properties.outputs.workspaceName.value" \
  --output tsv)

echo "✅ Workspace created: $WORKSPACE_NAME"

# Get endpoint keys
echo "🔑 Getting endpoint keys..."
az ml online-endpoint show-keys \
  --name "videomae-base" \
  --resource-group "postcraft-ai-foundry" \
  --query "primaryKey" \
  --output tsv > videomae_key.txt

az ml online-endpoint show-keys \
  --name "clip-vit-l14" \
  --resource-group "postcraft-ai-foundry" \
  --query "primaryKey" \
  --output tsv > clip_key.txt

echo "✅ Setup complete! Check the generated key files."
echo "📝 Update your .env file with the endpoints and keys." 