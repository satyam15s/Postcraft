# PowerShell script to deploy AI models to Azure AI Foundry
param(
    [string]$WorkspaceName = "postcraft-workspace",
    [string]$ResourceGroup = "postcraft-ai-foundry",
    [string]$Location = "eastus"
)

Write-Host "🚀 Deploying AI models to Azure AI Foundry..." -ForegroundColor Green

# Login to Azure
Write-Host "🔐 Logging into Azure..." -ForegroundColor Yellow
az login

# Install AI Foundry extension if not already installed
Write-Host "📦 Installing Azure AI Foundry extension..." -ForegroundColor Yellow
az extension add --name ai-foundry

# Deploy VideoMAE model
Write-Host "📸 Deploying VideoMAE model..." -ForegroundColor Yellow
az ai-foundry model deploy `
    --workspace-name $WorkspaceName `
    --resource-group $ResourceGroup `
    --name "videomae-base" `
    --model "microsoft/videomae-base" `
    --sku "Standard_NC24rs_v3"

# Deploy CLIP model
Write-Host "🏆 Deploying CLIP-ViT-L/14 model..." -ForegroundColor Yellow
az ai-foundry model deploy `
    --workspace-name $WorkspaceName `
    --resource-group $ResourceGroup `
    --name "clip-vit-l14" `
    --model "openai/clip-vit-large-patch14" `
    --sku "Standard_NC24rs_v3"

# Deploy BLIP model
Write-Host "✍️ Deploying BLIP-2-Flan-T5-XXL model..." -ForegroundColor Yellow
az ai-foundry model deploy `
    --workspace-name $WorkspaceName `
    --resource-group $ResourceGroup `
    --name "blip-2-flan-t5-xxl" `
    --model "Salesforce/blip2-flan-t5-xxl" `
    --sku "Standard_NC24rs_v3"

# Deploy GPT-4 Vision model
Write-Host "✨ Deploying GPT-4 Vision model..." -ForegroundColor Yellow
az ai-foundry model deploy `
    --workspace-name $WorkspaceName `
    --resource-group $ResourceGroup `
    --name "gpt-4o-vision" `
    --model "gpt-4o" `
    --sku "Standard_NC24rs_v3"

# Deploy SDXL model
Write-Host "🎨 Deploying SDXL Style Transfer model..." -ForegroundColor Yellow
az ai-foundry model deploy `
    --workspace-name $WorkspaceName `
    --resource-group $ResourceGroup `
    --name "sdxl-style-transfer-lora" `
    --model "stabilityai/stable-diffusion-xl-base-1.0" `
    --sku "Standard_NC24rs_v3"

# Deploy Stable Video Diffusion model
Write-Host "🎬 Deploying Stable Video Diffusion model..." -ForegroundColor Yellow
az ai-foundry model deploy `
    --workspace-name $WorkspaceName `
    --resource-group $ResourceGroup `
    --name "stable-video-diffusion" `
    --model "stabilityai/stable-video-diffusion-img2vid-xt" `
    --sku "Standard_NC24rs_v3"

Write-Host "✅ All models deployed successfully!" -ForegroundColor Green

# Get endpoint information
Write-Host "📋 Getting endpoint information..." -ForegroundColor Yellow
az ml online-endpoint list --workspace-name $WorkspaceName --resource-group $ResourceGroup

Write-Host "🎉 Setup complete! Update your .env file with the endpoint URLs and keys." -ForegroundColor Green 