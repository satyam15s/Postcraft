@description('The name of the AI Foundry workspace')
param workspaceName string = 'postcraft-workspace'

@description('The location for the AI Foundry workspace')
param location string = resourceGroup().location

// AI Foundry Workspace
resource aiFoundryWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-02-01' = {
  name: workspaceName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: workspaceName
    description: 'AI Foundry workspace for PostCraft Pro video generation'
  }
}

// Compute Instance for Model Deployment
resource computeInstance 'Microsoft.MachineLearningServices/computes@2024-02-01' = {
  name: 'postcraft-compute'
  location: location
  properties: {
    computeType: 'ComputeInstance'
    properties: {
      vmSize: 'Standard_NC24rs_v3'
      sshSettings: {
        sshPublicAccess: 'Disabled'
      }
    }
  }
}

// Model Registry
resource modelRegistry 'Microsoft.MachineLearningServices/registries@2024-02-01' = {
  name: 'postcraft-model-registry'
  location: location
  properties: {
    description: 'Model registry for PostCraft Pro video generation models'
  }
}

// Outputs
output workspaceName string = aiFoundryWorkspace.name
output workspaceId string = aiFoundryWorkspace.id
output computeName string = computeInstance.name
output registryName string = modelRegistry.name 