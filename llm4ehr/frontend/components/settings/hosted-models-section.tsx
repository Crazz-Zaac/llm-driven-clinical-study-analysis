"use client"

import { useState } from "react"
import { useAppStore } from "@/lib/store"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Eye,
  EyeOff,
  Check,
  ExternalLink,
  Info,
  Shield,
  AlertCircle,
  Database
} from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

const POPULAR_MODELS = [
  { id: "meta-llama/Llama-3.1-8B-Instruct", name: "Llama 3.1 8B", provider: "Meta" },
  { id: "mistralai/Mistral-7B-Instruct-v0.3", name: "Mistral 7B", provider: "Mistral AI" },
  { id: "microsoft/Phi-3-mini-4k-instruct", name: "Phi-3 Mini", provider: "Microsoft" },
  { id: "google/gemma-2-9b-it", name: "Gemma 2 9B", provider: "Google" },
  { id: "Qwen/Qwen2-7B-Instruct", name: "Qwen2 7B", provider: "Alibaba" },
]

export function HostedModelsSection() {
  const {
    hostedModelName,
    hostedApiKey,
    useHostedModels,
    selectedEmbeddingModel,
    localModels,
    setHostedModelName,
    setHostedApiKey,
    setSelectedEmbeddingModel,
    indexingState,
    setNeedsReindex,
  } = useAppStore()

  const [showApiKey, setShowApiKey] = useState(false)
  const [apiKeySaved, setApiKeySaved] = useState(false)

  const embeddingModels = localModels.filter(m => m.type === "embedding" && m.status === "downloaded")

  const handleApiKeyChange = (value: string) => {
    setHostedApiKey(value)
    setApiKeySaved(false)
  }

  const handleSaveApiKey = () => {
    if (hostedApiKey) {
      setApiKeySaved(true)
      // The key is automatically persisted via zustand persist
    }
  }

  const handleEmbeddingModelChange = (modelId: string) => {
    setSelectedEmbeddingModel(modelId)
    // Trigger reindex if there are context sources and the model changed
    if (indexingState.lastEmbeddingModel && indexingState.lastEmbeddingModel !== modelId) {
      setNeedsReindex(true)
    }
  }

  const maskApiKey = (key: string) => {
    if (!key) return ""
    if (key.length <= 8) return "*".repeat(key.length)
    return key.slice(0, 4) + "*".repeat(key.length - 8) + key.slice(-4)
  }

  if (!useHostedModels) {
    return null
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Hosted Models</CardTitle>
        <CardDescription>
          Connect to hosted model providers like Hugging Face, OpenAI, or Anthropic.
          Your API key is stored securely in your browser and never sent to our servers.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* API Key Section */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="api-key" className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              API Key
            </Label>
            {apiKeySaved && hostedApiKey && (
              <Badge variant="default" className="gap-1">
                <Check className="h-3 w-3" />
                Saved
              </Badge>
            )}
          </div>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Input
                id="api-key"
                type={showApiKey ? "text" : "password"}
                placeholder="Enter your API key (e.g., hf_...)"
                value={hostedApiKey}
                onChange={(e) => handleApiKeyChange(e.target.value)}
                className="pr-10 font-mono"
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
                onClick={() => setShowApiKey(!showApiKey)}
              >
                {showApiKey ? (
                  <EyeOff className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <Eye className="h-4 w-4 text-muted-foreground" />
                )}
                <span className="sr-only">{showApiKey ? "Hide" : "Show"} API key</span>
              </Button>
            </div>
            <Button onClick={handleSaveApiKey} disabled={!hostedApiKey}>
              Save Key
            </Button>
          </div>
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <Shield className="h-3 w-3" />
              Stored locally in your browser - encrypted at rest
            </span>
            <a
              href="https://huggingface.co/settings/tokens"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-primary hover:underline"
            >
              Get Hugging Face token <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>

        {/* Model Name */}
        <div className="space-y-3">
          <Label htmlFor="model-name">Chat Model</Label>
          <Input
            id="model-name"
            placeholder="e.g., meta-llama/Llama-3.1-8B-Instruct"
            value={hostedModelName}
            onChange={(e) => setHostedModelName(e.target.value)}
          />
          <p className="text-xs text-muted-foreground">
            Enter the full model identifier from your provider (e.g., Hugging Face Hub path)
          </p>
        </div>

        {/* Embedding Model Selection */}
        <div className="space-y-3">
          <Label htmlFor="embedding-model" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Embedding Model
          </Label>
          {embeddingModels.length > 0 ? (
            <>
              <Select
                value={selectedEmbeddingModel || ""}
                onValueChange={handleEmbeddingModelChange}
              >
                <SelectTrigger id="embedding-model">
                  <SelectValue placeholder="Select an embedding model" />
                </SelectTrigger>
                <SelectContent>
                  {embeddingModels.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                The embedding model is used to create vector representations of documents for semantic search and retrieval.
              </p>
            </>
          ) : (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                No local embedding models available. Please download an embedding model in the Models tab to enable context-aware search.
              </AlertDescription>
            </Alert>
          )}
        </div>

        {/* Popular Models */}
        <div className="space-y-3">
          <Label>Popular Models</Label>
          <div className="grid gap-2 sm:grid-cols-2">
            {POPULAR_MODELS.map((model) => {
              const isSelected = hostedModelName === model.id
              return (
                <button
                  key={model.id}
                  onClick={() => setHostedModelName(isSelected ? "" : model.id)}
                  className={`flex items-center justify-between p-3 rounded-lg border transition-colors text-left ${isSelected
                    ? "border-primary bg-primary/5"
                    : "border-border hover:bg-muted/50"
                    }`}
                >
                  <div>
                    <span className="font-medium text-sm">{model.name}</span>
                    <p className="text-xs text-muted-foreground">{model.provider}</p>
                  </div>
                  {isSelected && (
                    <Check className="h-4 w-4 text-primary" />
                  )}
                </button>
              )
            })}
          </div>
        </div>

        {/* Warnings */}
        {!hostedApiKey && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              An API key is required to use hosted models. Most providers require authentication
              for API access.
            </AlertDescription>
          </Alert>
        )}

        <div className="flex items-start gap-2 rounded-lg bg-muted/50 p-3 text-sm">
          <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
          <div className="text-muted-foreground space-y-1">
            <p>
              <strong>Privacy Note:</strong> When using hosted models, your prompts and data
              are sent to the model provider&apos;s servers for processing.
            </p>
            <p>
              For maximum privacy with sensitive data, consider using local models instead.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
