"use client"

import { useEffect, useState } from "react"
import { useAppStore, type HostedProvider } from "@/lib/store"
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
} from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

type HostedModelOption = {
  id: string
  name: string
  provider: string
}

type HostedProviderOption = {
  id: HostedProvider
  label: string
  keyLink: { label: string; href: string }
  placeholder: string
  helperText: string
  models: HostedModelOption[]
}

const HOSTED_PROVIDERS: HostedProviderOption[] = [
  {
    id: "openai",
    label: "OpenAI",
    keyLink: {
      label: "Get OpenAI API key",
      href: "https://platform.openai.com/api-keys",
    },
    placeholder: "e.g., gpt-4o-mini",
    helperText: "Enter the OpenAI chat model ID.",
    models: [
      { id: "gpt-4o-mini", name: "GPT-4o mini", provider: "OpenAI" },
      { id: "gpt-4o", name: "GPT-4o", provider: "OpenAI" },
    ],
  },
  {
    id: "mistral",
    label: "Mistral AI",
    keyLink: {
      label: "Get Mistral API key",
      href: "https://console.mistral.ai/api-keys",
    },
    placeholder: "e.g., mistral-small-latest",
    helperText: "Enter the Mistral chat model ID.",
    models: [
      { id: "mistral-small-latest", name: "Mistral Small", provider: "Mistral AI" },
      { id: "mistral-large-latest", name: "Mistral Large", provider: "Mistral AI" },
      { id: "ministral-8b-latest", name: "Ministral 8B", provider: "Mistral AI" },
    ],
  },
  {
    id: "huggingface",
    label: "Hugging Face",
    keyLink: {
      label: "Get Hugging Face token",
      href: "https://huggingface.co/settings/tokens",
    },
    placeholder: "e.g., meta-llama/Llama-3.1-8B-Instruct",
    helperText: "Enter the full Hugging Face Hub model path.",
    models: [
      { id: "meta-llama/Llama-3.1-8B-Instruct", name: "Llama 3.1 8B", provider: "Meta" },
      { id: "mistralai/Mistral-7B-Instruct-v0.3", name: "Mistral 7B", provider: "Mistral AI" },
      { id: "microsoft/Phi-3-mini-4k-instruct", name: "Phi-3 Mini", provider: "Microsoft" },
      { id: "google/gemma-2-9b-it", name: "Gemma 2 9B", provider: "Google" },
      { id: "Qwen/Qwen2-7B-Instruct", name: "Qwen2 7B", provider: "Alibaba" },
    ],
  },
]

const providerConfigById = Object.fromEntries(
  HOSTED_PROVIDERS.map((provider) => [provider.id, provider]),
) as Record<HostedProvider, HostedProviderOption>

export function HostedModelsSection() {
  const {
    hostedModelName,
    hostedApiKey,
    hostedProvider,
    useHostedModels,
    setHostedModelName,
    setHostedApiKey,
    setHostedProvider,
  } = useAppStore()

  const [showApiKey, setShowApiKey] = useState(false)
  const selectedProvider = providerConfigById[hostedProvider] ? hostedProvider : "openai"
  const [draftProvider, setDraftProvider] = useState<HostedProvider>(selectedProvider)
  const [draftModelName, setDraftModelName] = useState(hostedModelName)
  const [draftApiKey, setDraftApiKey] = useState(hostedApiKey)
  const providerConfig = providerConfigById[draftProvider]
  const trimmedDraftApiKey = draftApiKey.trim()
  const trimmedDraftModelName = draftModelName.trim()
  const canSaveSettings = Boolean(trimmedDraftApiKey && trimmedDraftModelName)
  const settingsAreSaved =
    canSaveSettings &&
    draftProvider === selectedProvider &&
    trimmedDraftModelName === hostedModelName.trim() &&
    trimmedDraftApiKey === hostedApiKey.trim()

  useEffect(() => {
    setDraftProvider(selectedProvider)
    setDraftModelName(hostedModelName)
    setDraftApiKey(hostedApiKey)
  }, [hostedApiKey, hostedModelName, selectedProvider])

  const handleApiKeyChange = (value: string) => {
    setDraftApiKey(value)
  }

  const handleSaveSettings = () => {
    if (!canSaveSettings) return

    setHostedProvider(draftProvider)
    setHostedModelName(trimmedDraftModelName)
    setHostedApiKey(trimmedDraftApiKey)
  }

  const handleProviderChange = (provider: HostedProvider) => {
    if (provider === draftProvider) {
      return
    }

    setDraftProvider(provider)
    setDraftApiKey("")
    setDraftModelName(providerConfigById[provider].models[0]?.id ?? "")
  }

  if (!useHostedModels) {
    return null
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Hosted Models</CardTitle>
        <CardDescription>
          Connect to hosted model providers like OpenAI, Mistral AI, or Hugging Face.
          Your API key is stored in this browser and sent only with hosted generation requests.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-3">
          <Label htmlFor="provider">Provider</Label>
          <Select value={draftProvider} onValueChange={(value) => handleProviderChange(value as HostedProvider)}>
            <SelectTrigger id="provider">
              <SelectValue placeholder="Select provider" />
            </SelectTrigger>
            <SelectContent>
              {HOSTED_PROVIDERS.map((provider) => (
                <SelectItem key={provider.id} value={provider.id}>
                  {provider.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* API Key Section */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="api-key" className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              API Key
            </Label>
            {settingsAreSaved && (
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
                placeholder={`Enter your ${providerConfig.label} API key`}
                value={draftApiKey}
                onChange={(e) => handleApiKeyChange(e.target.value)}
                required
                aria-required="true"
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
            <Button onClick={handleSaveSettings} disabled={!canSaveSettings}>
              Save Settings
            </Button>
          </div>
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <Shield className="h-3 w-3" />
              Stored locally in this browser
            </span>
            <a
              href={providerConfig.keyLink.href}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-primary hover:underline"
            >
              {providerConfig.keyLink.label} <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>

        {/* Model Name */}
        <div className="space-y-3">
          <Label htmlFor="model-name">Chat Model</Label>
          <Input
            id="model-name"
            placeholder={providerConfig.placeholder}
            value={draftModelName}
            onChange={(e) => setDraftModelName(e.target.value)}
          />
          <p className="text-xs text-muted-foreground">
            {providerConfig.helperText}
          </p>
        </div>

        {/* Popular Models */}
        <div className="space-y-3">
          <Label>Suggested Chat Models</Label>
          <div className="grid gap-2 sm:grid-cols-2">
            {providerConfig.models.map((model) => {
              const isSelected = draftModelName === model.id
              return (
                <button
                  key={model.id}
                  onClick={() => setDraftModelName(isSelected ? "" : model.id)}
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
        {!trimmedDraftApiKey && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              An API key is required before hosted model settings can be saved.
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
