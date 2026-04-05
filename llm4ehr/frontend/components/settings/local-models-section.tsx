"use client"

import { useState } from "react"
import { useAppStore } from "@/lib/store"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  Download,
  Check,
  Loader2,
  MessageSquare,
  Eye,
  Trash2,
  Plus,
  Info
} from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

export function LocalModelsSection() {
  const {
    localModels,
    selectedChatModel,
    selectedVisionModel,
    useHostedModels,
    setSelectedChatModel,
    setSelectedVisionModel,
    addLocalModel,
    updateLocalModel,
    removeLocalModel,
  } = useAppStore()

  const [newModelName, setNewModelName] = useState("")
  const [newModelType, setNewModelType] = useState<"chat" | "embedding" | "vision">("chat")
  const [isDownloading, setIsDownloading] = useState<string | null>(null)

  const chatModels = localModels.filter(m => m.type === "chat")
  const visionModels = localModels.filter(m => m.type === "vision")

  const handleAddModel = () => {
    if (!newModelName.trim()) return

    const modelId = newModelName.toLowerCase().replace(/[^a-z0-9]/g, "-")

    // Check if model already exists
    if (localModels.some(m => m.id === modelId)) {
      return
    }

    addLocalModel({
      id: modelId,
      name: newModelName.trim(),
      size: "Calculating...",
      type: newModelType,
      status: "downloading",
      downloadProgress: 0,
    })

    setIsDownloading(modelId)
    setNewModelName("")

    // Simulate download progress
    let progress = 0
    const interval = setInterval(() => {
      progress += Math.random() * 10
      if (progress >= 100) {
        updateLocalModel(modelId, {
          status: "downloaded",
          downloadProgress: 100,
          size: `${(Math.random() * 5 + 1).toFixed(1)} GB`
        })
        setIsDownloading(null)
        clearInterval(interval)
      } else {
        updateLocalModel(modelId, { downloadProgress: Math.min(progress, 99) })
      }
    }, 300)
  }

  const handleDownload = (id: string) => {
    setIsDownloading(id)
    updateLocalModel(id, { status: "downloading", downloadProgress: 0 })

    let progress = 0
    const interval = setInterval(() => {
      progress += Math.random() * 15
      if (progress >= 100) {
        updateLocalModel(id, { status: "downloaded", downloadProgress: 100 })
        setIsDownloading(null)
        clearInterval(interval)
      } else {
        updateLocalModel(id, { downloadProgress: Math.min(progress, 99) })
      }
    }, 300)
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "chat": return <MessageSquare className="h-4 w-4" />
      case "embedding": return <Database className="h-4 w-4" />
      case "vision": return <Eye className="h-4 w-4" />
      default: return <MessageSquare className="h-4 w-4" />
    }
  }

  if (useHostedModels) {
    return null
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Local Models</CardTitle>
        <CardDescription>
          Download and manage models that run directly on your device. Enter a model name (e.g., llama-3.2-3b, mistral-7b)
          to download it. The system will validate and fetch the model automatically.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Add New Model */}
        <div className="space-y-3">
          <Label>Add New Model</Label>
          <div className="flex gap-2">
            <Input
              placeholder="Enter model name (e.g., llama-3.2-3b)"
              value={newModelName}
              onChange={(e) => setNewModelName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAddModel()}
              className="flex-1"
            />
            <Select value={newModelType} onValueChange={(v) => setNewModelType(v as typeof newModelType)}>
              <SelectTrigger className="w-[140px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="chat">Chat</SelectItem>
                <SelectItem value="embedding">Embedding</SelectItem>
                <SelectItem value="vision">Vision</SelectItem>
              </SelectContent>
            </Select>
            <Button onClick={handleAddModel} disabled={!newModelName.trim() || isDownloading !== null}>
              <Plus className="h-4 w-4 mr-1" />
              Add
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Enter a model identifier. The backend will download and validate the model from supported sources.
          </p>
        </div>

        {/* Chat Models */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4" />
              Chat Models
            </Label>
            {chatModels.filter(m => m.status === "downloaded").length > 0 && (
              <Select
                value={selectedChatModel || ""}
                onValueChange={setSelectedChatModel}
              >
                <SelectTrigger className="w-[200px]">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {chatModels.filter(m => m.status === "downloaded").map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
          <div className="space-y-2">
            {chatModels.length === 0 ? (
              <div className="text-sm text-muted-foreground p-4 border rounded-lg border-dashed text-center">
                No chat models added. Add a model above to get started.
              </div>
            ) : (
              chatModels.map((model) => (
                <ModelRow
                  key={model.id}
                  model={model}
                  isSelected={selectedChatModel === model.id}
                  onSelect={() => setSelectedChatModel(model.id)}
                  onDownload={() => handleDownload(model.id)}
                  onRemove={() => removeLocalModel(model.id)}
                  icon={getTypeIcon(model.type)}
                />
              ))
            )}
          </div>
        </div>

        {/* Vision Models */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="flex items-center gap-2">
              <Eye className="h-4 w-4" />
              Vision Models
            </Label>
            {visionModels.filter(m => m.status === "downloaded").length > 0 && (
              <Select
                value={selectedVisionModel || ""}
                onValueChange={setSelectedVisionModel}
              >
                <SelectTrigger className="w-[200px]">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {visionModels.filter(m => m.status === "downloaded").map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
          <div className="space-y-2">
            {visionModels.length === 0 ? (
              <div className="text-sm text-muted-foreground p-4 border rounded-lg border-dashed text-center">
                No vision models added. Add a model above to get started.
              </div>
            ) : (
              visionModels.map((model) => (
                <ModelRow
                  key={model.id}
                  model={model}
                  isSelected={selectedVisionModel === model.id}
                  onSelect={() => setSelectedVisionModel(model.id)}
                  onDownload={() => handleDownload(model.id)}
                  onRemove={() => removeLocalModel(model.id)}
                  icon={getTypeIcon(model.type)}
                />
              ))
            )}
          </div>
        </div>

        <div className="flex items-start gap-2 rounded-lg bg-muted/50 p-3 text-sm">
          <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
          <p className="text-muted-foreground">
            Models are downloaded and stored locally. Larger models require more disk space and compute power
            but generally provide better quality responses.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}

interface ModelRowProps {
  model: {
    id: string
    name: string
    size: string
    status: "available" | "downloading" | "downloaded"
    downloadProgress?: number
  }
  isSelected: boolean
  onSelect: () => void
  onDownload: () => void
  onRemove: () => void
  icon: React.ReactNode
}

function ModelRow({ model, isSelected, onSelect, onDownload, onRemove, icon }: ModelRowProps) {
  return (
    <div className={`flex items-center justify-between p-3 rounded-lg border transition-colors ${isSelected ? "border-primary bg-primary/5" : "border-border"
      }`}>
      <div className="flex items-center gap-3 min-w-0">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-muted">
          {icon}
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium truncate">{model.name}</span>
            <Badge variant="secondary" className="text-xs">{model.size}</Badge>
            {isSelected && <Badge variant="default" className="text-xs">Active</Badge>}
          </div>
          <p className="text-xs text-muted-foreground">{model.id}</p>
        </div>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {model.status === "available" && (
          <Button size="sm" variant="outline" onClick={onDownload}>
            <Download className="h-4 w-4 mr-1" />
            Download
          </Button>
        )}
        {model.status === "downloading" && (
          <div className="flex items-center gap-2">
            <Progress value={model.downloadProgress} className="w-20 h-2" />
            <span className="text-xs text-muted-foreground w-8">{Math.round(model.downloadProgress || 0)}%</span>
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          </div>
        )}
        {model.status === "downloaded" && (
          <>
            <Button
              size="sm"
              variant={isSelected ? "default" : "outline"}
              onClick={onSelect}
            >
              {isSelected ? (
                <>
                  <Check className="h-4 w-4 mr-1" />
                  Selected
                </>
              ) : (
                "Select"
              )}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={onRemove}
              className="text-muted-foreground hover:text-destructive"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </>
        )}
      </div>
    </div>
  )
}
